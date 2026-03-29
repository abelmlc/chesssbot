import math
import chess
import numpy as np
import torch
import torch.nn.functional as F

from config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, POLICY_SIZE, MCTS_BATCH_SIZE

# ============================================================
# Board Encoding (19 planes of 8x8)
# ============================================================

def encode_board(board):
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK

    for piece_type in range(1, 7):
        for sq in chess.scan_forward(board.pieces_mask(piece_type, board.turn)):
            if flip:
                sq = chess.square_mirror(sq)
            planes[piece_type - 1][chess.square_rank(sq)][chess.square_file(sq)] = 1.0
        for sq in chess.scan_forward(board.pieces_mask(piece_type, not board.turn)):
            if flip:
                sq = chess.square_mirror(sq)
            planes[piece_type + 5][chess.square_rank(sq)][chess.square_file(sq)] = 1.0

    my = board.turn
    opp = not board.turn
    planes[12][:, :] = float(board.has_kingside_castling_rights(my))
    planes[13][:, :] = float(board.has_queenside_castling_rights(my))
    planes[14][:, :] = float(board.has_kingside_castling_rights(opp))
    planes[15][:, :] = float(board.has_queenside_castling_rights(opp))

    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if flip else board.ep_square
        planes[16][chess.square_rank(ep)][chess.square_file(ep)] = 1.0

    planes[17][:, :] = board.halfmove_clock / 100.0
    planes[18][:, :] = board.fullmove_number / 200.0

    return planes


# ============================================================
# Move Encoding (4672 = 73 planes x 64 from-squares)
# ============================================================

QUEEN_DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]
KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]
_KNIGHT_MOVE_INDEX = {m: i for i, m in enumerate(KNIGHT_MOVES)}
_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}


def encode_move(move, turn):
    from_sq = move.from_square
    to_sq = move.to_square

    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - from_row
    dc = chess.square_file(to_sq) - from_col

    if move.promotion and move.promotion != chess.QUEEN:
        dir_idx = dc + 1
        piece_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        plane = 64 + dir_idx * 3 + piece_map[move.promotion]
    else:
        knight_idx = _KNIGHT_MOVE_INDEX.get((dr, dc))
        if knight_idx is not None:
            plane = 56 + knight_idx
        else:
            sign = lambda x: (1 if x > 0 else -1 if x < 0 else 0)
            direction = _QUEEN_DIR_INDEX[(sign(dr), sign(dc))]
            distance = max(abs(dr), abs(dc))
            plane = direction * 7 + (distance - 1)

    return plane * 64 + from_row * 8 + from_col


# ============================================================
# MCTS with Batched NN Evaluation
# ============================================================

class MCTSNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children']

    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _evaluate_single(board, model, device):
    encoded = encode_board(board)
    tensor = torch.FloatTensor(encoded).unsqueeze(0).to(device)
    with torch.inference_mode():
        policy_logits, value = model(tensor)
    policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
    return policy, value.item()


def _evaluate_batch(boards_encoded, model, device):
    tensor = torch.FloatTensor(np.array(boards_encoded)).to(device)
    with torch.inference_mode():
        policy_logits, values = model(tensor)
    policies = F.softmax(policy_logits, dim=1).cpu().numpy()
    values = values.cpu().numpy().flatten()
    return policies, values


def _select_child(node, c_puct):
    best_score = -float('inf')
    best_move = None
    best_child = None
    sqrt_parent = math.sqrt(max(node.visit_count, 1))

    for move, child in node.children.items():
        q = -child.q_value
        u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    return best_move, best_child


def _get_legal_policy(policy, board):
    move_priors = {}
    for move in board.legal_moves:
        idx = encode_move(move, board.turn)
        move_priors[move] = max(policy[idx], 1e-8)
    total = sum(move_priors.values())
    for m in move_priors:
        move_priors[m] /= total
    return move_priors


def mcts_search(board, model, device, num_sims, c_puct=C_PUCT, add_noise=True,
                batch_size=MCTS_BATCH_SIZE):
    root = MCTSNode(prior=0.0)
    policy, _ = _evaluate_single(board, model, device)
    move_priors = _get_legal_policy(policy, board)

    if add_noise and move_priors:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(move_priors))
        for i, move in enumerate(move_priors):
            move_priors[move] = (
                (1 - DIRICHLET_EPSILON) * move_priors[move]
                + DIRICHLET_EPSILON * noise[i]
            )

    for move, prior in move_priors.items():
        root.children[move] = MCTSNode(prior=prior)

    sim = 0
    while sim < num_sims:
        batch_leaves = []
        batch_terminals = []
        claimed = set()

        current_batch = min(batch_size, num_sims - sim)

        for _ in range(current_batch):
            node = root
            scratch = board.copy(stack=False)
            path = [node]

            while node.children:
                move, child = _select_child(node, c_puct)
                scratch.push(move)
                node = child
                path.append(node)

            if scratch.is_game_over():
                outcome = scratch.outcome()
                if outcome.winner is None:
                    value = 0.0
                elif outcome.winner == scratch.turn:
                    value = 1.0
                else:
                    value = -1.0
                batch_terminals.append((path, value))
            elif id(node) in claimed:
                batch_terminals.append((path, 0.0))
            else:
                claimed.add(id(node))
                batch_leaves.append((path, scratch, node))
                for n in path:
                    n.visit_count += 1
                    n.value_sum -= 1.0

            sim += 1

        if batch_leaves:
            encoded = [encode_board(b) for _, b, _ in batch_leaves]
            policies, values = _evaluate_batch(encoded, model, device)

            for i, (path, scratch, node) in enumerate(batch_leaves):
                for n in path:
                    n.visit_count -= 1
                    n.value_sum += 1.0

                legal_priors = _get_legal_policy(policies[i], scratch)
                for m, p in legal_priors.items():
                    node.children[m] = MCTSNode(prior=p)

                value = values[i]
                for n in reversed(path):
                    n.value_sum += value
                    n.visit_count += 1
                    value = -value

        for path, value in batch_terminals:
            for n in reversed(path):
                n.value_sum += value
                n.visit_count += 1
                value = -value

    return root


def get_mcts_policy(root, board):
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    total = sum(c.visit_count for c in root.children.values())
    if total == 0:
        return policy
    for move, child in root.children.items():
        idx = encode_move(move, board.turn)
        policy[idx] = child.visit_count / total
    return policy


def select_move(root, temperature):
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if temperature == 0 or temperature < 1e-3:
        idx = np.argmax(visits)
    else:
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        idx = np.random.choice(len(moves), p=probs)

    return moves[idx]
