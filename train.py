"""
AlphaZero self-play training — optimized for RTX 5090 (CUDA).

Workflow:
  1. [Local]  python prepare_dataset.py gm_games.pgn   → data/*.npy
  2. [Server] python main.py                            → pretrain + self-play

main.py calls this file directly. You can also run it standalone:
  python train.py                 # self-play only (skips pretrain if model exists)
  python train.py --skip-pretrain # force skip pre-training
"""

import os
import sys
import time
import json
import math
import random
import signal

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from config import (
    C_PUCT, TEMP_THRESHOLD,
    NUM_ITERATIONS, GAMES_PER_ITERATION, EPOCHS_PER_ITERATION,
    BATCH_SIZE, LEARNING_RATE_START, LEARNING_RATE_END, WEIGHT_DECAY, BUFFER_SIZE,
    MODEL_DIR, STATS_DIR,
    RANDOM_OPENING_MOVES, FRAC_GAMES_VS_PREVIOUS, OPPONENT_POOL_SIZE,
    ADJUDICATE_MATERIAL_START, ADJUDICATE_MATERIAL_END, ADJUDICATE_FADE_ITERATIONS,
    MAX_GAME_MOVES, REPETITION_PENALTY,
    MCTS_SIMS_TRAINING_SCHEDULE,
    BENCHMARK_GAMES, BENCHMARK_MCTS_SIMS, BENCHMARK_WIN_THRESHOLD, BENCHMARK_CONSECUTIVE_WINS,
    EVAL_V1_GAMES, EVAL_V1_MCTS_SIMS,
    STOCKFISH_PATH, STOCKFISH_EVAL_DEPTH, STOCKFISH_EVAL_MCTS,
    PRETRAIN_BOARDS, PRETRAIN_MOVES, PRETRAIN_VALUES,
)
from model import AlphaZeroNet
from mcts import encode_board, mcts_search, get_mcts_policy, select_move


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
        # Enable TF32 for Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return dev
    print("WARNING: CUDA not available, falling back to CPU")
    return torch.device("cpu")


# ============================================================
# Schedules
# ============================================================

def get_mcts_sims(iteration):
    schedule = MCTS_SIMS_TRAINING_SCHEDULE
    if iteration <= schedule[0][0]:
        return schedule[0][1]
    if iteration >= schedule[-1][0]:
        return schedule[-1][1]
    for i in range(len(schedule) - 1):
        it_a, sims_a = schedule[i]
        it_b, sims_b = schedule[i + 1]
        if it_a <= iteration <= it_b:
            t = (iteration - it_a) / max(1, it_b - it_a)
            return int(sims_a + t * (sims_b - sims_a))
    return schedule[-1][1]


def get_learning_rate(iteration, total_iterations):
    progress = min(1.0, (iteration - 1) / max(1, total_iterations - 1))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return LEARNING_RATE_END + (LEARNING_RATE_START - LEARNING_RATE_END) * cosine


# ============================================================
# Self-Play
# ============================================================

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _play_random_opening(board, num_moves):
    for _ in range(num_moves):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        board.push(random.choice(legal))


def _material_balance(board):
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            balance += val if piece.color == chess.WHITE else -val
    return balance


def _get_adjudication_threshold(iteration):
    progress = min(1.0, max(0, iteration - 1) / max(1, ADJUDICATE_FADE_ITERATIONS))
    return ADJUDICATE_MATERIAL_START + progress * (ADJUDICATE_MATERIAL_END - ADJUDICATE_MATERIAL_START)


def _check_adjudication(board, threshold):
    balance = _material_balance(board)
    if balance >= threshold:
        return chess.WHITE
    elif balance <= -threshold:
        return chess.BLACK
    return None


def self_play_game(model, device, mcts_sims, opponent_model=None, iteration=1):
    board = chess.Board()
    history = []
    move_count = 0
    adjudicated_winner = None

    if RANDOM_OPENING_MOVES > 0:
        num_random = random.randint(0, RANDOM_OPENING_MOVES)
        _play_random_opening(board, num_random)
        move_count = len(board.move_stack)

    repetition_draw = False

    while not board.is_game_over():
        if opponent_model is not None and board.turn == chess.BLACK:
            active_model = opponent_model
        else:
            active_model = model

        root = mcts_search(board, active_model, device, mcts_sims, C_PUCT, add_noise=True)
        policy = get_mcts_policy(root, board)
        encoded = encode_board(board)
        history.append((encoded, policy, board.turn))

        temp = 1.0 if move_count < TEMP_THRESHOLD else 0
        move = select_move(root, temp)
        board.push(move)
        move_count += 1

        if board.can_claim_threefold_repetition():
            repetition_draw = True
            break

        adj_threshold = _get_adjudication_threshold(iteration)
        if move_count > 20:
            adjudicated_winner = _check_adjudication(board, adj_threshold)
            if adjudicated_winner is not None:
                break

        if move_count >= MAX_GAME_MOVES:
            break

    outcome = board.outcome()
    training_data = []
    for encoded, policy, turn in history:
        if adjudicated_winner is not None:
            value = 1.0 if adjudicated_winner == turn else -1.0
        elif repetition_draw:
            value = -REPETITION_PENALTY
        elif outcome is not None and outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            value = -REPETITION_PENALTY
        elif outcome is None or outcome.winner is None:
            value = 0.0
        elif outcome.winner == turn:
            value = 1.0
        else:
            value = -1.0
        training_data.append((encoded, policy, value))

    return training_data, outcome, move_count, adjudicated_winner, repetition_draw


# ============================================================
# Benchmark
# ============================================================

def benchmark_match(model, opponent_model, device, num_games=BENCHMARK_GAMES, mcts_sims=BENCHMARK_MCTS_SIMS):
    wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = chess.Board()
        model_is_white = (game_idx % 2 == 0)
        move_count = 0

        if RANDOM_OPENING_MOVES > 0:
            num_random = random.randint(0, RANDOM_OPENING_MOVES)
            _play_random_opening(board, num_random)
            move_count = len(board.move_stack)

        game_decided = False
        while not board.is_game_over() and move_count < MAX_GAME_MOVES:
            is_model_turn = (board.turn == chess.WHITE) == model_is_white
            active_model = model if is_model_turn else opponent_model

            root = mcts_search(board, active_model, device, mcts_sims, C_PUCT, add_noise=False)
            move = select_move(root, temperature=0)
            board.push(move)
            move_count += 1

            if board.can_claim_threefold_repetition():
                draws += 1
                game_decided = True
                break

            if move_count > 20:
                adj = _check_adjudication(board, 10)
                if adj is not None:
                    if (adj == chess.WHITE) == model_is_white:
                        wins += 1
                    game_decided = True
                    break

        if not game_decided:
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                draws += 1
            elif (outcome.winner == chess.WHITE) == model_is_white:
                wins += 1

    win_rate = (wins + 0.5 * draws) / num_games
    return win_rate, wins, draws, num_games - wins - draws


def stockfish_survival(model, device):
    import chess.engine
    if not os.path.exists(STOCKFISH_PATH):
        return None

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"  Stockfish not available: {e}")
        return None

    board = chess.Board()
    move_count = 0
    try:
        while not board.is_game_over() and move_count < MAX_GAME_MOVES:
            if board.turn == chess.WHITE:
                root = mcts_search(board, model, device, STOCKFISH_EVAL_MCTS, C_PUCT, add_noise=False)
                move = select_move(root, temperature=0)
            else:
                result = engine.play(board, chess.engine.Limit(depth=STOCKFISH_EVAL_DEPTH))
                move = result.move
            board.push(move)
            move_count += 1
            if board.can_claim_threefold_repetition():
                break
    finally:
        engine.quit()

    return move_count


# ============================================================
# Replay Buffer
# ============================================================

BUFFER_PATH = os.path.join(MODEL_DIR, "replay_buffer.pt")


def save_buffer(replay_buffer):
    torch.save(replay_buffer, BUFFER_PATH)


def load_buffer():
    if os.path.exists(BUFFER_PATH):
        try:
            buf = torch.load(BUFFER_PATH, weights_only=False)
            print(f"Loaded replay buffer: {len(buf):,} positions")
            return buf
        except Exception as e:
            print(f"Warning: could not load replay buffer ({e}), starting fresh")
    return []


# ============================================================
# Training
# ============================================================

def train_epoch(model, optimizer, scaler, replay_buffer, device):
    model.train()
    data = list(replay_buffer)
    np.random.shuffle(data)

    total_p_loss = 0.0
    total_v_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        if len(batch) < 16:
            continue

        boards, policies, values = zip(*batch)
        boards_t   = torch.FloatTensor(np.array(boards)).to(device, non_blocking=True)
        policies_t = torch.FloatTensor(np.array(policies)).to(device, non_blocking=True)
        values_t   = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            pred_p, pred_v = model(boards_t)
            policy_loss = -(policies_t * F.log_softmax(pred_p, dim=1)).sum(dim=1).mean()
            value_loss  = F.mse_loss(pred_v, values_t)
            loss = policy_loss + value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_p_loss += policy_loss.item()
        total_v_loss += value_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0
    return total_p_loss / num_batches, total_v_loss / num_batches


# ============================================================
# Supervised Pre-training on .npy dataset
# ============================================================

def pretrain(model, optimizer, scaler, device, epochs=6):
    """Train on pre-computed .npy dataset. Called once at startup if no model exists."""

    if not os.path.exists(PRETRAIN_BOARDS):
        print("No pre-training data found (data/boards.npy missing). Skipping pretrain.")
        print("Run prepare_dataset.py locally to generate the dataset.\n")
        return

    import struct

    # Detect count from file size
    board_bytes = os.path.getsize(PRETRAIN_BOARDS)
    # .npy files have a header — load with numpy
    boards = np.load(PRETRAIN_BOARDS, mmap_mode='r')
    moves  = np.load(PRETRAIN_MOVES,  mmap_mode='r')
    values = np.load(PRETRAIN_VALUES, mmap_mode='r')
    n = len(boards)

    print(f"\n{'='*60}")
    print(f"  PRE-TRAINING on {n:,} positions ({epochs} epochs)")
    print(f"{'='*60}\n")

    indices = np.arange(n)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_p = 0.0
        total_v = 0.0
        num_batches = 0
        t0 = time.time()

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            idx = indices[start:end]

            boards_t = torch.FloatTensor(boards[idx]).to(device, non_blocking=True)
            moves_t  = torch.LongTensor(moves[idx]).to(device, non_blocking=True)
            values_t = torch.FloatTensor(values[idx]).unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                policy_logits, pred_v = model(boards_t)
                policy_loss = F.cross_entropy(policy_logits, moves_t)
                value_loss  = F.mse_loss(pred_v, values_t)
                loss = policy_loss + value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_p += policy_loss.item()
            total_v += value_loss.item()
            num_batches += 1

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{epochs} — "
              f"policy: {total_p/num_batches:.4f}  "
              f"value: {total_v/num_batches:.4f}  "
              f"({elapsed:.0f}s)")

    print()


# ============================================================
# Save / Load / Cleanup
# ============================================================

def save_version(model, optimizer, iteration, stats, loss_history):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"version_{iteration}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'stats': {
            'policy_loss': stats.get('policy_loss', 0),
            'value_loss': stats.get('value_loss', 0),
            'results': stats.get('results', {}),
        }
    }, path)

    meta = {
        'iteration': iteration,
        'policy_loss': stats.get('policy_loss', 0),
        'value_loss': stats.get('value_loss', 0),
        'results': stats.get('results', {}),
        'loss_history': loss_history[-50:],   # keep last 50 for size
    }
    meta_path = os.path.join(MODEL_DIR, f"version_{iteration}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def list_versions():
    if not os.path.exists(MODEL_DIR):
        return []
    versions = []
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("version_") and fname.endswith(".pt"):
            try:
                v = int(fname.replace("version_", "").replace(".pt", ""))
                versions.append(v)
            except ValueError:
                pass
    return sorted(versions)


def cleanup_old_versions(keep_last=5):
    versions = list_versions()
    if len(versions) <= keep_last:
        return
    protected = set(versions[-keep_last:])
    for v in versions:
        if v in protected or v == 1 or v % 10 == 0:
            continue
        for suffix in [".pt", "_meta.json"]:
            p = os.path.join(MODEL_DIR, f"version_{v}{suffix}")
            if os.path.exists(p):
                os.remove(p)


def load_version(version, device):
    path = os.path.join(MODEL_DIR, f"version_{version}.pt")
    model = AlphaZeroNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('stats', {})


def save_iteration_stats(iteration, stats_data):
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, f"iter_{iteration:04d}.json")
    with open(path, 'w') as f:
        json.dump(stats_data, f, indent=2)


# ============================================================
# Main Training Loop
# ============================================================

def main(skip_pretrain=False):
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    model = AlphaZeroNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_START, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()   # mixed precision

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    versions = list_versions()
    start_iteration = 1
    replay_buffer = load_buffer()
    loss_history = []

    if versions:
        latest = max(versions)
        print(f"Resuming from version {latest}...")
        checkpoint = torch.load(
            os.path.join(MODEL_DIR, f"version_{latest}.pt"),
            map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = latest + 1
    elif not skip_pretrain:
        # First run — pretrain on GM data if available
        pretrain(model, optimizer, scaler, device, epochs=6)
        # Save as version 1
        save_version(model, optimizer, 1, {}, [])
        print(f"Saved version 1 (pre-trained)\n")
        start_iteration = 2

    benchmark_opponent_version = 1
    benchmark_consecutive_wins = 0

    print(f"Config: {GAMES_PER_ITERATION} games/iter, {EPOCHS_PER_ITERATION} epochs/iter")
    print(f"LR: {LEARNING_RATE_START} → {LEARNING_RATE_END} (cosine)")
    print(f"Starting from iteration {start_iteration}\n")

    # Graceful shutdown on Ctrl+C — saves current state
    _stop = [False]
    def _handle_sigint(sig, frame):
        print("\n\nCtrl+C received — finishing current iteration then stopping...")
        _stop[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        for iteration in range(start_iteration, start_iteration + NUM_ITERATIONS):
            if _stop[0]:
                break

            mcts_sims = get_mcts_sims(iteration)
            lr = get_learning_rate(iteration, NUM_ITERATIONS)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            print(f"{'='*60}")
            print(f"  ITERATION {iteration}  |  MCTS: {mcts_sims}  |  LR: {lr:.6f}")
            print(f"{'='*60}")

            # --- Self-Play Phase ---
            model.eval()
            iteration_data = []
            results = {'white': 0, 'black': 0, 'draw': 0}
            terminations = {}
            game_lengths = []
            iter_start = time.time()
            game_times = []

            # Load opponent for fraction of games
            opponent_model = None
            opp_version = None
            num_vs_previous = int(GAMES_PER_ITERATION * FRAC_GAMES_VS_PREVIOUS)
            available_versions = list_versions()
            if available_versions:
                pool = available_versions[-min(OPPONENT_POOL_SIZE, len(available_versions)):]
                opp_version = random.choice(pool)
                opponent_model, _ = load_version(opp_version, device)
                print(f"  {num_vs_previous} games vs version {opp_version}")
            else:
                num_vs_previous = 0

            adj_thresh = _get_adjudication_threshold(iteration)
            print(f"  Adjudication threshold: {adj_thresh:.0f}")

            for game_num in range(1, GAMES_PER_ITERATION + 1):
                game_start = time.time()
                use_opponent = opponent_model is not None and game_num <= num_vs_previous

                data, outcome, num_moves, adj_winner, rep_draw = self_play_game(
                    model, device, mcts_sims,
                    opponent_model=opponent_model if use_opponent else None,
                    iteration=iteration
                )

                iteration_data.extend(data)
                game_lengths.append(num_moves)
                elapsed = time.time() - game_start
                game_times.append(elapsed)

                winner = None
                if outcome and outcome.winner is not None:
                    winner = outcome.winner
                    term = outcome.termination.name
                elif adj_winner is not None:
                    winner = adj_winner
                    term = "ADJUDICATION"
                elif rep_draw:
                    term = "THREEFOLD_REPETITION"
                elif outcome:
                    term = outcome.termination.name
                else:
                    term = "MAX_MOVES"

                terminations[term] = terminations.get(term, 0) + 1

                if winner == chess.WHITE:
                    results['white'] += 1
                    result_str = "1-0"
                elif winner == chess.BLACK:
                    results['black'] += 1
                    result_str = "0-1"
                else:
                    results['draw'] += 1
                    result_str = "1/2"

                mode_str = f" [vs v{opp_version}]" if use_opponent else ""
                print(f"  Game {game_num:3d}/{GAMES_PER_ITERATION} | "
                      f"{result_str} | {num_moves:3d} moves | {term:22s} | "
                      f"{elapsed:.1f}s{mode_str}")

            # Add to replay buffer
            replay_buffer.extend(iteration_data)
            if len(replay_buffer) > BUFFER_SIZE:
                replay_buffer = replay_buffer[-BUFFER_SIZE:]
            save_buffer(replay_buffer)

            total_games = sum(results.values())
            sp_elapsed = time.time() - iter_start
            print(f"\n  Self-play done: {len(iteration_data):,} positions in {sp_elapsed:.0f}s")
            print(f"  W:{results['white']} B:{results['black']} D:{results['draw']} "
                  f"| Avg length: {np.mean(game_lengths):.0f} moves")

            # --- Training Phase ---
            print(f"\n  Training on {len(replay_buffer):,} positions...")
            model.train()
            p_loss = v_loss = 0.0
            for epoch in range(1, EPOCHS_PER_ITERATION + 1):
                p_loss, v_loss = train_epoch(model, optimizer, scaler, replay_buffer, device)
                print(f"    Epoch {epoch:2d}/{EPOCHS_PER_ITERATION} | "
                      f"P: {p_loss:.4f}  V: {v_loss:.4f}  Total: {p_loss+v_loss:.4f}")

            loss_history.append((iteration, p_loss, v_loss))

            stats = {
                'policy_loss': p_loss,
                'value_loss': v_loss,
                'results': results,
            }

            # --- Save ---
            save_version(model, optimizer, iteration, stats, loss_history)
            print(f"\n  Saved version {iteration}")

            # --- Benchmark ---
            if benchmark_opponent_version > 1 and benchmark_opponent_version in list_versions():
                print(f"\n  Benchmark: {BENCHMARK_GAMES} games vs v{benchmark_opponent_version}...")
                model.eval()
                bm_opp, _ = load_version(benchmark_opponent_version, device)
                win_rate, bm_w, bm_d, bm_l = benchmark_match(model, bm_opp, device)
                del bm_opp
                print(f"  Result: {win_rate:.0%} (W:{bm_w} D:{bm_d} L:{bm_l})")

                if win_rate >= BENCHMARK_WIN_THRESHOLD:
                    benchmark_consecutive_wins += 1
                    if benchmark_consecutive_wins >= BENCHMARK_CONSECUTIVE_WINS:
                        next_opp = benchmark_opponent_version + 10
                        if next_opp in list_versions():
                            benchmark_opponent_version = next_opp
                            benchmark_consecutive_wins = 0
                            print(f"  >>> Advancing benchmark to v{benchmark_opponent_version}")
                else:
                    benchmark_consecutive_wins = 0

            # --- V1 Eval ---
            v1_win_rate = None
            if 1 in list_versions() and iteration > 1:
                print(f"\n  Eval vs v1: {EVAL_V1_GAMES} games...")
                model.eval()
                v1_model, _ = load_version(1, device)
                v1_wr, v1_w, v1_d, v1_l = benchmark_match(
                    model, v1_model, device,
                    num_games=EVAL_V1_GAMES, mcts_sims=EVAL_V1_MCTS_SIMS
                )
                del v1_model
                v1_win_rate = v1_wr
                print(f"  vs v1: {v1_wr:.0%} (W:{v1_w} D:{v1_d} L:{v1_l})")

                if benchmark_opponent_version == 1:
                    if v1_wr >= BENCHMARK_WIN_THRESHOLD:
                        benchmark_consecutive_wins += 1
                    else:
                        benchmark_consecutive_wins = 0
                    if benchmark_consecutive_wins >= BENCHMARK_CONSECUTIVE_WINS:
                        if 10 in list_versions():
                            benchmark_opponent_version = 10
                            benchmark_consecutive_wins = 0
                            print(f"  >>> Advancing benchmark to v10")

            # --- Stockfish (optional) ---
            sf_moves = None
            if os.path.exists(STOCKFISH_PATH):
                print(f"\n  Stockfish survival test...")
                model.eval()
                sf_moves = stockfish_survival(model, device)
                if sf_moves is not None:
                    print(f"  Survived {sf_moves} moves")

            # --- Stats ---
            iter_elapsed = time.time() - iter_start
            save_iteration_stats(iteration, {
                'iteration': iteration,
                'mcts_sims': mcts_sims,
                'learning_rate': lr,
                'policy_loss': p_loss,
                'value_loss': v_loss,
                'total_loss': p_loss + v_loss,
                'results': results,
                'terminations': terminations,
                'avg_game_length': float(np.mean(game_lengths)),
                'total_positions': len(iteration_data),
                'buffer_size': len(replay_buffer),
                'cycle_time_seconds': iter_elapsed,
                'avg_game_time_seconds': float(np.mean(game_times)),
                'v1_win_rate': v1_win_rate,
                'stockfish_survival_moves': sf_moves,
            })

            cleanup_old_versions(keep_last=OPPONENT_POOL_SIZE)
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"\n  Iteration {iteration} done in {iter_elapsed:.0f}s "
                  f"| GPU mem: {gpu_mem:.1f} GB\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise

    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()
    main(skip_pretrain=args.skip_pretrain)
