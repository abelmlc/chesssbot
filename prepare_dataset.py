"""
Prepare a pre-training dataset from PGN files.

Run this LOCALLY (before uploading to the server) to convert PGN games
into compact .npy files ready for supervised pre-training.

Usage:
    python prepare_dataset.py gm_games.pgn
    python prepare_dataset.py gm_games.pgn --output-dir my_data/
    python prepare_dataset.py *.pgn --max-games 50000

Output:
    data/boards.npy   — float32 (N, 19, 8, 8)
    data/moves.npy    — int64   (N,)
    data/values.npy   — float32 (N,)
    data/meta.json    — game count, position count

Upload the data/ folder to your server alongside online_training/.
"""

import argparse
import os
import struct
import json
import chess
import chess.pgn
import numpy as np

# Inline board/move encoding (no dependency on other modules)

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


def parse_pgn(pgn_paths, output_dir, max_games=None):
    os.makedirs(output_dir, exist_ok=True)

    boards_path = os.path.join(output_dir, "boards.npy")
    moves_path  = os.path.join(output_dir, "moves.npy")
    values_path = os.path.join(output_dir, "values.npy")
    meta_path   = os.path.join(output_dir, "meta.json")

    idx = 0
    game_count = 0

    with open(boards_path, 'wb') as fb, \
         open(moves_path,  'wb') as fm, \
         open(values_path, 'wb') as fv:

        for pgn_path in pgn_paths:
            print(f"Parsing {pgn_path}...")
            with open(pgn_path) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    result = game.headers.get("Result", "*")
                    if result == "1-0":
                        white_value = 1.0
                    elif result == "0-1":
                        white_value = -1.0
                    elif result == "1/2-1/2":
                        white_value = 0.0
                    else:
                        continue

                    board = game.board()
                    for move in game.mainline_moves():
                        value = white_value if board.turn == chess.WHITE else -white_value
                        fb.write(encode_board(board).tobytes())
                        fm.write(struct.pack('<q', encode_move(move, board.turn)))
                        fv.write(struct.pack('<f', value))
                        idx += 1
                        board.push(move)

                    game_count += 1
                    if game_count % 500 == 0:
                        print(f"  {game_count} games, {idx:,} positions...", flush=True)

                    if max_games and game_count >= max_games:
                        break

            if max_games and game_count >= max_games:
                break

    if idx == 0:
        print("No positions found!")
        return

    # Convert raw binary to proper .npy files
    print(f"\nFinalizing .npy files ({idx:,} positions)...")

    boards_raw = np.frombuffer(open(boards_path, 'rb').read(), dtype=np.float32).reshape(idx, 19, 8, 8)
    moves_raw  = np.frombuffer(open(moves_path,  'rb').read(), dtype=np.int64)
    values_raw = np.frombuffer(open(values_path, 'rb').read(), dtype=np.float32)

    np.save(boards_path, boards_raw)
    np.save(moves_path,  moves_raw)
    np.save(values_path, values_raw)

    with open(meta_path, 'w') as f:
        json.dump({'games': game_count, 'positions': idx}, f, indent=2)

    print(f"\nDone!")
    print(f"  Games:     {game_count:,}")
    print(f"  Positions: {idx:,}")
    print(f"  Output:    {output_dir}/")
    print(f"  boards.npy  {os.path.getsize(boards_path) / 1e6:.0f} MB")
    print(f"  moves.npy   {os.path.getsize(moves_path)  / 1e6:.0f} MB")
    print(f"  values.npy  {os.path.getsize(values_path) / 1e6:.0f} MB")
    print(f"\nUpload the '{output_dir}/' folder to your server's online_training/ directory.")


def main():
    parser = argparse.ArgumentParser(description="Convert PGN games to .npy dataset for server training")
    parser.add_argument("pgn_files", nargs="+", help="PGN file(s) to process")
    parser.add_argument("--output-dir", "-o", default="data", help="Output directory (default: data/)")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")
    args = parser.parse_args()

    parse_pgn(args.pgn_files, args.output_dir, max_games=args.max_games)


if __name__ == "__main__":
    main()
