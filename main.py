"""
Entry point — just run: python main.py

What it does:
  1. Checks GPU is available
  2. If data/boards.npy missing but a .pgn is found in data/ → converts it automatically
  3. If no model exists → pre-trains on the dataset
  4. Launches self-play training loop (runs until Ctrl+C)

Setup on server:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install python-chess numpy
  (optional) apt install stockfish

Put your .pgn file in the data/ folder before running.
"""

import sys
import os


def check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import chess
    except ImportError:
        missing.append("python-chess")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        if "torch" in missing:
            print("\nFor CUDA 12.x (RTX 5090):")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected.")
        print("Training will run on CPU — much slower.")
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != 'y':
            sys.exit(0)
    else:
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        print(f"GPU detected: {props.name} ({vram:.1f} GB VRAM)")
        if vram < 8:
            print("WARNING: Less than 8GB VRAM — consider reducing BATCH_SIZE in config.py")


def convert_pgn_if_needed():
    """If .npy files are missing but a .pgn exists in data/, convert it automatically."""
    if os.path.exists("data/boards.npy"):
        return  # already done

    import glob
    pgn_files = glob.glob("data/*.pgn")
    if not pgn_files:
        return  # nothing to convert

    print(f"Found PGN file(s): {pgn_files}")
    print("Converting to .npy dataset (this runs once)...\n")

    from prepare_dataset import parse_pgn
    parse_pgn(pgn_files, output_dir="data")
    print()


def print_banner():
    print("=" * 60)
    print("  AlphaZero Chess Training — GPU Edition")
    print("=" * 60)

    # Show dataset status
    if os.path.exists("data/boards.npy"):
        import numpy as np
        try:
            n = len(np.load("data/boards.npy", mmap_mode='r'))
            print(f"  Dataset:    data/*.npy  ({n:,} positions)")
        except Exception:
            print(f"  Dataset:    data/*.npy  (found)")
    else:
        print("  Dataset:    NOT FOUND (will skip pretrain)")
        print("              Run prepare_dataset.py locally to generate it")

    from train import list_versions
    versions = list_versions()
    if versions:
        print(f"  Model:      version_{max(versions)}.pt (resuming)")
    else:
        print("  Model:      none (will create from pretrain or scratch)")

    print("=" * 60)
    print()


if __name__ == "__main__":
    check_deps()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip supervised pre-training even if data is available")
    args = parser.parse_args()

    check_gpu()
    convert_pgn_if_needed()
    print_banner()

    from train import main
    main(skip_pretrain=args.skip_pretrain)
