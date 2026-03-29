# AlphaZero Chess — GPU Training

Optimized for **RTX 5090** (32GB VRAM, 60GB RAM, 16 vCPU).

## Workflow

### Step 1 — Prepare dataset locally (on your Mac)

```bash
# From your local chessbot/ folder:
python prepare_dataset.py gm_games.pgn --output-dir online_training/data/
```

This converts PGN games into compact `.npy` files (no API calls on the server).

### Step 2 — Push to GitHub

```bash
git add online_training/
git commit -m "add online_training"
git push
```

> Make sure `data/` is NOT in `.gitignore` — the `.npy` files need to go up,
> or upload them separately with `scp`.

### Step 3 — Setup server

```bash
git clone https://github.com/youruser/yourrepo
cd yourrepo/online_training

# Install PyTorch (CUDA 12.x for RTX 5090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install python-chess numpy

# Optional: Stockfish
apt install stockfish
```

### Step 4 — Train

```bash
python main.py
```

That's it. The script will:
1. Detect the GPU
2. Pre-train on `data/*.npy` if no model exists yet
3. Run self-play training indefinitely (Ctrl+C to stop cleanly)

### Resume after interruption

```bash
python main.py   # automatically resumes from latest saved version
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — just run this |
| `train.py` | Self-play loop + supervised pretrain |
| `model.py` | AlphaZeroNet (19 res blocks, 256 filters) |
| `mcts.py` | Board encoding + batched MCTS |
| `config.py` | All hyperparameters |
| `prepare_dataset.py` | **Run locally** — converts PGN → .npy |
| `data/` | Pre-computed training data (boards/moves/values .npy) |
| `models/` | Saved model versions |
| `stats/` | Per-iteration JSON stats |

## GPU optimizations

- **Mixed precision** (FP16) via `torch.cuda.amp` — ~2x faster training
- **TF32** enabled for matmul and cuDNN
- **cuDNN benchmark** mode for fixed-size inputs
- **Large batch size** (2048) to saturate VRAM
- **Large MCTS batch** (256) for GPU-efficient inference
- **Large replay buffer** (500k positions) fitting in 60GB RAM
- **non_blocking transfers** for CPU→GPU overlap
