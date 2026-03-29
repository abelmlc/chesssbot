# ============================================================
# Config — optimized for RTX 5090 (32GB VRAM, 60GB RAM, 16 vCPU)
# ============================================================

# --- Network Architecture (full AlphaZero scale) ---
NUM_RES_BLOCKS = 19
NUM_FILTERS = 256
INPUT_PLANES = 19
POLICY_SIZE = 4672           # 73 planes * 8 * 8

# --- MCTS ---
MCTS_SIMS = 800
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD = 8
MCTS_BATCH_SIZE = 256        # Large batch — GPU can handle it

# MCTS sims ramp-up schedule (iteration, sims)
MCTS_SIMS_TRAINING_SCHEDULE = [
    (1,   400),
    (30,  600),
    (100, 800),
    (300, 1200),
]

# --- Self-Play Variety ---
RANDOM_OPENING_MOVES = 4
FRAC_GAMES_VS_PREVIOUS = 0.3
OPPONENT_POOL_SIZE = 5

# --- Adjudication ---
ADJUDICATE_MATERIAL_START = 15
ADJUDICATE_MATERIAL_END = 30
ADJUDICATE_FADE_ITERATIONS = 200
MAX_GAME_MOVES = 300
REPETITION_PENALTY = 0.7

# --- Training ---
NUM_ITERATIONS = 2000
GAMES_PER_ITERATION = 200    # More games per iter thanks to fast GPU
EPOCHS_PER_ITERATION = 10
BATCH_SIZE = 2048            # Large batch for 32GB VRAM
LEARNING_RATE_START = 0.0003
LEARNING_RATE_END = 0.00003
WEIGHT_DECAY = 1e-4
BUFFER_SIZE = 500_000        # Large buffer — 60GB RAM allows it

# --- Benchmark ---
BENCHMARK_GAMES = 50
BENCHMARK_MCTS_SIMS = 100
BENCHMARK_WIN_THRESHOLD = 0.75
BENCHMARK_CONSECUTIVE_WINS = 3

# --- V1 Eval ---
EVAL_V1_GAMES = 100
EVAL_V1_MCTS_SIMS = 100

# --- Stockfish (optional on server) ---
STOCKFISH_PATH = "/usr/games/stockfish"   # apt install stockfish
STOCKFISH_EVAL_DEPTH = 16
STOCKFISH_EVAL_MCTS = 800

# --- Paths ---
MODEL_DIR = "models"
STATS_DIR = "stats"

# --- Pre-processed dataset paths ---
# Place your .npy files here before launching training
PRETRAIN_BOARDS = "data/boards.npy"
PRETRAIN_MOVES  = "data/moves.npy"
PRETRAIN_VALUES = "data/values.npy"
PRETRAIN_COUNT  = 0   # set automatically at runtime — leave at 0
