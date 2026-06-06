"""Global hyperparameter configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "model_input_with_raw.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "result"

# ── data ──
SUBSET_FILTER = "train"          # use only this subset for small-batch validation; None = all
SAMPLE_LIMIT = None              # max rows to load for quick validation; None = all
VAL_RATIO = 0.1                  # fraction of training data held out for validation
TARGET_COL = "tau_km"

# feature groups for the multi-branch architecture
# Iteration order matters: physical branch (spatial+solar) comes first,
# temporal branch second — matches model.py's column-slicing logic.
FEATURE_GROUPS = {
    "spatial":  ["proc_sin_lon", "proc_cos_lon", "proc_lat", "proc_mlat"],
    "solar":    ["proc_kp", "proc_f107", "proc_vtec"],
    "temporal": ["proc_sin_lt",  "proc_cos_lt",  "proc_sin_doy", "proc_cos_doy",
                 "proc_cos_chi"],
}

# ── model ──
SPATIAL_DIM = len(FEATURE_GROUPS["spatial"])   # 4
TEMPORAL_DIM = len(FEATURE_GROUPS["temporal"])  # 5
SOLAR_DIM = len(FEATURE_GROUPS["solar"])        # 3

BRANCH_HIDDEN = [128, 256]                     # shared expansion path per branch
FUSION_DIM = 384                                # concat → linear project

# (in_dim, hidden_dim) — in_dim of block N+1 must match hidden_dim of block N
RESIDUAL_BLOCKS = [
    (384, 512),   # in=384, out=512   (skip projects 384→512)
    (512, 512),   # in=512, out=512   (skip is identity)
    (512, 256),   # in=512, out=256   (skip projects 512→256)
    (256, 128),   # in=256, out=128   (skip projects 256→128)
]

HEAD_DIMS = [128, 64, 32, 1]
DROPOUT = 0.15
ACTIVATION = "gelu"

# ── training ──
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0

# scheduler: "cosine" | "plateau"
SCHEDULER = "cosine"
T_0 = 20            # cosine restart period (epochs)
T_MULT = 2
LR_MIN = 1e-6

CKPT_SUFFIX = ""                 # suffix for checkpoint filenames ("full", "v2", etc.)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
