"""
Project 1 Config:
-> adjust tunable parameters here
"""

from pathlib import Path

#paths
DATA_DIR   = Path(__file__).parent / "data"
SAVE_PATH  = "best_model.pt"
LOG_DIR    = "runs/experiment"

#data
IMG_SIZE   = 224
VAL_RATIO  = 0.15        # fraction of data held out for validation
SEED       = 42

#model
NUM_CLASSES  = 10
HIDDEN_SIZE  = 256        # TODO: test 512, 1024 for potential accuracy gains

#train
BATCH_SIZE   = 32
N_EPOCHS     = 20
LR           = 3e-4
NUM_WORKERS  = 4          # increase toward CPU core count for faster loading