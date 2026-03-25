from pathlib import Path

# Project root is one level above `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_MODEL_DIR = PROJECT_ROOT / "custom_vit_model"
# Final fine-tuned weights (HF format): training writes here; app.py / main.py load from here.
CHECKPOINT_FINAL_DIR = PROJECT_ROOT / "models" / "checkpoint-final"
TRAINED_MODEL_DIR = CHECKPOINT_FINAL_DIR
# Training metrics exports (historical snapshot moved under archive/; new runs use this path).
RESULTS_DIR = PROJECT_ROOT / "archive" / "results"
