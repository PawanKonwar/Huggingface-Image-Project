from pathlib import Path

# Project root is one level above `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_MODEL_DIR = PROJECT_ROOT / "custom_vit_model"
TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_model"

