"""
Launch the Gradio UI using fine-tuned weights under ``models/checkpoint-final/``.

The path is resolved relative to this file’s directory (project root), so it works
when the repo is deployed to a cloud instance with the same layout.

For local default (``./trained_model``), use ``python main.py`` instead.
"""

from pathlib import Path

from src.web.app import launch

# Project root = directory containing this file
_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = _ROOT / "models" / "checkpoint-final"


if __name__ == "__main__":
    launch(model_path=CHECKPOINT_DIR)
