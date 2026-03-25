"""
Launch the Gradio UI using fine-tuned weights under ``models/checkpoint-final/``.

Path is resolved from ``src.utils.paths`` (project root + ``models/checkpoint-final``),
so it works locally and on Hugging Face Spaces with the same repo layout.
"""

from src.utils.paths import CHECKPOINT_FINAL_DIR
from src.web.app import launch


if __name__ == "__main__":
    launch(model_path=CHECKPOINT_FINAL_DIR)
