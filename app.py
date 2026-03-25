import os
from pathlib import Path
from src.web.app import launch

# Project root
_ROOT = Path(__file__).resolve().parent
# Force this to be a string for compatibility with Transformers
CHECKPOINT_DIR = str(_ROOT / "models" / "checkpoint-final")

if __name__ == "__main__":
    # DEBUG: Let's see if the files are actually where we think they are
    if os.path.exists(CHECKPOINT_DIR):
        print(f"✅ Found directory: {CHECKPOINT_DIR}")
        print(f"📁 Files inside: {os.listdir(CHECKPOINT_DIR)}")
    else:
        print(f"❌ DIRECTORY NOT FOUND: {CHECKPOINT_DIR}")
        # Let's see what IS in the models folder
        models_path = str(_ROOT / "models")
        if os.path.exists(models_path):
            print(f"📂 Contents of /models/: {os.listdir(models_path)}")
    
    launch(model_path=CHECKPOINT_DIR)