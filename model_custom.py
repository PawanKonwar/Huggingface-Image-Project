"""
Backward-compatible wrapper for creating the custom ViT model.

Use:
  python model_custom.py
"""

from src.models.model_custom import create_custom_model, get_class_names_from_data


if __name__ == "__main__":
    class_names = get_class_names_from_data()
    create_custom_model(class_names)
  