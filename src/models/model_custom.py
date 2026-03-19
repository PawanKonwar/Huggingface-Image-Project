"""
Custom Vision Transformer model for image classification.
Modifies google/vit-base-patch16-224 from 1000 to N classes (N = number of subfolders in data_dir).
"""

from pathlib import Path

from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch.nn as nn

from src.utils.paths import CUSTOM_MODEL_DIR, DATA_DIR


def get_class_names_from_data(data_dir=None):
    """
    Scan data_dir for subfolders; each subfolder name is a class.
    Returns a sorted list of class names for stable label ordering.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No subfolders found in {data_dir}")
    return class_names


def create_custom_model(class_names, save_path=None):
    """
    Create and save a custom Vision Transformer model with len(class_names) classes.

    Args:
        class_names: List of class names (e.g. from get_class_names_from_data).
        save_path: Directory to save the custom model.

    Returns:
        model, processor
    """
    if save_path is None:
        save_path = CUSTOM_MODEL_DIR
    num_classes = len(class_names)
    if num_classes < 1:
        raise ValueError("class_names must contain at least one class")

    print("Loading base model: google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')

    print(f"Modifying classification head from 1000 to {num_classes} classes...")
    hidden_size = model.config.hidden_size
    model.classifier = nn.Linear(hidden_size, num_classes)

    model.config.num_labels = num_classes
    model.config.id2label = {i: name for i, name in enumerate(class_names)}
    model.config.label2id = {name: i for i, name in enumerate(class_names)}

    print(f"Saving custom model to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print("Custom model created successfully!")
    print(f"Classes ({num_classes}): {class_names}")

    return model, processor


if __name__ == "__main__":
    class_names = get_class_names_from_data(DATA_DIR)
    create_custom_model(class_names, save_path=CUSTOM_MODEL_DIR)
