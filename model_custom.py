"""
Custom Vision Transformer model for 5-class classification.
Modifies google/vit-base-patch16-224 from 1000 to 5 classes.
"""

from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn as nn

def create_custom_model(class_names, save_path='./custom_vit_model'):
    """
    Create and save a custom Vision Transformer model with 5 classes.
    
    Args:
        class_names: List of 5 class names
        save_path: Directory to save the custom model
    
    Returns:
        model, processor
    """
    if len(class_names) != 5:
        raise ValueError("Must provide exactly 5 class names")
    
    print("Loading base model: google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    print("Modifying classification head from 1000 to 5 classes...")
    # Get the embedding dimension
    hidden_size = model.config.hidden_size
    
    # Replace the classifier head
    model.classifier = nn.Linear(hidden_size, 5)
    
    # Update model configuration
    model.config.num_labels = 5
    model.config.id2label = {i: name for i, name in enumerate(class_names)}
    model.config.label2id = {name: i for i, name in enumerate(class_names)}
    
    print(f"Saving custom model to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"Custom model created successfully!")
    print(f"Classes: {class_names}")
    
    return model, processor


if __name__ == "__main__":
    # Your 5 custom classes
    my_classes = [
        'my_cat',
        'my_dog',
        'my_car',
        'my_house',
        'my_phone'
    ]
    
    # Create the custom model
    create_custom_model(my_classes, save_path='./custom_vit_model')

