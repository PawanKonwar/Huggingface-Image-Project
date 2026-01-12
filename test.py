"""
Test script for the trained Vision Transformer model.
"""

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from pathlib import Path
import argparse

def test_image(image_path, model_path='./trained_model'):
    """Test a single image."""
    
    print("="*60)
    print("Testing Trained Model")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    
    class_names = [model.config.id2label[i] for i in range(5)]
    print(f"Classes: {class_names}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    print(f"\nLoading image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get predictions
    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_idx]
    confidence = probabilities[predicted_idx].item()
    
    # Display results
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Image: {Path(image_path).name}")
    print(f"\nPredicted: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    
    print(f"\nAll predictions:")
    all_probs = []
    for idx in range(5):
        label = model.config.id2label[idx]
        prob = probabilities[idx].item()
        all_probs.append((label, prob))
    
    all_probs.sort(key=lambda x: x[1], reverse=True)
    for i, (label, prob) in enumerate(all_probs, 1):
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {label:15s} {prob:6.2%} {bar}")
    
    print("-"*60)


def test_directory(image_dir, model_path='./trained_model'):
    """Test all images in a directory."""
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in extensions]
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    print("="*60)
    
    # Load model once
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    class_names = [model.config.id2label[i] for i in range(5)]
    
    # Test each image
    for i, image_path in enumerate(image_paths, 1):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predicted_idx = outputs.logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_idx]
            confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][predicted_idx].item()
            
            print(f"{i}. {image_path.name:30s} → {predicted_label:15s} ({confidence:.2%})")
        except Exception as e:
            print(f"{i}. {image_path.name:30s} → ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Vision Transformer model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file')
    parser.add_argument('--directory', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--model_path', type=str, default='./trained_model',
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.image:
        test_image(args.image, args.model_path)
    elif args.directory:
        test_directory(args.directory, args.model_path)
    else:
        print("Please provide --image or --directory")
        parser.print_help()

