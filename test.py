"""
Test script for the trained Vision Transformer model.
"""

import argparse
from pathlib import Path

from PIL import Image

from src.api.inference import (
    draw_overlay,
    get_top_k_probs,
    load_model,
    predict,
)

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "trained_model")

UNCERTAINTY_THRESHOLD = 0.90  # Show top-2 when top confidence below this


def test_image(image_path, model_path=DEFAULT_MODEL_PATH, overlay_path='prediction_output.jpg'):
    """Test a single image: confidence scores, visual overlay, and top-2 when uncertain."""
    print("="*60)
    print("Testing Trained Model")
    print("="*60)
    print(f"\nLoading model from {model_path}...")
    processor, model, device = load_model(model_path)
    class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Classes: {class_names}")
    print(f"\nLoading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    print("Running inference...")
    predicted_label, confidence_pct, probabilities, id2label = predict(processor, model, device, image)
    confidence = confidence_pct / 100.0
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Image: {Path(image_path).name}")
    print(f"\nPredicted: {predicted_label}: {confidence_pct:.1f}%")
    if confidence < UNCERTAINTY_THRESHOLD:
        top2 = get_top_k_probs(probabilities, id2label, k=2)
        print("\nTop 2 (uncertainty detected):")
        for i, (label, prob) in enumerate(top2, 1):
            print(f"  {i}. {label}: {prob * 100:.1f}%")
    print(f"\nAll predictions:")
    all_probs = get_top_k_probs(probabilities, id2label, k=len(id2label))
    for i, (label, prob) in enumerate(all_probs, 1):
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {label}: {prob * 100:.1f}% {bar}")
    print("-"*60)
    draw_overlay(image, predicted_label, confidence_pct, output_path=overlay_path)
    print(f"\nSaved overlay image to {overlay_path}")


def test_directory(image_dir, model_path=DEFAULT_MODEL_PATH):
    """Test all images in a directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_paths = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    print(f"Found {len(image_paths)} images")
    print("="*60)
    processor, model, device = load_model(model_path)
    for i, image_path in enumerate(image_paths, 1):
        try:
            predicted_label, confidence_pct, probs, id2label = predict(processor, model, device, str(image_path))
            confidence = confidence_pct / 100.0
            line = f"{i}. {image_path.name:30s} → {predicted_label}: {confidence_pct:.1f}%"
            if confidence < UNCERTAINTY_THRESHOLD:
                top2 = get_top_k_probs(probs, id2label, k=2)
                line += f"  [Top 2: {top2[0][0]}: {top2[0][1]*100:.1f}%, {top2[1][0]}: {top2[1][1]*100:.1f}%]"
            print(line)
        except Exception as e:
            print(f"{i}. {image_path.name:30s} → ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Vision Transformer model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file')
    parser.add_argument('--directory', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='prediction_output.jpg',
                        help='Output path for overlay image (single image only)')
    
    args = parser.parse_args()
    
    if args.image:
        test_image(args.image, args.model_path, overlay_path=args.output)
    elif args.directory:
        test_directory(args.directory, args.model_path)
    else:
        print("Please provide --image or --directory")
        parser.print_help()

