"""
Shared inference and overlay logic for the Vision Transformer classifier.
Used by test.py and app.py to avoid code duplication.
"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import ViTImageProcessor, ViTForImageClassification

from src.utils.paths import TRAINED_MODEL_DIR


def load_model(model_path=None):
    """Load processor and model; return (processor, model, device)."""
    if model_path is None:
        model_path = TRAINED_MODEL_DIR
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def predict(processor, model, device, image):
    """
    Run inference on a PIL Image.
    image: PIL.Image (RGB).
    Returns: (predicted_label, confidence_pct, probabilities_tensor, id2label).
    """
    if image is None:
        return None, 0.0, None, None
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_idx = logits.argmax(-1).item()
    id2label = model.config.id2label
    predicted_label = id2label[predicted_idx]
    confidence_pct = probabilities[predicted_idx].item() * 100
    return predicted_label, confidence_pct, probabilities, id2label


def get_top_k_probs(probabilities, id2label, k=2):
    """Return list of (label, prob) sorted by prob descending."""
    if probabilities is None or id2label is None:
        return []
    probs = [(id2label[i], probabilities[i].item()) for i in range(len(probabilities))]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[:k]


def draw_overlay(image, predicted_label, confidence_pct, output_path=None):
    """
    Draw label and confidence at top-left on a copy of the image.
    image: PIL.Image (will be copied; not modified).
    output_path: if set, save the result to this path.
    Returns: PIL.Image with overlay (same as saved if output_path given).
    """
    img = image.copy() if hasattr(image, "copy") else Image.open(image).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    text = f"{predicted_label}: {confidence_pct:.1f}%"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
    padding = 8
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text, font=font)
    x, y = padding, padding
    draw.rectangle(
        [x, y, x + tw + 2 * padding, y + th + 2 * padding],
        fill=(0, 0, 0),
        outline=(255, 255, 255),
    )
    draw.text((x + padding, y + padding), text, fill=(255, 255, 255), font=font)
    if output_path:
        img.save(output_path)
    return img
