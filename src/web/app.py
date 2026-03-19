"""
Gradio web interface for the trained Vision Transformer classifier.
"""

import gradio as gr
from PIL import Image as PILImage

from src.api.inference import draw_overlay, load_model, predict
from src.utils.paths import DATA_DIR, TRAINED_MODEL_DIR

# Load model once at startup
processor, model, device = load_model(TRAINED_MODEL_DIR)


def _example_paths():
    """Collect one image per class from data/ for Gradio examples."""
    paths = []
    if not DATA_DIR.exists():
        return paths
    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".gif"):
                paths.append(str(f))
                break
    return paths


def run(image):
    """
    Run inference and return label with confidence and overlay image.
    image: PIL Image or numpy array from Gradio.
    """
    if image is None:
        return "No image provided.", None
    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image).convert("RGB")
    predicted_label, confidence_pct, _, _ = predict(processor, model, device, image)
    label_text = f"{predicted_label}: {confidence_pct:.1f}%"
    overlay_image = draw_overlay(image, predicted_label, confidence_pct, output_path=None)
    return label_text, overlay_image


example_paths = _example_paths()
examples = [[p] for p in example_paths] if example_paths else None

demo = gr.Interface(
    fn=run,
    inputs=gr.Image(label="Image", type="pil"),
    outputs=[
        gr.Textbox(label="Label with Confidence Score", lines=1),
        gr.Image(label="Image with prediction overlay"),
    ],
    title="Vision Transformer Image Classifier",
    description="Upload an image to get a predicted class and confidence. The image with overlay is shown on the right.",
    examples=examples,
)

def launch():
    demo.launch()


if __name__ == "__main__":
    launch()
