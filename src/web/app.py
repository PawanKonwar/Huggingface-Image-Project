"""
Gradio web interface for the trained Vision Transformer classifier.
"""

from pathlib import Path

import gradio as gr
from PIL import Image as PILImage

from src.api.inference import draw_overlay, load_model, predict
from src.utils.paths import DATA_DIR, TRAINED_MODEL_DIR

_HEADER_HTML = """
<div style="text-align: center; max-width: 900px; margin: 0 auto 1.25rem; padding: 0 0.5rem;">
  <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em;">
    Vision Intelligence Pro
  </h1>
  <p style="margin: 0.6rem 0 0; opacity: 0.85; font-size: 0.98rem; line-height: 1.45;">
    Fine-tuned <strong>ViT</strong> classifier — strong generalization on curated data
    (validation accuracy ~80% after dataset cleanup).
  </p>
</div>
"""


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


def _build_demo(processor, model, device):
    """Build Gradio Blocks bound to the given loaded model."""

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

    theme = gr.themes.Soft(primary_hue="indigo")

    with gr.Blocks(theme=theme, title="Vision Intelligence Pro") as demo:
        gr.HTML(_HEADER_HTML)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Image Input")
                image_input = gr.Image(
                    type="pil",
                    label="Upload or paste an image",
                    height=380,
                    sources=["upload", "clipboard", "webcam"],
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Predictions")
                predictions_output = gr.Textbox(
                    label="Label & confidence",
                    lines=2,
                    placeholder="Run Analyze to see the predicted class and score.",
                    interactive=False,
                )
                gr.Markdown("### Visualization")
                visualization_output = gr.Image(
                    label="Overlay preview",
                    type="pil",
                    height=380,
                )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[image_input],
                label="Quick examples (one per class)",
            )

        analyze_btn.click(
            fn=run,
            inputs=[image_input],
            outputs=[predictions_output, visualization_output],
        )

    return demo


def launch(model_path=None, **launch_kwargs):
    """
    Load weights from ``model_path`` (Hugging Face save directory) and start Gradio.

    Parameters
    ----------
    model_path :
        Directory with config + weights. If None, uses ``models/checkpoint-final`` (see ``TRAINED_MODEL_DIR``).
    launch_kwargs :
        Forwarded to ``gradio.Blocks.launch`` (e.g. ``server_name="0.0.0.0"``, ``server_port=7860``).
    """
    if model_path is None:
        model_path = TRAINED_MODEL_DIR
    mp = Path(model_path)
    print(f"Loading model from {mp.resolve()} ...")
    processor, model, device = load_model(mp)
    demo = _build_demo(processor, model, device)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch()
