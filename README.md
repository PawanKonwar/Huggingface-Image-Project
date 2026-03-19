# Hugging Face Image Classification Project

Custom Vision Transformer (ViT) fine-tuned for **your own classes** using `google/vit-base-patch16-224`. A **dynamic framework** that supports **any number of classes**—no hardcoded labels.

## Overview

This project:

1. Uses the pre-trained **`google/vit-base-patch16-224`** model
2. **Dynamically** infers classes by scanning `./data`: each subfolder name becomes a class (e.g. `my_cat`, `my_dog`, `my_car`, …)
3. Modifies the model from 1000 ImageNet classes to **N** custom classes (N = number of subfolders)
4. Trains on your images with augmentation, stratified train/val split, and frozen backbone
5. Tests with **confidence scores**, **uncertainty detection**, and **prediction overlays**
6. Provides a **Gradio web UI** for interactive inference

You can use 5 classes, 10 classes, or any number—just add one folder per class under `./data`.

## New Features

- **Data augmentation** — Training uses `RandomResizedCrop(224)`, `RandomHorizontalFlip`, and `ColorJitter`. Validation uses deterministic `Resize(224, 224)` only. Transforms are applied in the dataset when loading each image.
- **Stratified 80/20 train/val split** — Uses `sklearn.model_selection.train_test_split` with `stratify=labels` so train and validation keep the same class proportions. Split is done on file paths before building datasets.
- **Confidence scores and uncertainty detection** — Inference applies softmax and reports **label: XX.X%**. When the top confidence is below 90%, the script prints the **top 2** classes and their percentages so you can see ambiguity.
- **Gradio web UI** — Run `python main.py` for a browser interface: upload an image, get a **Label with Confidence Score** and an **Image with prediction overlay**. Example images from `data/` are preloaded for quick testing.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare your dataset

Organize images in one folder per class under `./data`. **Folder names = class names.**

```
data/
  my_cat/
    image1.jpg
    ...
  my_dog/
    ...
  my_car/
  my_house/
  my_phone/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`. Add as many classes as you want.

### Step 2: Create the custom model

This scans `./data` and builds a model with one output per class:

```bash
python model_custom.py
```

Creates `./custom_vit_model` with **N** classes (N = number of subfolders in `./data`). No code change needed when you add or remove classes.

### Step 3: Train

```bash
python train.py --data_dir ./data --epochs 5 --batch_size 8
```

**Options:** `--data_dir`, `--model_path`, `--output_dir`, `--epochs`, `--batch_size`, `--learning_rate`. Training uses an 80% train / 20% validation stratified split and reports validation accuracy and per-class metrics at the end of each epoch.

### Step 4: Test (CLI)

**Single image (prints confidence, top-2 if uncertain, saves overlay):**

```bash
python test.py --image my_photo.jpg
```

**Directory of images:**

```bash
python test.py --directory ./my_test_photos
```

**Custom overlay path:**

```bash
python test.py --image photo.jpg --output result.jpg
```

### Step 5: Test (Web UI)

Launch the Gradio app (loads model from `./trained_model`):

```bash
python main.py
```

Open the URL shown in the terminal (e.g. http://127.0.0.1:7860). Upload an image to get:

- **Label with Confidence Score** (e.g. `my_cat: 98.5%`)
- **Image with prediction overlay** (label + confidence drawn on the image)

Use the **Examples** (one image per class from `data/`) to try the model immediately.

## Usage Summary

| Command | Description |
|--------|-------------|
| `python model_custom.py` | Build custom model from `./data` class folders |
| `python train.py [--data_dir ./data] [--epochs 5] ...` | Train with stratified 80/20 split |
| `python test.py --image <path>` | Single-image test + overlay saved as `prediction_output.jpg` |
| `python test.py --directory <dir>` | Batch test; confidence and top-2 when uncertain |
| `python main.py` | Start Gradio web UI for interactive inference |

## Documentation

- **README.md** — This file (overview, features, usage)
- **USER_GUIDE.md** — Step-by-step guide and troubleshooting
- **COMPREHENSIVE_RESULTS.md** — Test results and analysis

## Project Structure

```
huggingface-image-project/
├── main.py                      # Entry point (launches Gradio UI)
├── app.py                       # Wrapper (backward-compatible: python app.py)
├── model_custom.py              # Wrapper (backward-compatible: python model_custom.py)
├── train.py                     # Wrapper (backward-compatible: python train.py)
├── test.py                      # CLI testing (confidence, overlay, top-2)
├── requirements.txt              # Dependencies
├── src/                          # Modular code
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── inference.py        # Shared inference & overlay logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_custom.py    # Dynamic model creation (N classes from ./data)
│   │   └── train.py           # Training (augmentation, stratified split, frozen backbone)
│   ├── web/
│   │   ├── __init__.py
│   │   └── app.py             # Gradio UI (imports from src.api.inference)
│   └── utils/
│       ├── __init__.py
│       ├── paths.py          # Project root/data/model path helpers
│       └── download_images_loremflickr.py
├── README.md                    # This file
├── USER_GUIDE.md                # Detailed user guide
├── COMPREHENSIVE_RESULTS.md     # Results and analysis
├── .gitignore
├── custom_vit_model/          # Created by model_custom.py (not in git)
├── trained_model/             # Created by train.py (not in git)
└── data/                      # Your images, one subfolder per class (not in git)
    ├── my_cat/
    ├── my_dog/
    ├── my_car/
    ├── my_house/
    └── my_phone/
```

## Complete Workflow

```bash
pip install -r requirements.txt
python model_custom.py
python train.py --data_dir ./data --epochs 5
python test.py --image my_photo.jpg
python main.py   # optional: web UI
```

## Customization (technical)

- **Base model:** `google/vit-base-patch16-224` (ViT, 224×224, 768-d)
- **Change:** Final layer `Linear(768, 1000)` → `Linear(768, N)`; `id2label` / `label2id` from class names
- **Training:** Backbone frozen; only the classification head is trained
- **Data:** Stratified 80% train / 20% validation; training augmentation, validation resize-only

## Tips

- Use at least 50–100 images per class when possible
- Keep similar proportions across classes for best stratified split
- Reduce `--batch_size` (e.g. 4 or 8) if you run out of memory

## Troubleshooting

- **No images found** — Ensure `data/<class_name>/` exists and filenames use supported extensions.
- **Model not found** — Run `python model_custom.py` first; then train so `./trained_model` exists before `test.py` or `main.py`.
- **Out of memory** — Use a smaller `--batch_size` in `train.py`.

## Requirements

- Python 3.8+
- See `requirements.txt` (PyTorch, Transformers, Gradio, scikit-learn, Pillow, etc.)

## License

This project uses the `google/vit-base-patch16-224` model from Hugging Face.
