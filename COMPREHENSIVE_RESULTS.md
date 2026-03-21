# Comprehensive Model Results

This document matches **on-disk data** under `./data/` and the exported metrics in `./results/` (produced at the end of `python train.py`).

## Project overview

| Item | Value |
|------|--------|
| **Base checkpoint** | `google/vit-base-patch16-224` |
| **Custom classes** | `my_cat`, `my_dog`, `my_car`, `my_house`, `my_phone` |
| **Train / validation split** | 80% / 20%, stratified, `random_state=42` (`sklearn.model_selection.train_test_split`) |
| **Validation accuracy (Trainer)** | **79.59%** (`eval_accuracy` in `results/eval_summary.json`: `0.795918…`) |
| **Reported accuracy (sklearn)** | **80%** (two-decimal rounding on the same 49 validation samples) |

---

## Data audit (house vs. dog label confusion)

Several downloaded or scraped images **mixed visual cues** between **`my_house`** and **`my_dog`**: e.g. dogs in front of buildings, wide outdoor facades that looked pet-centric thumbnails, or house exteriors tagged loosely as “animal” scenes. That **label noise** inflated confusion between the two classes and hurt validation metrics until the folders were manually reviewed.

**What we did**

- Opened suspect files in `data/my_house/` and `data/my_dog/` and **removed or re-filed** mislabeled examples.
- Applied the same discipline to **`my_phone`** and **`my_house`** buckets (removing off-topic or broken downloads), which previously helped lift accuracy from roughly **~51%** to **~80%** on validation before the current dataset size.

After cleanup, counts are closer to balanced across most classes; `my_house` and `my_phone` remain somewhat smaller (see tables below).

---

## Dataset inventory (actual `./data/` counts)

These counts are the **number of image files** per class (extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`). They are mirrored in `results/dataset_split.csv`.

| Class | Images on disk |
|-------|----------------|
| my_car | 60 |
| my_cat | 60 |
| my_dog | 60 |
| my_house | 46 |
| my_phone | 16 |
| **Total** | **242** |

### `my_phone` and the “13 images” note

- There are **16** curated phone images in **`data/my_phone/`** in this repository snapshot.
- With the stratified **80/20** split and **`random_state=42`**, **13** of those 16 are assigned to **training** and **3** to **validation** (see next table). So “13” corresponds to **training samples for `my_phone`**, not the folder count.

---

## Stratified split sizes (matches `train.py`)

Same numbers as `results/dataset_split.csv` (generated automatically when you train).

| Class | Total | Train | Validation |
|-------|------:|------:|-----------:|
| my_car | 60 | 48 | 12 |
| my_cat | 60 | 48 | 12 |
| my_dog | 60 | 48 | 12 |
| my_house | 46 | 36 | 10 |
| my_phone | 16 | 13 | 3 |
| **ALL** | **242** | **193** | **49** |

---

## Validation metrics (per class)

Values below are from a full training run (default **30 epochs**, batch size **8**, capped LR **2e-5**), matching the sklearn `classification_report` printed by `src/models/train.py` and saved to **`results/validation_per_class.csv`**.

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|--------:|
| my_car | 0.78 | 0.58 | 0.67 | 12 |
| my_cat | 0.86 | 1.00 | 0.92 | 12 |
| my_dog | 0.82 | 0.75 | 0.78 | 12 |
| my_house | 0.77 | 1.00 | 0.87 | 10 |
| my_phone | 0.50 | 0.33 | 0.40 | 3 |
| **macro avg** | **0.74** | **0.73** | **0.73** | 49 |
| **weighted avg** | **0.79** | **0.80** | **0.78** | 49 |

**Interpretation**

- **`my_phone`** has only **3** validation images, so precision/recall **swing heavily** with a single mistake. Treat phone metrics as **high-variance** unless you add more phone images or use k-fold / more val data.
- **`my_cat`** / **`my_house`** show strong recall on this split; **`my_car`** recall is lower—worth checking remaining confusion pairs (car vs. house façade, etc.).

---

## Where the numbers live in the repo

| Path | Purpose |
|------|---------|
| `results/dataset_split.csv` | Per-class and total train/val counts |
| `results/validation_per_class.csv` | Per-class precision, recall, F1, support + averages |
| `results/eval_summary.json` | `eval_accuracy`, sample counts, split metadata |
| `src/models/train.py` | Writes the above files after each training run |

To refresh all metrics after changing data or hyperparameters:

```bash
python model_custom.py   # if classes/counts changed
python train.py
```

---

## CLI smoke checks

```bash
python test.py --image path/to/image.jpg
python test.py --directory ./some_folder
python main.py   # Gradio UI
```

---

*Last aligned with exported `results/*` and `data/` layout as documented above.*
