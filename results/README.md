# Training results (machine-readable)

These files are **regenerated automatically** when you run `python train.py` (see `src/models/train.py`).

| File | Contents |
|------|-----------|
| `dataset_split.csv` | Image counts per class + stratified 80/20 train/val sizes (`random_state=42`) |
| `validation_per_class.csv` | Validation precision / recall / F1 / support per class (+ macro/weighted averages) |
| `eval_summary.json` | `eval_accuracy`, train/val sample counts |

Use them to keep `README.md` and `COMPREHENSIVE_RESULTS.md` aligned with your latest run.
