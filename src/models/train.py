"""
Training script for fine-tuning the custom Vision Transformer model.
"""

import csv
import json
import torch
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

from src.utils.paths import CUSTOM_MODEL_DIR, DATA_DIR, RESULTS_DIR, TRAINED_MODEL_DIR


def _collect_paths_and_labels(data_dir, class_names):
    """Scan data_dir for class subfolders; return (list of image paths, list of labels)."""
    data_path = Path(data_dir)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    paths, labels = [], []
    for class_name in class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
        label_id = label2id[class_name]
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in extensions:
                paths.append(str(p))
                labels.append(label_id)
    return paths, labels


class ImageDataset(Dataset):
    """
    Dataset for image classification with a preprocessing pipeline that applies
    transforms dynamically when loading each image (in __getitem__).
    - Training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, then processor (resize + normalize).
    - Validation/test: Resize only, then processor (normalize); no random augmentation.
    """

    def __init__(self, processor, class_names, mode, data_dir=None, image_paths=None, labels=None):
        """
        Args:
            processor: Hugging Face ViTImageProcessor (resize + normalize after torchvision transforms).
            class_names: List of class names (must match folder names).
            mode: 'train' for augmentation; 'val' for resize + normalize only (no random flipping).
            data_dir: Root directory with class subdirectories (used only if image_paths is None).
            image_paths: Optional list of image paths (after stratified split).
            labels: Optional list of labels, same length as image_paths.
        """
        self.processor = processor
        self.class_names = class_names
        self.label2id = {name: idx for idx, name in enumerate(class_names)}
        self.mode = mode

        if image_paths is not None and labels is not None:
            self.images = list(image_paths)
            self.labels = list(labels)
        else:
            if data_dir is None:
                raise ValueError("Provide either (image_paths, labels) or data_dir.")
            self.images = []
            self.labels = []
            data_path = Path(data_dir)
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            for class_name in class_names:
                class_dir = data_path / class_name
                if not class_dir.exists():
                    print(f"Warning: {class_dir} does not exist. Skipping.")
                    continue
                label_id = self.label2id[class_name]
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in extensions:
                        self.images.append(str(img_path))
                        self.labels.append(label_id)

        # Preprocessing pipeline: augmentation for training; resize-only for validation/test
        if mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ])
        else:
            # Validation/test: only resize and normalize (no random flipping or cropping)
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

        if len(self.images) > 0:
            print(f"\nDataset ({mode}): {len(self.images)} images")
            for class_name in class_names:
                count = sum(1 for label in self.labels if label == self.label2id[class_name])
                print(f"  {class_name}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Apply transforms dynamically at load time (per-image; different randomness for train)
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.image_transform(image)
        # Processor: resize (if needed) + normalize + to tensor
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train(data_dir=None, model_path=None, output_dir=None,
          epochs=30, batch_size=8, learning_rate=2e-5):
    """Train the custom model."""

    print("="*60)
    print("Training Custom Vision Transformer")
    print("="*60)

    if data_dir is None:
        data_dir = DATA_DIR
    if model_path is None:
        model_path = CUSTOM_MODEL_DIR
    if output_dir is None:
        output_dir = TRAINED_MODEL_DIR

    output_dir_str = str(output_dir)

    # When unfreezing transformer layers, use a smaller LR to avoid destroying
    # already-learned representations for the backbone.
    learning_rate = min(float(learning_rate), 2e-5)

    # Load model
    print(f"\nLoading model from {model_path}...")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)

    class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Classes: {class_names}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)

    # Freeze the Vision Transformer backbone, then unfreeze the last 2 encoder layers
    # so the model can adapt to your domain (e.g., houses/phones) without overfitting.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze the last 2 ViT encoder layers (keeps most of the backbone frozen).
    # ViTForImageClassification uses: model.vit.encoder.layer (ModuleList of blocks).
    vit_encoder_layers = model.vit.encoder.layer
    for param in vit_encoder_layers[-2:].parameters():
        param.requires_grad = True

    # Collect all image paths and labels, then stratified 80% train / 20% validation
    print(f"\nLoading dataset from {data_dir}...")
    all_paths, all_labels = _collect_paths_and_labels(data_dir, class_names)
    if not all_paths:
        raise ValueError(f"No images found in {data_dir}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=42,
    )

    # Persist dataset split counts for README / results docs (matches stratified 80/20, random_state=42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_csv = RESULTS_DIR / "dataset_split.csv"
    train_ct = Counter(train_labels)
    val_ct = Counter(val_labels)
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "total_images", "train_images", "val_images"])
        for idx, name in enumerate(class_names):
            tr = train_ct.get(idx, 0)
            va = val_ct.get(idx, 0)
            w.writerow([name, tr + va, tr, va])
        w.writerow(["ALL", len(all_paths), len(train_labels), len(val_labels)])

    train_dataset = ImageDataset(
        processor, class_names, mode='train',
        image_paths=train_paths, labels=train_labels,
    )
    val_dataset = ImageDataset(
        processor, class_names, mode='val',
        image_paths=val_paths, labels=val_labels,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Collate handles variable-sized batches (e.g. last batch smaller than batch_size) and empty batch
    def collate_fn(batch):
        if not batch:
            return {
                'pixel_values': torch.empty(0, 3, 224, 224),
                'labels': torch.empty(0, dtype=torch.long),
            }
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}

    # Training arguments (dataloader_drop_last=False so last small batch does not crash training)
    training_args = TrainingArguments(
        output_dir=output_dir_str,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f'{output_dir_str}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = (predictions == labels).astype(float).mean()
        return {'accuracy': accuracy}

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-"*60)

    trainer.train()

    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"Validation accuracy: {eval_results.get('eval_accuracy', 0):.2%}")

    # Per-class precision, recall, F1-score
    pred_output = trainer.predict(val_dataset)
    pred_labels = pred_output.predictions.argmax(axis=-1)
    true_labels = pred_output.label_ids
    class_names_eval = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print("\nPer-class metrics (validation set):")
    print(classification_report(true_labels, pred_labels, target_names=class_names_eval))

    # Export validation metrics to archive/results/ (see RESULTS_DIR)
    labels_idx = list(range(len(class_names_eval)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=labels_idx,
        zero_division=0,
    )
    val_acc = float(eval_results.get("eval_accuracy", accuracy_score(true_labels, pred_labels)))
    metrics_csv = RESULTS_DIR / "validation_per_class.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for i, name in enumerate(class_names_eval):
            w.writerow([name, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
        p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )
        p_wa, r_wa, f_wa, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        w.writerow(["macro_avg", f"{p_ma:.4f}", f"{r_ma:.4f}", f"{f_ma:.4f}", len(true_labels)])
        w.writerow(["weighted_avg", f"{p_wa:.4f}", f"{r_wa:.4f}", f"{f_wa:.4f}", len(true_labels)])
    summary_path = RESULTS_DIR / "eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "eval_accuracy": val_acc,
                "val_samples": len(val_labels),
                "train_samples": len(train_labels),
                "random_state": 42,
                "stratify": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote metrics to {metrics_csv} and {summary_path}")

    # Save model
    print(f"\nSaving model to {output_dir_str}...")
    trainer.save_model()
    processor.save_pretrained(output_dir_str)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train custom Vision Transformer')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with class subdirectories')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to custom model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (will be capped at 2e-5)')

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
