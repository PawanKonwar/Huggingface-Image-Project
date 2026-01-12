"""
Training script for fine-tuning the custom Vision Transformer model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
from pathlib import Path
import os

class ImageDataset(Dataset):
    """Dataset class for image classification."""
    
    def __init__(self, data_dir, processor, class_names):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.class_names = class_names
        self.label2id = {name: idx for idx, name in enumerate(class_names)}
        
        self.images = []
        self.labels = []
        
        # Load images from class subdirectories
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for class_name in class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist. Skipping.")
                continue
            
            label_id = self.label2id[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.images.append(str(img_path))
                    self.labels.append(label_id)
        
        print(f"\nDataset: {len(self.images)} images")
        for class_name in class_names:
            count = sum(1 for label in self.labels if label == self.label2id[class_name])
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train(data_dir='./data', model_path='./custom_vit_model', output_dir='./trained_model',
          epochs=5, batch_size=8, learning_rate=2e-5):
    """Train the custom model."""
    
    print("="*60)
    print("Training Custom Vision Transformer")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    
    class_names = [model.config.id2label[i] for i in range(5)]
    print(f"Classes: {class_names}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)
    
    # Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    full_dataset = ImageDataset(data_dir, processor, class_names)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data collator
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
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
    print(f"\nStarting training...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-"*60)
    
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"Validation accuracy: {eval_results.get('eval_accuracy', 0):.2%}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom Vision Transformer')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory with class subdirectories')
    parser.add_argument('--model_path', type=str, default='./custom_vit_model',
                        help='Path to custom model')
    parser.add_argument('--output_dir', type=str, default='./trained_model',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

