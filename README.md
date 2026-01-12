# Hugging Face Image Classification Project

Custom Vision Transformer (ViT) model fine-tuned for 5 specific classes using `google/vit-base-patch16-224`.

## Overview

This project:
1. Uses the pre-trained `google/vit-base-patch16-224` model
2. Modifies it from 1000 classes to 5 custom classes: **my_cat**, **my_dog**, **my_car**, **my_house**, **my_phone**
3. Trains it on your custom image dataset
4. Tests it on your own photos

## Customization Explanation

### What Was Changed

The base `google/vit-base-patch16-224` model is pre-trained on ImageNet with 1000 general classes (like "cat", "dog", "car", etc.). This project customizes it for 5 specific classes:

**Before (Base Model):**
- 1000 output classes (ImageNet categories)
- Generic labels like "Egyptian cat", "golden retriever", "sports car"
- Classifier head: `Linear(768, 1000)`

**After (Custom Model):**
- 5 output classes: `my_cat`, `my_dog`, `my_car`, `my_house`, `my_phone`
- Personalized labels for your specific objects
- Classifier head: `Linear(768, 5)`

### How It Works

1. **Model Architecture Modification** (`model_custom.py`):
   - Loads the pre-trained ViT model (weights preserved)
   - Replaces the final classification layer from 1000 → 5 outputs
   - Updates label mappings (`id2label`, `label2id`)
   - Keeps all pre-trained feature extraction layers (transfer learning)

2. **Fine-Tuning** (`train.py`):
   - Freezes most layers, trains only the new classifier head
   - Uses your custom images to learn class-specific features
   - Adapts the model to recognize your specific objects

3. **Key Benefits**:
   - Leverages pre-trained features (no training from scratch)
   - Fast training (only classifier head needs learning)
   - Personalized for your specific objects
   - Requires less data than training from scratch

### Technical Details

- **Base Model**: Vision Transformer (ViT) with patch size 16×16, 224×224 input
- **Feature Dimension**: 768 (hidden size)
- **Modification**: Final linear layer changed from `768 → 1000` to `768 → 5`
- **Training**: Fine-tuning with custom dataset using Hugging Face Trainer

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Create Custom Model

Run the script to create a custom model with your 5 classes:
```bash
python model_custom.py
```

This creates a custom model in `./custom_vit_model` with 5 classes instead of 1000.

**Your classes:**
- my_cat
- my_dog
- my_car
- my_house
- my_phone

### Step 2: Prepare Your Dataset

Organize your images in this structure:
```
data/
  my_cat/
    image1.jpg
    image2.jpg
    ...
  my_dog/
    image1.jpg
    ...
  my_car/
    ...
  my_house/
    ...
  my_phone/
    ...
```

**Important:**
- Folder names must match the class names exactly: `my_cat`, `my_dog`, `my_car`, `my_house`, `my_phone`
- Use common formats: .jpg, .jpeg, .png, .bmp, .gif
- Aim for at least 50-100 images per class for best results
- The `data/` folder structure has been created for you - just add your images!

### Step 3: Train the Model

```bash
python train.py --data_dir ./data --epochs 5 --batch_size 8
```

**Parameters:**
- `--data_dir`: Directory with class subdirectories (default: `./data`)
- `--model_path`: Path to custom model (default: `./custom_vit_model`)
- `--output_dir`: Where to save trained model (default: `./trained_model`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8, reduce if out of memory)
- `--learning_rate`: Learning rate (default: 2e-5)

**Example with custom parameters:**
```bash
python train.py --data_dir ./data --epochs 10 --batch_size 16 --learning_rate 2e-5
```

### Step 4: Test Your Photos

**Single image:**
```bash
python test.py --image my_photo.jpg
```

**All images in a directory:**
```bash
python test.py --directory ./my_test_photos
```

**Use a different model:**
```bash
python test.py --image photo.jpg --model_path ./my_trained_model
```

## Documentation

- **README.md** - This file (project overview and quick start)
- **USER_GUIDE.md** - Complete step-by-step user guide with detailed instructions
- **COMPREHENSIVE_RESULTS.md** - Detailed test results, comparisons, and analysis

## Project Structure

```
huggingface-image-project/
├── requirements.txt              # Dependencies
├── model_custom.py              # Create custom 5-class model
├── train.py                     # Training script
├── test.py                      # Testing script
├── README.md                    # Main documentation (this file)
├── USER_GUIDE.md                # Detailed user guide
├── COMPREHENSIVE_RESULTS.md     # Detailed results and analysis
├── .gitignore                   # Git ignore file
├── custom_vit_model/            # Created by model_custom.py (not in git)
├── trained_model/               # Created by train.py (not in git)
└── data/                        # Your training images (not in git)
    ├── my_cat/
    ├── my_dog/
    ├── my_car/
    ├── my_house/
    └── my_phone/
```

## Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create custom model
python model_custom.py

# 3. Add your images to data/ subdirectories
#    - data/my_cat/your_cat_images.jpg
#    - data/my_dog/your_dog_images.jpg
#    - data/my_car/your_car_images.jpg
#    - data/my_house/your_house_images.jpg
#    - data/my_phone/your_phone_images.jpg

# 4. Train the model
python train.py --data_dir ./data --epochs 5

# 5. Test your photos
python test.py --image my_photo.jpg
```

## Tips for Best Results

- **More data = better accuracy**: Use at least 50-100 images per class
- **Image quality**: Use clear, well-lit images
- **Variety**: Include different angles, backgrounds, lighting conditions
- **Batch size**: Reduce if you run out of memory (try 4 or 8)
- **Epochs**: Start with 5, increase if validation accuracy is still improving
- **Balance**: Try to have roughly equal number of images per class

## Troubleshooting

**No images found:**
- Check that your data directory structure matches the expected format
- Verify folder names match exactly: `my_cat`, `my_dog`, `my_car`, `my_house`, `my_phone`
- Ensure images have supported extensions (.jpg, .png, etc.)

**Out of memory:**
- Reduce `--batch_size` (try 4 or 8)
- Use smaller images or resize before training

**Low accuracy:**
- Add more training images per class
- Ensure images are clear and representative
- Try training for more epochs
- Check that test images are similar to training data

**Model not found:**
- Make sure you've run `python model_custom.py` before training
- Check that `./custom_vit_model` exists

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list

## Results

See `COMPREHENSIVE_RESULTS.md` for:
- Detailed test results on all images
- Before/After customization comparison
- Performance analysis and statistics
- Class-wise breakdown and confidence scores

**Quick Summary**:
- Accuracy: 100% (5/5 correct predictions)
- Average Confidence: 78.32%
- High Confidence Classes: 4/5 (80%)

## GitHub Repository

This project is ready to push to GitHub:

```bash
# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/huggingface-image-project.git
git branch -M main
git push -u origin main
```

**Note**: Model files (`custom_vit_model/`, `trained_model/`) and images (`data/`) are excluded via `.gitignore` (too large/private).

## License

This project uses the `google/vit-base-patch16-224` model from Hugging Face.

