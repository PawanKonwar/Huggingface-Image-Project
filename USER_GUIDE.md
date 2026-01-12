# User Guide

Complete step-by-step guide for using the Hugging Face Image Classification Project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start](#quick-start)
3. [Detailed Step-by-Step Guide](#detailed-step-by-step-guide)
4. [Using the Scripts](#using-the-scripts)
5. [Common Workflows](#common-workflows)
6. [Tips and Best Practices](#tips-and-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- **Python 3.8+** (Python 3.10 or newer recommended)
- **pip** (Python package installer)
- **Git** (optional, for version control)

### System Requirements

- **RAM**: At least 8GB (16GB recommended)
- **Storage**: ~2GB for model files and dependencies
- **GPU**: Optional but recommended for faster training (CPU works too)

---

## Quick Start

For experienced users, here's the fast track:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create custom model
python model_custom.py

# 3. Add images to data/ subdirectories
#    - data/my_cat/your_images.jpg
#    - data/my_dog/your_images.jpg
#    - etc.

# 4. Train the model
python train.py --data_dir ./data --epochs 5

# 5. Test your images
python test.py --image your_image.jpg
```

**That's it!** For detailed explanations, continue reading below.

---

## Detailed Step-by-Step Guide

### Step 1: Installation

#### 1.1 Clone or Download the Project

If you have the project in a Git repository:
```bash
git clone <repository-url>
cd huggingface-image-project
```

Or simply download and extract the project folder.

#### 1.2 Install Python Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Expected output:**
- PyTorch will be installed (may take a few minutes)
- Transformers library
- Pillow (image processing)
- Accelerate (training acceleration)

**Troubleshooting:**
- If you get permission errors, use: `pip install --user -r requirements.txt`
- On macOS/Linux, you might need: `pip3 install -r requirements.txt`
- If installation fails, ensure you have Python 3.8+ installed

#### 1.3 Verify Installation

Verify everything is installed correctly:

```bash
python --version  # Should show Python 3.8+
python -c "import torch; import transformers; print('✓ All packages installed')"
```

---

### Step 2: Prepare Your Custom Model

#### 2.1 Understand What You're Creating

The `model_custom.py` script:
- Downloads the base `google/vit-base-patch16-224` model (first time only)
- Modifies it from 1000 ImageNet classes to your 5 custom classes
- Saves the custom model to `./custom_vit_model/`

**Your 5 classes are:**
- `my_cat`
- `my_dog`
- `my_car`
- `my_house`
- `my_phone`

#### 2.2 Run the Model Customization Script

```bash
python model_custom.py
```

**First run will:**
- Download the base model (~330MB) - this may take a few minutes
- Modify the classification head
- Save to `./custom_vit_model/`

**Expected output:**
```
Loading base model: google/vit-base-patch16-224
Modifying classification head from 1000 to 5 classes...
Saving custom model to ./custom_vit_model...
Custom model created successfully!
Classes: ['my_cat', 'my_dog', 'my_car', 'my_house', 'my_phone']
```

**Note:** This only needs to be run once. The custom model is saved and can be reused.

---

### Step 3: Prepare Your Training Data

#### 3.1 Understand the Required Structure

Your images must be organized in folders matching your class names exactly:

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

#### 3.2 Organize Your Images

1. **Create class folders** (if not already created):
   ```bash
   mkdir -p data/my_cat data/my_dog data/my_car data/my_house data/my_phone
   ```

2. **Copy your images** to the appropriate folders:
   - Cat images → `data/my_cat/`
   - Dog images → `data/my_dog/`
   - Car images → `data/my_car/`
   - House images → `data/my_house/`
   - Phone images → `data/my_phone/`

3. **Image Requirements:**
   - **Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`
   - **Recommended**: At least 50-100 images per class
   - **Quality**: Clear, well-lit images
   - **Variety**: Different angles, backgrounds, lighting

#### 3.3 Verify Your Data Structure

Check that your images are properly organized:

```bash
# On Linux/macOS
for dir in data/*/; do echo "$(basename "$dir"): $(find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l) images"; done
```

Or manually check each folder contains images.

---

### Step 4: Train the Model

#### 4.1 Basic Training Command

```bash
python train.py --data_dir ./data --epochs 5 --batch_size 8
```

**What this does:**
- Loads your custom model
- Loads images from `data/` directory
- Splits data: 80% training, 20% validation
- Trains for 5 epochs
- Saves the trained model to `./trained_model/`

#### 4.2 Training Parameters Explained

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `--data_dir` | `./data` | Directory with your images | If images are elsewhere |
| `--model_path` | `./custom_vit_model` | Path to custom model | If using different model |
| `--output_dir` | `./trained_model` | Where to save trained model | To save to different location |
| `--epochs` | `5` | Number of training iterations | Increase for more training (10-20) |
| `--batch_size` | `8` | Images per batch | Reduce if out of memory (4-8) |
| `--learning_rate` | `2e-5` | Learning speed | Usually fine as-is |

#### 4.3 Example Training Commands

**Quick training (small dataset):**
```bash
python train.py --data_dir ./data --epochs 3 --batch_size 4
```

**Thorough training (more data):**
```bash
python train.py --data_dir ./data --epochs 10 --batch_size 16 --learning_rate 2e-5
```

**Training with custom paths:**
```bash
python train.py --data_dir ./my_images --output_dir ./my_trained_model --epochs 5
```

#### 4.4 What to Expect During Training

**Training output:**
```
============================================================
Training Custom Vision Transformer
============================================================

Loading model from ./custom_vit_model...
Classes: ['my_cat', 'my_dog', 'my_car', 'my_house', 'my_phone']
Device: cpu

Loading dataset from ./data...
Dataset: 150 images
  my_cat: 30 images
  my_dog: 30 images
  ...
Train: 120, Val: 30

Starting training...
Epochs: 5, Batch size: 8, LR: 2e-05
------------------------------------------------------------
[Training progress bars...]
Evaluating...
Validation accuracy: 85.23%

Saving model to ./trained_model...
============================================================
Training completed!
============================================================
```

**Training time:**
- Small dataset (50 images): 1-2 minutes
- Medium dataset (200 images): 5-10 minutes
- Large dataset (1000+ images): 30+ minutes
- With GPU: Much faster (3-5x speedup)

---

### Step 5: Test Your Model

#### 5.1 Test a Single Image

```bash
python test.py --image path/to/your/image.jpg
```

**Example:**
```bash
python test.py --image my_photo.jpg
python test.py --image data/my_cat/cat_test.jpg
python test.py --image /Users/yourname/Pictures/test_photo.jpg
```

**Output:**
```
============================================================
Testing Trained Model
============================================================

Loading model from ./trained_model...
Classes: ['my_cat', 'my_dog', 'my_car', 'my_house', 'my_phone']

Loading image: my_photo.jpg
Running inference...

------------------------------------------------------------
PREDICTION RESULTS
------------------------------------------------------------
Image: my_photo.jpg

Predicted: my_cat
Confidence: 94.61%

All predictions:
  1. my_cat          94.61% ████████████████████████████
  2. my_phone         1.55% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  3. my_house         1.34% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  4. my_dog           1.31% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  5. my_car           1.19% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
------------------------------------------------------------
```

#### 5.2 Test Multiple Images

**Test all images in a directory:**
```bash
python test.py --directory ./my_test_photos
```

**Output:**
```
Found 10 images
============================================================
1. photo1.jpg                    → my_cat          (94.61%)
2. photo2.jpg                    → my_dog          (87.23%)
3. photo3.jpg                    → my_car          (92.15%)
...
```

#### 5.3 Use a Different Trained Model

If you've trained multiple models:

```bash
python test.py --image photo.jpg --model_path ./my_custom_trained_model
```

---

## Using the Scripts

### model_custom.py

**Purpose:** Create a custom model with 5 classes instead of 1000.

**Usage:**
```bash
python model_custom.py
```

**What it does:**
- Loads base model (downloads if first time)
- Modifies classification head
- Saves to `./custom_vit_model/`

**When to run:** Once before training

**Customization:** Edit the `my_classes` list in the script to change class names.

---

### train.py

**Purpose:** Train the custom model on your images.

**Usage:**
```bash
python train.py [options]
```

**Options:**
```
--data_dir DIR          Directory with class subdirectories (default: ./data)
--model_path PATH       Path to custom model (default: ./custom_vit_model)
--output_dir DIR        Output directory (default: ./trained_model)
--epochs N              Number of epochs (default: 5)
--batch_size N          Batch size (default: 8)
--learning_rate FLOAT   Learning rate (default: 2e-5)
```

**Examples:**
```bash
# Basic
python train.py

# With custom data directory
python train.py --data_dir ./my_images

# More training
python train.py --epochs 10 --batch_size 16

# All options
python train.py --data_dir ./data --epochs 5 --batch_size 8 --learning_rate 2e-5
```

---

### test.py

**Purpose:** Test the trained model on new images.

**Usage:**
```bash
python test.py --image IMAGE_PATH
python test.py --directory DIR_PATH
```

**Options:**
```
--image PATH            Path to single image file
--directory PATH        Directory containing images
--model_path PATH       Path to trained model (default: ./trained_model)
```

**Examples:**
```bash
# Single image
python test.py --image photo.jpg

# Directory of images
python test.py --directory ./test_photos

# Different model
python test.py --image photo.jpg --model_path ./my_model
```

---

## Common Workflows

### Workflow 1: First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create custom model (downloads base model)
python model_custom.py

# 3. Add your images to data/ folders

# 4. Train
python train.py --data_dir ./data --epochs 5

# 5. Test
python test.py --image test_image.jpg
```

---

### Workflow 2: Retrain with More Data

```bash
# 1. Add more images to data/ folders

# 2. Retrain (overwrites previous model)
python train.py --data_dir ./data --epochs 5

# 3. Test again
python test.py --image test_image.jpg
```

---

### Workflow 3: Train Multiple Models

```bash
# Train model version 1
python train.py --output_dir ./model_v1 --epochs 5

# Train model version 2 (different epochs)
python train.py --output_dir ./model_v2 --epochs 10

# Test both
python test.py --image photo.jpg --model_path ./model_v1
python test.py --image photo.jpg --model_path ./model_v2
```

---

### Workflow 4: Quick Testing Loop

```bash
# Test, train, test again
python test.py --image test.jpg
python train.py --epochs 3
python test.py --image test.jpg
```

---

## Tips and Best Practices

### Data Collection

✅ **Do:**
- Collect 50-100+ images per class
- Include variety (angles, lighting, backgrounds)
- Use clear, high-quality images
- Keep similar images per class (consistent objects)

❌ **Don't:**
- Use too few images (<10 per class)
- Use only similar images (no variety)
- Include blurry or dark images
- Mix different objects in same class

### Training

✅ **Do:**
- Start with 5 epochs, increase if needed
- Monitor validation accuracy
- Use batch_size 8-16 (reduce if out of memory)
- Balance number of images per class

❌ **Don't:**
- Train too many epochs (may overfit)
- Use very large batch_size (out of memory)
- Train with unbalanced classes
- Skip validation

### Testing

✅ **Do:**
- Test on images similar to training data
- Test multiple images
- Check confidence scores
- Verify predictions make sense

❌ **Don't:**
- Test on completely different images
- Trust single test result
- Ignore low confidence scores
- Expect perfect results with little data

---

## Troubleshooting

### Installation Issues

**Problem:** `pip: command not found`
- **Solution:** Use `pip3` instead, or install pip

**Problem:** Permission denied
- **Solution:** Use `pip install --user -r requirements.txt`

**Problem:** Package installation fails
- **Solution:** Update pip: `pip install --upgrade pip`

---

### Model Creation Issues

**Problem:** `model_custom.py` fails
- **Check:** Internet connection (needs to download model first time)
- **Check:** Disk space (needs ~500MB)
- **Solution:** Run again, model download may take time

**Problem:** Model files not created
- **Check:** Script completed without errors
- **Check:** `./custom_vit_model/` directory exists
- **Solution:** Run script again

---

### Training Issues

**Problem:** `No images found`
- **Check:** Images are in correct folders (`data/my_cat/`, etc.)
- **Check:** Folder names match class names exactly
- **Check:** Images have correct extensions (.jpg, .png, etc.)
- **Solution:** Verify directory structure matches requirements

**Problem:** Out of memory
- **Solution:** Reduce batch_size: `--batch_size 4` or `--batch_size 2`
- **Solution:** Use smaller images or fewer images
- **Solution:** Close other applications

**Problem:** Training is very slow
- **Solution:** Normal on CPU, expect 5-30 minutes
- **Solution:** Use GPU if available (automatic if CUDA available)
- **Solution:** Reduce number of images or epochs

**Problem:** Low validation accuracy
- **Solution:** Add more training images
- **Solution:** Train for more epochs
- **Solution:** Check image quality
- **Solution:** Ensure balanced dataset

---

### Testing Issues

**Problem:** `Model not found`
- **Check:** You've run `train.py` successfully
- **Check:** `./trained_model/` directory exists
- **Solution:** Train model first: `python train.py`

**Problem:** `Image not found`
- **Check:** Image path is correct
- **Check:** Image file exists
- **Solution:** Use full path or relative path from project directory

**Problem:** Low confidence predictions
- **Solution:** Add more training data
- **Solution:** Test on images similar to training data
- **Solution:** Retrain with more epochs

**Problem:** Wrong predictions
- **Solution:** Add more training images
- **Solution:** Check image quality
- **Solution:** Ensure test images are similar to training images
- **Solution:** Train for more epochs

---

### General Issues

**Problem:** Python version error
- **Check:** Python version: `python --version`
- **Solution:** Use Python 3.8+ (install if needed)

**Problem:** Import errors
- **Solution:** Reinstall dependencies: `pip install -r requirements.txt`
- **Solution:** Check virtual environment is activated (if using)

**Problem:** Script not found
- **Check:** You're in the project directory
- **Solution:** Use full path or `cd` to project directory

---

## Getting Help

### Check Documentation
- `README.md` - Main project documentation
- `COMPREHENSIVE_RESULTS.md` - Detailed results and analysis

### Common Solutions
1. **Read error messages carefully** - They often tell you what's wrong
2. **Check file paths** - Ensure paths are correct
3. **Verify installation** - Make sure all packages are installed
4. **Check data structure** - Ensure images are organized correctly

### Next Steps
- Add more training data for better accuracy
- Experiment with different hyperparameters
- Try different training epochs
- Test on various images

---

**Happy training! 🚀**

