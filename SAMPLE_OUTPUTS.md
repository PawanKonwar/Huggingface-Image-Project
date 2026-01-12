# Sample Input/Output Examples

This document shows example predictions from the trained model.

## Test Results

### Example 1: Cat Image

**Input Image**: `data/my_cat/cat.jpg`

**Output (Latest Training - 5 epochs)**:
```
Predicted: my_cat
Confidence: 94.61%

All predictions:
  1. my_cat          94.61% ████████████████████████████
  2. my_phone         1.55% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  3. my_house         1.34% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  4. my_dog           1.31% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  5. my_car           1.19% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

**Analysis**: The model correctly identified the cat image with excellent confidence (94.61%) after retraining with 5 epochs. Confidence improved from 79.58% to 94.61%.

---

### Example 2: Dog Image

**Input Image**: `data/my_dog/dog.jpg`

**Output (Latest Training - 5 epochs)**:
```
Predicted: my_dog
Confidence: 31.94%

All predictions:
  1. my_dog          31.94% █████████░░░░░░░░░░░░░░░░░░░░
  2. my_house        30.24% █████████░░░░░░░░░░░░░░░░░░░░
  3. my_cat          21.96% ██████░░░░░░░░░░░░░░░░░░░░░░░░
  4. my_phone        10.52% ███░░░░░░░░░░░░░░░░░░░░░░░░░░░
  5. my_car           5.34% █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

**Analysis**: The model predicted "my_dog" correctly but with lower confidence (31.94%). This class needs more training data for better confidence.

---

## Before vs After Customization

### Before (Base Model - 1000 Classes)

When using the original `google/vit-base-patch16-224` model, predictions would look like:

```
Predicted: Egyptian cat
Confidence: 45.23%

Top 5 predictions:
  1. Egyptian cat     45.23%
  2. tabby cat       12.45%
  3. tiger cat        8.32%
  4. Persian cat      5.12%
  5. Siamese cat      3.89%
```

**Issues**:
- Generic ImageNet categories
- Not personalized to your specific objects
- Can't distinguish "my_cat" vs "my_dog" vs "my_car" etc.

### After (Custom Model - 5 Classes)

With our customized model:

```
Predicted: my_cat
Confidence: 79.58%

All predictions:
  1. my_cat          79.58%
  2. my_dog           5.50%
  3. my_phone         5.40%
  4. my_car           5.31%
  5. my_house         4.22%
```

**Benefits**:
- Personalized to your 5 specific classes
- Clear distinction between your objects
- Higher confidence on correct predictions
- Simpler output (only 5 classes instead of 1000)

---

## Training Statistics

**Training Configuration**:
- Epochs: 5 (latest training)
- Batch Size: 1
- Learning Rate: 2e-5
- Training Samples: 4 images
- Validation Samples: 1 image
- Final Validation Accuracy: 100%
- Training Loss: 0.18 (final), 0.89 (initial)

**Note**: With only 1 image per class, the model has limited data to learn from. For production use, add 50-100+ images per class for better accuracy.

---

## Usage Example

```bash
# Test a single image
python test.py --image my_photo.jpg

# Test all images in a directory
python test.py --directory ./test_photos
```

The model outputs:
- Predicted class name
- Confidence percentage
- Probability distribution across all 5 classes
- Visual confidence bars

