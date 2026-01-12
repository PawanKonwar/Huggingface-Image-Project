# Customization Details

## Brief Explanation of Customization

This project customizes the `google/vit-base-patch16-224` Vision Transformer model from 1000 general ImageNet classes to 5 specific personalized classes.

## What Changed

### 1. Classification Head Modification

**Original Model**:
```python
classifier = Linear(768, 1000)  # 1000 ImageNet classes
```

**Customized Model**:
```python
classifier = Linear(768, 5)  # 5 custom classes
```

### 2. Label Mappings

**Before**:
- 1000 labels: "Egyptian cat", "golden retriever", "sports car", "palace", "cellular telephone", etc.

**After**:
- 5 labels: `my_cat`, `my_dog`, `my_car`, `my_house`, `my_phone`

### 3. Model Configuration

Updated `model.config`:
- `num_labels`: 1000 → 5
- `id2label`: {0: "my_cat", 1: "my_dog", 2: "my_car", 3: "my_house", 4: "my_phone"}
- `label2id`: {"my_cat": 0, "my_dog": 1, "my_car": 2, "my_house": 3, "my_phone": 4}

## Why This Works

1. **Transfer Learning**: The pre-trained ViT model already learned powerful image features from ImageNet. We keep these features and only retrain the final classification layer.

2. **Feature Reuse**: The 768-dimensional feature representation from the transformer encoder works well for any image classification task, not just ImageNet classes.

3. **Fine-Tuning**: By training only the classifier head on your custom data, the model adapts to recognize your specific objects while maintaining the general image understanding from pre-training.

## Code Implementation

The customization happens in `model_custom.py`:

```python
# Load pre-trained model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Replace classifier head
hidden_size = model.config.hidden_size  # 768
model.classifier = nn.Linear(hidden_size, 5)  # New: 5 classes

# Update configuration
model.config.num_labels = 5
model.config.id2label = {i: name for i, name in enumerate(class_names)}
model.config.label2id = {name: i for i, name in enumerate(class_names)}
```

## Training Process

1. **Data Preparation**: Images organized in class subdirectories
2. **Fine-Tuning**: Train the new classifier head on your custom images
3. **Validation**: Test on held-out images to measure accuracy
4. **Deployment**: Use the trained model for inference on new images

## Results

- **Base Model**: Recognizes 1000 generic ImageNet categories
- **Custom Model**: Recognizes your 5 specific personalized classes
- **Accuracy**: Improves with more training data (aim for 50-100+ images per class)

