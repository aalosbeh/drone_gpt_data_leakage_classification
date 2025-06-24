# API Documentation

This document provides comprehensive API documentation for the GPT-Based Drone Data Leakage Classification system.

## Table of Contents

- [Core Classes](#core-classes)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Examples](#examples)
- [Error Handling](#error-handling)

## Core Classes

### GPTClassifier

The main classification class for drone data leakage detection using GPT models.

#### Constructor

```python
GPTClassifier(model_name='gpt2', num_labels=2, device=None)
```

**Parameters:**
- `model_name` (str): Name of the pre-trained GPT model. Default: 'gpt2'
- `num_labels` (int): Number of classification labels. Default: 2
- `device` (str, optional): Device to run the model on ('cpu', 'cuda'). Auto-detected if None.

**Example:**
```python
classifier = GPTClassifier(model_name='gpt2', num_labels=2)
```

#### Methods

##### `fit(X_train, y_train, X_val=None, y_val=None, **kwargs)`

Train the GPT model on the provided dataset.

**Parameters:**
- `X_train` (np.ndarray): Training features (n_samples, n_features)
- `y_train` (np.ndarray): Training labels (n_samples,)
- `X_val` (np.ndarray, optional): Validation features
- `y_val` (np.ndarray, optional): Validation labels
- `**kwargs`: Additional training parameters

**Returns:**
- `dict`: Training history and metrics

**Example:**
```python
history = classifier.fit(X_train, y_train, X_val, y_val, 
                        num_epochs=3, batch_size=8, learning_rate=2e-5)
```

##### `predict(X)`

Make predictions on new data.

**Parameters:**
- `X` (np.ndarray): Input features (n_samples, n_features)

**Returns:**
- `np.ndarray`: Predicted class labels (n_samples,)

**Example:**
```python
predictions = classifier.predict(X_test)
```

##### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (np.ndarray): Input features (n_samples, n_features)

**Returns:**
- `np.ndarray`: Class probabilities (n_samples, n_classes)

**Example:**
```python
probabilities = classifier.predict_proba(X_test)
```

##### `score(X, y)`

Calculate accuracy score on test data.

**Parameters:**
- `X` (np.ndarray): Test features
- `y` (np.ndarray): True labels

**Returns:**
- `float`: Accuracy score

**Example:**
```python
accuracy = classifier.score(X_test, y_test)
```

##### `save_model(path)`

Save the trained model to disk.

**Parameters:**
- `path` (str): Directory path to save the model

**Example:**
```python
classifier.save_model('models/my_model')
```

##### `from_pretrained(path)` (Class Method)

Load a pre-trained model from disk.

**Parameters:**
- `path` (str): Directory path containing the saved model

**Returns:**
- `GPTClassifier`: Loaded classifier instance

**Example:**
```python
classifier = GPTClassifier.from_pretrained('models/fine_tuned')
```

### DroneDataset

PyTorch dataset class for handling drone traffic data.

#### Constructor

```python
DroneDataset(features, labels, tokenizer, max_length=256)
```

**Parameters:**
- `features` (np.ndarray): Feature array (n_samples, n_features)
- `labels` (np.ndarray): Label array (n_samples,)
- `tokenizer`: Hugging Face tokenizer instance
- `max_length` (int): Maximum sequence length. Default: 256

#### Methods

##### `__len__()`

Returns the number of samples in the dataset.

##### `__getitem__(idx)`

Returns a single sample as a dictionary with keys: 'input_ids', 'attention_mask', 'labels'.

### ModelEvaluator

Comprehensive evaluation utilities for model assessment.

#### Constructor

```python
ModelEvaluator(output_dir='outputs')
```

**Parameters:**
- `output_dir` (str): Directory for saving evaluation outputs

#### Methods

##### `evaluate_model(model, X_test, y_test)`

Comprehensive model evaluation with multiple metrics.

**Parameters:**
- `model`: Trained classifier instance
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): Test labels

**Returns:**
- `dict`: Comprehensive evaluation metrics

##### `plot_confusion_matrix(y_true, y_pred, title, save_path)`

Generate and save confusion matrix visualization.

##### `plot_roc_curve(y_true, y_proba, title, save_path)`

Generate and save ROC curve visualization.

## Utility Functions

### Data Processing

#### `load_drone_data(filepath)`

Load drone traffic data from CSV file.

**Parameters:**
- `filepath` (str): Path to the CSV data file

**Returns:**
- `tuple`: (features, labels) as numpy arrays

**Example:**
```python
X, y = load_drone_data('data/dataset_64.csv')
```

#### `preprocess_features(X)`

Standardize features using z-score normalization.

**Parameters:**
- `X` (np.ndarray): Raw feature array

**Returns:**
- `np.ndarray`: Standardized features

#### `create_text_representation(features, feature_names)`

Convert numerical features to text representation for GPT processing.

**Parameters:**
- `features` (np.ndarray): Numerical features (n_features,)
- `feature_names` (list): List of feature names

**Returns:**
- `str`: Text representation of features

#### `split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=42)`

Split dataset into train/validation/test sets.

**Parameters:**
- `X` (np.ndarray): Features
- `y` (np.ndarray): Labels
- `test_size` (float): Proportion for test set
- `val_size` (float): Proportion for validation set
- `random_state` (int): Random seed

**Returns:**
- `tuple`: (X_train, X_val, X_test, y_train, y_val, y_test)

### Evaluation Functions

#### `calculate_metrics(y_true, y_pred, y_proba=None)`

Calculate comprehensive classification metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `y_proba` (np.ndarray, optional): Prediction probabilities

**Returns:**
- `dict`: Dictionary containing accuracy, precision, recall, f1, and optionally roc_auc

#### `plot_learning_curves(train_losses, val_losses, save_path=None)`

Plot training and validation learning curves.

#### `statistical_significance_test(results1, results2, metric='accuracy')`

Perform statistical significance testing between two sets of results.

**Parameters:**
- `results1` (list): First set of metric values
- `results2` (list): Second set of metric values
- `metric` (str): Metric name for reporting

**Returns:**
- `dict`: Statistical test results including p-value and effect size

## Configuration

### Configuration File Structure

The system uses a YAML configuration file (`config.yaml`) for managing parameters:

```yaml
model:
  name: "gpt2"                    # Base model name
  num_labels: 2                   # Number of classification labels
  max_length: 256                 # Maximum sequence length
  dropout: 0.1                    # Dropout probability

training:
  learning_rate: 2e-5             # Learning rate
  batch_size: 8                   # Training batch size
  num_epochs: 3                   # Number of training epochs
  warmup_steps: 100               # Warmup steps for learning rate
  weight_decay: 0.01              # Weight decay for regularization
  gradient_accumulation_steps: 2   # Gradient accumulation

data:
  test_size: 0.2                  # Test set proportion
  validation_size: 0.2            # Validation set proportion
  random_state: 42                # Random seed
  feature_names:                  # List of feature names
    - "Mean_IAT"
    - "Median_IAT"
    # ... additional features

evaluation:
  metrics:                        # Evaluation metrics to compute
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "roc_auc"
  cross_validation_folds: 5       # Number of CV folds

paths:
  data_dir: "data"               # Data directory
  model_dir: "models"            # Model directory
  output_dir: "outputs"          # Output directory
  figures_dir: "figures"         # Figures directory
```

### Loading Configuration

```python
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
```

## Examples

### Complete Training Pipeline

```python
import numpy as np
from utils.data_utils import load_drone_data, split_dataset
from utils.model_utils import GPTClassifier
from utils.eval_utils import ModelEvaluator

# Load and prepare data
X, y = load_drone_data('data/dataset_64.csv')
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

# Initialize and train model
classifier = GPTClassifier(model_name='gpt2', num_labels=2)
history = classifier.fit(X_train, y_train, X_val, y_val, 
                        num_epochs=3, batch_size=8)

# Evaluate model
evaluator = ModelEvaluator(output_dir='outputs')
results = evaluator.evaluate_model(classifier, X_test, y_test)

# Save model
classifier.save_model('models/my_trained_model')

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1-Score: {results['f1']:.4f}")
```

### Inference on New Data

```python
# Load pre-trained model
classifier = GPTClassifier.from_pretrained('models/fine_tuned')

# Prepare new data
new_features = np.array([[0.001, 0.0008, 0.0005, ...]])  # Single sample

# Make predictions
prediction = classifier.predict(new_features)
probability = classifier.predict_proba(new_features)

print(f"Predicted class: {prediction[0]}")
print(f"Class probabilities: {probability[0]}")
```

### Batch Processing

```python
import pandas as pd
from utils.data_utils import preprocess_features

# Load model
classifier = GPTClassifier.from_pretrained('models/fine_tuned')

# Process batch of samples
df = pd.read_csv('new_data.csv')
features = preprocess_features(df.values)

# Batch prediction
predictions = classifier.predict(features)
probabilities = classifier.predict_proba(features)

# Save results
results_df = pd.DataFrame({
    'prediction': predictions,
    'low_risk_prob': probabilities[:, 0],
    'high_risk_prob': probabilities[:, 1]
})
results_df.to_csv('predictions.csv', index=False)
```

### Custom Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Make predictions
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

# Detailed classification report
report = classification_report(y_test, y_pred, 
                             target_names=['Low Risk', 'High Risk'])
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

## Error Handling

### Common Exceptions

#### `ModelNotTrainedError`

Raised when attempting to use an untrained model for prediction.

```python
try:
    predictions = classifier.predict(X_test)
except ModelNotTrainedError:
    print("Model must be trained before making predictions")
```

#### `InvalidDataFormatError`

Raised when input data format is incorrect.

```python
try:
    X, y = load_drone_data('invalid_file.csv')
except InvalidDataFormatError as e:
    print(f"Data format error: {e}")
```

#### `InsufficientDataError`

Raised when dataset is too small for training.

```python
try:
    classifier.fit(X_small, y_small)
except InsufficientDataError:
    print("Dataset too small for training")
```

### Best Practices

1. **Always validate input data shape and type**
2. **Check model training status before inference**
3. **Handle GPU memory limitations gracefully**
4. **Use appropriate batch sizes for available memory**
5. **Implement proper logging for debugging**

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check model device placement**:
   ```python
   print(f"Model device: {next(classifier.model.parameters()).device}")
   ```

3. **Monitor memory usage**:
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"GPU memory: {torch.cuda.memory_allocated()}")
   ```

4. **Validate data preprocessing**:
   ```python
   print(f"Feature statistics: mean={X.mean():.4f}, std={X.std():.4f}")
   ```

For additional support, please refer to the main documentation or open an issue on GitHub.

