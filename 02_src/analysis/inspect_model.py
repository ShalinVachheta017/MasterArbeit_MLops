"""
Model Inspection Script

This script installs TensorFlow (if needed) and inspects the existing model
to understand its architecture, input/output shapes, and parameters.

Run this to understand what the model expects for training.
"""

import sys
from pathlib import Path

# Setup paths - navigate to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "03_models" / "pretrained" / "fine_tuned_model_1dcnnbilstm.keras"

print("="*80)
print("MODEL INSPECTION SCRIPT")
print("="*80)

# Check if TensorFlow is installed
try:
    import tensorflow as tf
    print(f"\n✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("\n❌ TensorFlow not installed!")
    print("\nInstalling TensorFlow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.14"])
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} installed successfully!")

# Check if model file exists
if not MODEL_PATH.exists():
    print(f"\n❌ Model file not found: {MODEL_PATH}")
    sys.exit(1)

print(f"\n✅ Model file found: {MODEL_PATH}")
print(f"   Size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# Load the model
print("\n" + "="*80)
print("LOADING MODEL...")
print("="*80)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("\n✅ Model loaded successfully!")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    sys.exit(1)

# Display model architecture
print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)

model.summary()

# Display input/output shapes
print("\n" + "="*80)
print("INPUT/OUTPUT DETAILS")
print("="*80)

print(f"\n📥 INPUT SHAPE:")
print(f"   - Shape: {model.input_shape}")
if len(model.input_shape) == 3:
    print(f"   - Batch size: Variable (None)")
    print(f"   - Sequence length (window size): {model.input_shape[1]}")
    print(f"   - Number of features: {model.input_shape[2]}")
    print(f"\n   💡 This means the model expects:")
    print(f"      - {model.input_shape[1]} timesteps per sample")
    print(f"      - {model.input_shape[2]} sensor readings per timestep")
else:
    print(f"   - Unexpected input shape format!")

print(f"\n📤 OUTPUT SHAPE:")
print(f"   - Shape: {model.output_shape}")
if len(model.output_shape) == 2:
    num_classes = model.output_shape[1]
    print(f"   - Number of classes: {num_classes}")
    
    if num_classes == 2:
        print(f"\n   💡 This is BINARY classification:")
        print(f"      - Class 0: (e.g., Calm/Normal)")
        print(f"      - Class 1: (e.g., Anxious/Stressed)")
    elif num_classes > 2:
        print(f"\n   💡 This is MULTI-CLASS classification:")
        for i in range(num_classes):
            print(f"      - Class {i}: (e.g., Anxiety Level {i})")
else:
    print(f"   - Unexpected output shape format!")

# Model configuration
print(f"\n" + "="*80)
print("MODEL CONFIGURATION")
print("="*80)

print(f"\n📊 Layer Summary:")
print(f"   - Total layers: {len(model.layers)}")
print(f"   - Trainable parameters: {model.count_params():,}")

# Extract layer types
layer_types = {}
for layer in model.layers:
    layer_type = type(layer).__name__
    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

print(f"\n📋 Layer Types:")
for layer_type, count in sorted(layer_types.items()):
    print(f"   - {layer_type}: {count}")

# Check if model has optimizer (was it compiled?)
if model.optimizer is not None:
    print(f"\n⚙️  Optimizer:")
    print(f"   - Type: {type(model.optimizer).__name__}")
    print(f"   - Learning rate: {model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate}")

# Check loss function
if hasattr(model, 'loss') and model.loss is not None:
    print(f"\n📉 Loss Function:")
    print(f"   - {model.loss}")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR TRAINING")
print("="*80)

window_size = model.input_shape[1] if len(model.input_shape) == 3 else "Unknown"
num_features = model.input_shape[2] if len(model.input_shape) == 3 else "Unknown"
num_classes = model.output_shape[1] if len(model.output_shape) == 2 else "Unknown"

print(f"""
✅ Required Data Preparation:

1. **Window Size:** {window_size} timesteps
   - At 50Hz, this is {window_size/50 if isinstance(window_size, int) else '?'} seconds of data per sample
   
2. **Features:** {num_features} sensors
   - Likely: Ax, Ay, Az (accelerometer)
            Gx, Gy, Gz (gyroscope)
   
3. **Labels:** {num_classes} classes
   - You need labeled data with {num_classes} distinct categories
   
4. **Data Shape for Training:**
   - X_train shape: (num_samples, {window_size}, {num_features})
   - y_train shape: (num_samples, {num_classes}) [one-hot encoded]

📝 Next Steps:

1. Create sliding windows from your time series data
   - Use window_size = {window_size}
   - Apply overlap (e.g., 50%) for more samples
   
2. Normalize your sensor data
   - Standardization: (x - mean) / std
   - Apply per sensor column
   
3. Split data: 70% train, 15% validation, 15% test

4. Create training script with same architecture
   - Use the model.summary() output above as reference
   
5. Train with appropriate hyperparameters:
   - Optimizer: Adam (learning_rate=0.001)
   - Loss: categorical_crossentropy (for multi-class)
   - Metrics: accuracy, precision, recall, f1-score

6. Use MLflow for experiment tracking
""")

# Save model information to JSON
import json

model_info = {
    "input_shape": [int(x) if x is not None else None for x in model.input_shape],
    "output_shape": [int(x) if x is not None else None for x in model.output_shape],
    "window_size": int(window_size) if isinstance(window_size, int) else None,
    "num_features": int(num_features) if isinstance(num_features, int) else None,
    "num_classes": int(num_classes) if isinstance(num_classes, int) else None,
    "total_params": int(model.count_params()),
    "layer_types": layer_types
}

output_path = BASE_DIR / "model" / "model_info.json"
with open(output_path, "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\n💾 Model information saved to: {output_path}")

print("\n" + "="*80)
print("INSPECTION COMPLETE!")
print("="*80)
print(f"\nNext: Run 'python src/analyze_data.py' to analyze your data files")
