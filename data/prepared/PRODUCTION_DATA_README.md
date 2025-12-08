# Production Data - Preprocessed for Inference

**Created:** 2025-12-07 16:08:59  
**Source:** `data/processed/sensor_fused_50Hz_converted.csv`  
**Pipeline:** `src/prepare_production_data.py`

---

## Files

- `production_X.npy` - Preprocessed sensor windows (no labels)
- `production_metadata.json` - Window metadata (timestamps, indices)

---

## Preprocessing Applied

1. **Unit Conversion:** Accelerometer converted from milliG to m/s² (factor: 0.00981)
2. **Sensor Extraction:** Ax, Ay, Az, Gx, Gy, Gz
3. **Normalization:** StandardScaler (using training data parameters)
4. **Windowing:** 200 samples, 50% overlap
5. **Output:** (N, 200, 6) float32 array

---

## Usage for Inference

```python
import numpy as np
import tensorflow as tf

# Load preprocessed data
X = np.load('data/prepared/production_X.npy')
print(f"Shape: {X.shape}")  # (N, 200, 6)

# Load model
model = tf.keras.models.load_model('models/pretrained/fine_tuned_model_1dcnnbilstm.keras')

# Predict
predictions = model.predict(X)
pred_classes = np.argmax(predictions, axis=1)
pred_confidence = np.max(predictions, axis=1)

print(f"Predicted classes: {pred_classes[:10]}")
print(f"Confidence scores: {pred_confidence[:10]}")
```

---

## Comparison with Training Data

- Same normalization (StandardScaler from training)
- Same window size (200 samples)
- Same overlap (50%)
- Same sensor order (Ax, Ay, Az, Gx, Gy, Gz)

**Ready for production inference!** ✓

---

## Next Steps

1. Run inference: `python src/inference/predict.py`
2. Monitor predictions: `python src/monitoring/drift_detection.py`
3. Analyze results: `python src/evaluation/analyze_production.py`
