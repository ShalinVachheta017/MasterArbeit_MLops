# Data Preparation Summary - November 4, 2025

## ✅ PREPARED DATA READY FOR TRAINING

### 📊 Dataset Overview

**Source**: `all_users_data_labeled.csv` (385,326 samples)
**Output**: `01_data/prepared/` directory
**Created**: 2025-11-04 16:09:24

### 🎯 Task: 11-Class Activity Recognition

**Activities (Anxiety-Related Behaviors)**:
- 0: ear_rubbing
- 1: forehead_rubbing
- 2: hair_pulling
- 3: hand_scratching
- 4: hand_tapping
- 5: knuckles_cracking
- 6: nail_biting
- 7: nape_rubbing
- 8: sitting
- 9: smoking
- 10: standing

### 🔧 Preprocessing Pipeline

**Window Configuration**:
- Window size: 200 timesteps (4 seconds at 50Hz)
- Overlap: 50% (100 timesteps)
- Total windows created: 3,852

**Normalization** (StandardScaler):
```
Mean: [3.22, 1.28, -3.53, 0.60, 0.23, 0.09]
Std:  [6.57, 4.35, 3.24, 49.93, 14.81, 14.17]
```

**Sensor Columns**: `[Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w]`

### 📦 Data Splits (By User - No Data Leakage!)

| Split | Users | Windows | Percentage |
|-------|-------|---------|------------|
| **Train** | [1, 2, 3, 4] | 2,538 | 65.9% |
| **Val** | [5] | 641 | 16.6% |
| **Test** | [6] | 673 | 17.5% |

### 📁 Output Files

```
01_data/prepared/
├── train_X.npy          # (2538, 200, 6) - Training windows
├── train_y.npy          # (2538,) - Training labels
├── train_metadata.json  # Window metadata (timestamps, users, activities)
├── val_X.npy            # (641, 200, 6) - Validation windows
├── val_y.npy            # (641,) - Validation labels
├── val_metadata.json
├── test_X.npy           # (673, 200, 6) - Test windows
├── test_y.npy           # (673,) - Test labels
├── test_metadata.json
└── config.json          # Full configuration + scaler parameters
```

### 🚀 Ready for Training!

**Input Shape**: `(batch_size, 200, 6)`
**Output Shape**: `(batch_size, 11)` - One-hot encoded

**Next Steps**:
1. ✅ Data prepared
2. ⏳ Build training script (`02_src/training/train_model.py`)
3. ⏳ Set up MLflow experiment tracking
4. ⏳ Train 1D-CNN-BiLSTM model
5. ⏳ Evaluate on test set

---

## 📝 Important Notes

### Production Data Format
**During deployment**, incoming data will have the format:
```
Columns: [timestamp, Ax, Ay, Az, Gx, Gy, Gz]  # NO activity labels!
```

**Processing Pipeline for Production**:
1. Load raw sensor data (6 axes + timestamp)
2. Apply same normalization (use saved scaler parameters from config.json)
3. Create sliding windows (200 timesteps, 50% overlap)
4. Feed to model for prediction
5. Model outputs: 11-class probabilities

### Key Differences: Training vs Production

| Aspect | Training Data | Production Data |
|--------|---------------|-----------------|
| **Columns** | `Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, activity, User` | `Ax, Ay, Az, Gx, Gy, Gz` |
| **Labels** | ✅ Has activity labels | ❌ No labels |
| **Sample Rate** | 50 Hz | 50 Hz (same) |
| **Value Range** | ±20-30 (normalized) | ±1000-2000 (raw) |
| **Normalization** | Fitted during training | **Must apply same scaler** |

### 🔑 Critical for Deployment

**Always normalize production data using the same scaler**:
```python
import json
import numpy as np

# Load scaler parameters
with open('01_data/prepared/config.json') as f:
    config = json.load(f)

mean = np.array(config['scaler_mean'])
std = np.array(config['scaler_scale'])

# Normalize incoming data
production_data_normalized = (production_data - mean) / std
```

---

## 🎯 Summary

✅ **Data prepared successfully for training**
✅ **Pipeline handles both labeled (training) and unlabeled (production) data**
✅ **Scaler parameters saved for production deployment**
✅ **User-based splits prevent data leakage**
✅ **3,852 windows ready for model training**

**Ready to proceed with Month 1 Week 4: Model Training! 🚀**
