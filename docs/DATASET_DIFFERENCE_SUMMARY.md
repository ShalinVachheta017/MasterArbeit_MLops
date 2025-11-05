# Dataset Difference Summary (Nov 5, 2025)

This document captures the exact problem blocking production: the production accelerometer data has a fundamentally different scale/units than the training data.

## What differs between datasets

- Channels: 6 sensors total in both datasets
  - Accelerometer: Ax, Ay, Az
  - Gyroscope: Gx, Gy, Gz
- Windowing used in training: 200 samples per window, 50% overlap

## Accelerometer vs Gyroscope

- Gyroscope (Gx, Gy, Gz):
  - Training and production have similar ranges and means
  - Conclusion: Gyroscope units are consistent across datasets
- Accelerometer (Ax, Ay, Az):
  - Production values are tens to hundreds of times larger than training values
  - Strong negative bias on Az in production (≈ -1001 mean), not physically plausible in m/s² or g
  - Conclusion: Accelerometer units/calibration differ between training and production

## Raw statistics (representative)

- Training (raw):
  - Ax/Ay/Az means ≈ [3.2, 1.3, -3.5], std ≈ [6.6, 4.4, 3.2]
  - Gx/Gy/Gz std ≈ [49.9, 14.8, 14.2]
- Production (raw after NaN fill):
  - Ax/Ay/Az means ≈ [-16.2, -19.0, -1001.6], std ≈ [11.3, 31.0, 19.9]
  - Gx/Gy/Gz std ≈ [5.3, 4.7, 1.9]

Key takeaway:
- Gyroscope: compatible
- Accelerometer: incompatible (unit mismatch likely)

## Why they differ (most likely causes)

1. Different accelerometer units (e.g., raw ADC counts vs m/s² or g)
2. Different device or calibration in production
3. Recording/export pipeline applying different scaling for accelerometer only

## Impact on the model

- Applying the training StandardScaler to production accelerometer values creates extreme normalized values far outside the training distribution.
- This results in invalid inputs to the model and poor predictions.

## Decision

- Do not ship a workaround (e.g., refitting only accelerometer scaler) because it changes the semantics of the input and hides the real issue.
- Fix the data source or convert units properly so production matches training.

## What’s needed to proceed

- Confirm the production accelerometer units and provide the conversion formula to align with training (e.g., raw counts → g → m/s², or directly to training units).
- If unit alignment is not possible soon, prepare a small labeled subset of production-style data and retrain/finetune a model for that distribution.

## Minimal action items

- Contact data provider/mentor with the above findings and request:
  - Sensor units for training and production datasets
  - Any applied scaling factors or calibration steps in the collection/export pipeline
  - Conversion formula for Ax/Ay/Az to match training units
