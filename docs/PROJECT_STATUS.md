# Project Status (Nov 5, 2025)

Single source of truth for current state. This replaces other ad‑hoc notes created today.

## Summary

- Goal: production‑ready pipeline for activity recognition using IMU data
- Current blocker: production accelerometer units differ from training → inputs become out‑of‑distribution for the pretrained model

## What’s done

- Clean repo structure and reproducible preprocessing for training data (StandardScaler; 200‑sample windows; 50% overlap)
- Verified that applying the training scaler to production exposes a massive accelerometer unit mismatch
- Gyroscope appears consistent across datasets

## Where we’re stuck (and why)

- Accelerometer (Ax/Ay/Az) in production has values tens to hundreds of times larger than training
- Likely different units (e.g., raw counts vs m/s² or g) or different device/calibration
- Without unit alignment, the pretrained model’s inputs are invalid and predictions untrustworthy

## Decision

- Do not use any workaround that fits new scalers on production only
- Proceed only after unit alignment or an agreed conversion formula

## Next steps (minimal)

1. Ask data provider/mentor for: exact units for training vs production; any scaling/calibration; conversion formula for Ax/Ay/Az
2. If conversion is provided, implement unit conversion → then apply the original training scaler
3. If conversion is not available soon, consider collecting a small labeled subset in production conditions and retrain/finetune a model for that distribution

## Pointers

- Detailed differences: see `docs/DATASET_DIFFERENCE_SUMMARY.md`
- Prepared data artifacts (training/val/test) remain under `data/prepared/`

## Data contract (units)

- Expected accelerometer units for training-compatible data: m/s² or g (consistent with training scaler); gyroscope: deg/s or rad/s as per training. Production must be converted to the same units before applying the training scaler.

## Mentor email (ready to send)

Subject: Production accelerometer units differ from training (blocking)

Body:

Hi [Name],

our production accelerometer values are tens to hundreds of times larger than training (e.g., Az mean ≈ -1001.6 vs training ≈ -3.5). Gyroscope channels look consistent. This indicates a unit/calibration mismatch on Ax/Ay/Az.

Could you please confirm:
1) The units used for accelerometer in training vs production,
2) Any scaling/calibration applied in the export pipeline,
3) The conversion formula to align production Ax/Ay/Az with training.

Once we have the conversion, we’ll convert production to training units and then reuse the original training scaler. If conversion isn’t available soon, we’ll need a small labeled subset in production conditions to retrain/finetune.

Thanks,
[Your Name]
