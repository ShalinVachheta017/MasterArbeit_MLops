"""
Convert Production Data Units: milliG to m/s²

This script fixes the unit mismatch between training and production data:
- Training data: Accelerometer already in m/s²
- Production data: Accelerometer still in milliG (needs conversion)
- Conversion factor: 0.00981 (milliG → m/s²)

Author: [Your Name]
Date: December 3, 2025
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Conversion factor from mentor
CONVERSION_FACTOR = 0.00981  # milliG to m/s²

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRODUCTION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sensor_fused_50Hz.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sensor_fused_50Hz_converted.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "preprocessing" / "unit_conversion.log"


def convert_production_data():
    """
    Convert production accelerometer data from milliG to m/s².
    Gyroscope data remains unchanged (already compatible).
    """

    print("=" * 80)
    print("PRODUCTION DATA UNIT CONVERSION")
    print("=" * 80)
    print(f"Input:  {PRODUCTION_DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Conversion Factor: {CONVERSION_FACTOR} (milliG → m/s²)")
    print("=" * 80)

    # Load production data
    print("\n[1/5] Loading production data...")
    df = pd.read_csv(PRODUCTION_DATA_PATH)
    print(f"✓ Loaded {len(df):,} samples")
    print(f"✓ Columns: {list(df.columns)}")

    # Verify expected columns exist
    accel_cols = ["Ax", "Ay", "Az"]
    gyro_cols = ["Gx", "Gy", "Gz"]

    for col in accel_cols + gyro_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Show BEFORE statistics
    print("\n[2/5] Statistics BEFORE conversion:")
    print("-" * 80)
    print("ACCELEROMETER (milliG):")
    for col in accel_cols:
        print(
            f"  {col}: mean={df[col].mean():10.3f}, std={df[col].std():10.3f}, "
            f"min={df[col].min():10.3f}, max={df[col].max():10.3f}"
        )
    print("\nGYROSCOPE (unchanged):")
    for col in gyro_cols:
        print(
            f"  {col}: mean={df[col].mean():10.3f}, std={df[col].std():10.3f}, "
            f"min={df[col].min():10.3f}, max={df[col].max():10.3f}"
        )

    # Apply conversion to accelerometer ONLY
    print("\n[3/5] Applying conversion to accelerometer channels...")
    df_converted = df.copy()

    for col in accel_cols:
        df_converted[col] = df[col] * CONVERSION_FACTOR
        print(f"✓ Converted {col}: milliG → m/s²")

    print("✓ Gyroscope channels unchanged (already compatible)")

    # Show AFTER statistics
    print("\n[4/5] Statistics AFTER conversion:")
    print("-" * 80)
    print("ACCELEROMETER (m/s²):")
    for col in accel_cols:
        old_mean = df[col].mean()
        new_mean = df_converted[col].mean()
        print(
            f"  {col}: mean={new_mean:10.3f} (was {old_mean:10.3f}), "
            f"std={df_converted[col].std():10.3f}, "
            f"min={df_converted[col].min():10.3f}, max={df_converted[col].max():10.3f}"
        )
    print("\nGYROSCOPE (unchanged):")
    for col in gyro_cols:
        print(
            f"  {col}: mean={df_converted[col].mean():10.3f}, std={df_converted[col].std():10.3f}, "
            f"min={df_converted[col].min():10.3f}, max={df_converted[col].max():10.3f}"
        )

    # Compare with training data expectations
    print("\n[5/5] Validation against training data:")
    print("-" * 80)
    print("Expected training data statistics (for reference):")
    print("  Ax: mean ≈ 3.2,   std ≈ 6.6")
    print("  Ay: mean ≈ 1.3,   std ≈ 4.4")
    print("  Az: mean ≈ -3.5,  std ≈ 3.2")
    print("\nConverted production data:")
    print(f"  Ax: mean ≈ {df_converted['Ax'].mean():.1f},   std ≈ {df_converted['Ax'].std():.1f}")
    print(f"  Ay: mean ≈ {df_converted['Ay'].mean():.1f},   std ≈ {df_converted['Ay'].std():.1f}")
    print(f"  Az: mean ≈ {df_converted['Az'].mean():.1f},  std ≈ {df_converted['Az'].std():.1f}")

    # Check if distributions are now reasonable
    az_mean_converted = df_converted["Az"].mean()
    az_mean_training = -3.5

    if abs(az_mean_converted - az_mean_training) < 20:  # Within reasonable range
        print("\n✓ SUCCESS: Converted Az mean is now closer to training data!")
        print(f"  Converted Az: {az_mean_converted:.1f} vs Training Az: {az_mean_training:.1f}")
    else:
        print("\n⚠ WARNING: Converted values still significantly different from training")
        print(f"  Converted Az: {az_mean_converted:.1f} vs Training Az: {az_mean_training:.1f}")
        print("  This might indicate additional scaling issues.")

    # Save converted data
    print(f"\nSaving converted data to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_converted.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Saved {len(df_converted):,} samples")

    # Create conversion log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "conversion_factor": CONVERSION_FACTOR,
        "input_file": str(PRODUCTION_DATA_PATH),
        "output_file": str(OUTPUT_PATH),
        "samples": len(df_converted),
        "converted_columns": accel_cols,
        "unchanged_columns": gyro_cols,
        "statistics_before": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in accel_cols + gyro_cols
        },
        "statistics_after": {
            col: {
                "mean": float(df_converted[col].mean()),
                "std": float(df_converted[col].std()),
                "min": float(df_converted[col].min()),
                "max": float(df_converted[col].max()),
            }
            for col in accel_cols + gyro_cols
        },
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"✓ Conversion log saved: {LOG_PATH}")

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"✓ Converted file: {OUTPUT_PATH}")
    print(f"✓ Next step: Run inference pipeline with converted data")
    print("=" * 80)

    return df_converted


if __name__ == "__main__":
    try:
        df_converted = convert_production_data()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
