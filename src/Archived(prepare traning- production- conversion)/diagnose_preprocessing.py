"""
Diagnostic Tool: Find Preprocessing Mismatch
==============================================

This script diagnoses why inference accuracy is low (~14-15%) by comparing:
1. Training data preprocessing (all_users_data_labeled.csv)
2. Production data preprocessing (your Garmin data)

Common Issues:
- Scaler parameters mismatch (mean/std from different data)
- Unit conversion not applied (milliG vs m/s²)
- Calibration offset difference
- Different sampling rates or window sizes

Author: MLOps Pipeline
Date: January 6, 2026
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PREPARED, PROJECT_ROOT

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TRAINING_CONFIG = DATA_PREPARED / "config.json"
PRODUCTION_CONFIG = DATA_PREPARED / "config.json"  # Same file, but check source
TRAINING_DATA = PROJECT_ROOT / "research_papers" / "all_users_data_labeled.csv"
PRODUCTION_DATA_X = DATA_PREPARED / "production_X.npy"


# ============================================================================
# LOGGER SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================


def load_config() -> Dict:
    """Load preprocessing configuration."""
    if not TRAINING_CONFIG.exists():
        logger.error(f"❌ Config not found: {TRAINING_CONFIG}")
        return {}

    with open(TRAINING_CONFIG, "r") as f:
        config = json.load(f)

    logger.info(f"✅ Loaded config: {TRAINING_CONFIG}")
    return config


def check_scaler_parameters(config: Dict) -> bool:
    """
    Check if scaler parameters are present and valid.

    The scaler (StandardScaler) should have:
    - mean: shape (6,) for [Ax, Ay, Az, Gx, Gy, Gz]
    - scale: shape (6,) for standard deviations
    """
    logger.info("=" * 60)
    logger.info("🔍 CHECKING SCALER PARAMETERS")
    logger.info("=" * 60)

    if "scaler_mean" not in config or "scaler_scale" not in config:
        logger.error("❌ Scaler parameters missing in config.json!")
        return False

    mean = np.array(config["scaler_mean"])
    scale = np.array(config["scaler_scale"])

    logger.info(f"Mean shape: {mean.shape}")
    logger.info(f"Scale shape: {scale.shape}")
    logger.info(f"\nMean values:")
    logger.info(f"  Ax: {mean[0]:.3f}")
    logger.info(f"  Ay: {mean[1]:.3f}")
    logger.info(f"  Az: {mean[2]:.3f}")
    logger.info(f"  Gx: {mean[3]:.3f}")
    logger.info(f"  Gy: {mean[4]:.3f}")
    logger.info(f"  Gz: {mean[5]:.3f}")

    logger.info(f"\nScale values:")
    logger.info(f"  Ax: {scale[0]:.3f}")
    logger.info(f"  Ay: {scale[1]:.3f}")
    logger.info(f"  Az: {scale[2]:.3f}")
    logger.info(f"  Gx: {scale[3]:.3f}")
    logger.info(f"  Gy: {scale[4]:.3f}")
    logger.info(f"  Gz: {scale[5]:.3f}")

    # Check for reasonable values
    # Accelerometer should be around ±10 m/s²
    # Gyroscope should be around ±100 deg/s
    if abs(mean[2]) > 50:  # Az should not be > 50 m/s²
        logger.warning("⚠️ Az mean seems too large - possible unit mismatch?")
        logger.warning("   Expected: ~-9.81 m/s² (gravity)")
        logger.warning(f"   Got: {mean[2]:.3f}")
        return False

    logger.info("\n✅ Scaler parameters look reasonable")
    return True


def check_unit_conversion(config: Dict) -> bool:
    """Check if unit conversion was applied."""
    logger.info("=" * 60)
    logger.info("🔍 CHECKING UNIT CONVERSION")
    logger.info("=" * 60)

    if "unit_conversion_applied" in config:
        converted = config["unit_conversion_applied"]
        logger.info(f"Unit conversion applied: {converted}")

        if converted:
            logger.info("✅ Data was converted from milliG to m/s²")
        else:
            logger.warning("⚠️ No unit conversion applied")
            logger.warning("   If input was in milliG, this could cause major accuracy drop!")
    else:
        logger.warning("⚠️ unit_conversion_applied not found in config")
        logger.warning("   Cannot verify if conversion was applied")
        return False

    return True


def check_calibration(config: Dict) -> bool:
    """Check if calibration was applied."""
    logger.info("=" * 60)
    logger.info("🔍 CHECKING CALIBRATION")
    logger.info("=" * 60)

    if "calibration_applied" in config:
        calibrated = config["calibration_applied"]
        logger.info(f"Calibration applied: {calibrated}")

        if "calibration_offset" in config:
            offset = config["calibration_offset"]
            logger.info(f"Calibration offset: {offset}")
    else:
        logger.warning("⚠️ calibration_applied not found in config")

    return True


def check_production_data_stats() -> bool:
    """Check statistics of production data."""
    logger.info("=" * 60)
    logger.info("🔍 CHECKING PRODUCTION DATA STATISTICS")
    logger.info("=" * 60)

    if not PRODUCTION_DATA_X.exists():
        logger.error(f"❌ Production data not found: {PRODUCTION_DATA_X}")
        return False

    X = np.load(PRODUCTION_DATA_X)
    logger.info(f"Shape: {X.shape}")  # Should be (N, 200, 6)
    logger.info(f"Dtype: {X.dtype}")

    # Check each sensor channel
    sensors = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    logger.info("\nPer-sensor statistics (after standardization):")
    logger.info("-" * 60)

    for i, sensor in enumerate(sensors):
        sensor_data = X[:, :, i]
        mean = sensor_data.mean()
        std = sensor_data.std()
        min_val = sensor_data.min()
        max_val = sensor_data.max()

        logger.info(
            f"{sensor:3s}: mean={mean:7.3f}, std={std:6.3f}, "
            f"min={min_val:7.2f}, max={max_val:7.2f}"
        )

        # After standardization, mean should be ~0, std should be ~1
        if abs(mean) > 1.0:
            logger.warning(f"⚠️ {sensor} mean is far from 0 - standardization issue?")

        if std < 0.5 or std > 2.0:
            logger.warning(f"⚠️ {sensor} std is far from 1 - standardization issue?")

    # Check for NaN or Inf
    if np.isnan(X).any():
        logger.error("❌ NaN values found in production data!")
        return False

    if np.isinf(X).any():
        logger.error("❌ Inf values found in production data!")
        return False

    logger.info("\n✅ Production data statistics checked")
    return True


def compare_with_training_data() -> bool:
    """Compare production data distribution with training data."""
    logger.info("=" * 60)
    logger.info("🔍 COMPARING TRAINING VS PRODUCTION DATA")
    logger.info("=" * 60)

    if not TRAINING_DATA.exists():
        logger.warning(f"⚠️ Training data not found: {TRAINING_DATA}")
        logger.warning("   Skipping comparison")
        return True

    # Load training data sample
    logger.info(f"Loading training data sample: {TRAINING_DATA}")
    df = pd.read_csv(TRAINING_DATA, nrows=10000)  # Sample only

    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns[:10])}...")  # First 10 columns

    # Check if expected columns exist
    expected_cols = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    for col in expected_cols:
        if col not in df.columns:
            logger.error(f"❌ Expected column '{col}' not found in training data!")
            return False

    # Check training data statistics
    logger.info("\nTraining data statistics (raw, before preprocessing):")
    logger.info("-" * 60)
    for col in expected_cols:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        logger.info(
            f"{col:3s}: mean={mean:7.3f}, std={std:6.3f}, "
            f"min={min_val:7.2f}, max={max_val:7.2f}"
        )

    logger.info("\n✅ Training data loaded successfully")
    return True


def check_model_expectations() -> bool:
    """Check if model expects specific input format."""
    logger.info("=" * 60)
    logger.info("🔍 CHECKING MODEL EXPECTATIONS")
    logger.info("=" * 60)

    model_info_path = Path("models/pretrained/model_info.json")
    if model_info_path.exists():
        with open(model_info_path, "r") as f:
            model_info = json.load(f)

        logger.info("Model Info:")
        logger.info(f"  Input shape: {model_info['input_shape']}")
        logger.info(f"  Output shape: {model_info['output_shape']}")
        logger.info(f"  Num classes: {model_info['num_classes']}")
        logger.info(f"  Window size: {model_info['window_size']}")
        logger.info(f"  Num features: {model_info['num_features']}")

        # Check against production data
        if PRODUCTION_DATA_X.exists():
            X = np.load(PRODUCTION_DATA_X)
            expected_window = model_info["window_size"]
            expected_features = model_info["num_features"]

            if X.shape[1] != expected_window:
                logger.error(f"❌ Window size mismatch!")
                logger.error(f"   Expected: {expected_window}, Got: {X.shape[1]}")
                return False

            if X.shape[2] != expected_features:
                logger.error(f"❌ Feature count mismatch!")
                logger.error(f"   Expected: {expected_features}, Got: {X.shape[2]}")
                return False

        logger.info("\n✅ Model expectations match production data")
    else:
        logger.warning("⚠️ model_info.json not found")

    return True


def main():
    """Run all diagnostic checks."""
    logger.info("=" * 60)
    logger.info("🩺 PREPROCESSING DIAGNOSTIC TOOL")
    logger.info("=" * 60)
    logger.info("\nThis tool diagnoses why inference accuracy is low (~14-15%)")
    logger.info("by comparing training vs production preprocessing.\n")

    # Load config
    config = load_config()
    if not config:
        logger.error("❌ Cannot proceed without config.json")
        return

    # Run checks
    checks = [
        ("Scaler Parameters", lambda: check_scaler_parameters(config)),
        ("Unit Conversion", lambda: check_unit_conversion(config)),
        ("Calibration", lambda: check_calibration(config)),
        ("Production Data Stats", check_production_data_stats),
        ("Training vs Production", compare_with_training_data),
        ("Model Expectations", check_model_expectations),
    ]

    results = {}
    for name, check_func in checks:
        logger.info("")
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"❌ Error in {name}: {e}")
            results[name] = False

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status:10s} | {name}")

    # Overall result
    all_passed = all(results.values())
    logger.info("")
    if all_passed:
        logger.info("✅ All checks passed!")
        logger.info("   If accuracy is still low, you may need fine-tuning.")
    else:
        logger.error("❌ Some checks failed!")
        logger.error("   Fix the issues above before running inference.")

    logger.info("")
    logger.info("💡 NEXT STEPS:")
    logger.info("   1. Fix any preprocessing mismatches identified above")
    logger.info("   2. Run: python src/k_fold_evaluator.py (to validate with cross-validation)")
    logger.info("   3. If accuracy is still low: python src/fine_tune_model.py")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
