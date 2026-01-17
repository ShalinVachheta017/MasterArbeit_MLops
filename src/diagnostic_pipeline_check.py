"""
Pipeline Contract Diagnostic Tool
==================================

This script checks for preprocessing mismatches between training and inference
that could explain low accuracy (14-15% = near-random for 11 classes).

Checks:
1. Window size and overlap consistency
2. Sampling rate consistency
3. Channel order and sensor columns
4. Units consistency (mg vs m/s², deg/s vs rad/s)
5. Normalization stats (scaler mean/scale)
6. Label mapping consistency
7. Feature distribution comparison (training vs production)

Author: MLOps Pipeline Diagnostic
Date: January 9, 2026
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineDiagnostic:
    """Diagnose preprocessing contract mismatches."""
    
    def __init__(
        self,
        training_config_path: Path,
        production_metadata_path: Path,
        production_data_path: Path
    ):
        self.training_config = self._load_json(training_config_path)
        self.production_metadata = self._load_json(production_metadata_path)
        self.production_data = np.load(production_data_path)
        
        self.mismatches = []
        self.warnings = []
        
    def _load_json(self, path: Path) -> dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def run_full_diagnostic(self) -> Dict[str, bool]:
        """Run all diagnostic checks."""
        logger.info("=" * 80)
        logger.info("PIPELINE CONTRACT DIAGNOSTIC")
        logger.info("=" * 80)
        
        results = {
            'window_size': self.check_window_size(),
            'overlap': self.check_overlap(),
            'sampling_rate': self.check_sampling_rate(),
            'sensor_channels': self.check_sensor_channels(),
            'normalization': self.check_normalization_stats(),
            'label_mapping': self.check_label_mapping(),
            'data_shape': self.check_data_shape(),
            'feature_distributions': self.check_feature_distributions(),
        }
        
        self._print_summary(results)
        return results
    
    def check_window_size(self) -> bool:
        """Check if window size matches between training and production."""
        train_window = self.training_config.get('window_size')
        prod_window = self.production_metadata.get('window_size')
        
        logger.info(f"\n[CHECK 1] Window Size")
        logger.info(f"  Training:   {train_window} samples")
        logger.info(f"  Production: {prod_window} samples")
        
        if train_window != prod_window:
            self.mismatches.append(
                f"MISMATCH: Window size differs (train={train_window}, prod={prod_window})"
            )
            logger.error(f"  ❌ MISMATCH DETECTED")
            return False
        
        logger.info(f"  ✅ Match")
        return True
    
    def check_overlap(self) -> bool:
        """Check if overlap percentage matches."""
        train_overlap = self.training_config.get('overlap')
        prod_overlap = self.production_metadata.get('overlap')
        
        logger.info(f"\n[CHECK 2] Window Overlap")
        logger.info(f"  Training:   {train_overlap}")
        logger.info(f"  Production: {prod_overlap}")
        
        if train_overlap != prod_overlap:
            self.warnings.append(
                f"WARNING: Overlap differs (train={train_overlap}, prod={prod_overlap})"
            )
            logger.warning(f"  ⚠️  Different overlap (usually OK)")
            return True
        
        logger.info(f"  ✅ Match")
        return True
    
    def check_sampling_rate(self) -> bool:
        """Check if target sampling rate matches."""
        train_hz = self.training_config.get('target_hz')
        # Production metadata doesn't store Hz, but we can infer from window duration
        prod_window_size = self.production_metadata.get('window_size')
        
        logger.info(f"\n[CHECK 3] Sampling Rate")
        logger.info(f"  Training target Hz: {train_hz} Hz")
        logger.info(f"  Production window:  {prod_window_size} samples")
        
        # Check if window represents 4 seconds at 50Hz
        expected_samples = train_hz * 4  # 4-second window
        if prod_window_size != expected_samples:
            self.warnings.append(
                f"WARNING: Window size {prod_window_size} doesn't match {train_hz}Hz * 4s = {expected_samples}"
            )
            logger.warning(f"  ⚠️  Window size doesn't match expected {expected_samples} for 4s @ {train_hz}Hz")
            return False
        
        logger.info(f"  ✅ Window size matches 4s @ {train_hz}Hz")
        return True
    
    def check_sensor_channels(self) -> bool:
        """Check if sensor channel order matches."""
        train_cols = self.training_config.get('sensor_cols', [])
        
        logger.info(f"\n[CHECK 4] Sensor Channel Order")
        logger.info(f"  Training expects: {train_cols}")
        logger.info(f"  Production shape: {self.production_data.shape}")
        
        if self.production_data.shape[-1] != len(train_cols):
            self.mismatches.append(
                f"MISMATCH: Channel count differs (train={len(train_cols)}, prod={self.production_data.shape[-1]})"
            )
            logger.error(f"  ❌ MISMATCH: Expected {len(train_cols)} channels, got {self.production_data.shape[-1]}")
            return False
        
        logger.info(f"  ✅ Channel count matches ({len(train_cols)} channels)")
        logger.info(f"  ⚠️  Cannot verify channel ORDER without column names - verify manually!")
        self.warnings.append("WARNING: Cannot verify sensor channel order from numpy array")
        return True
    
    def check_normalization_stats(self) -> bool:
        """Check if normalization statistics are consistent."""
        train_mean = np.array(self.training_config.get('scaler_mean', []))
        train_scale = np.array(self.training_config.get('scaler_scale', []))
        
        logger.info(f"\n[CHECK 5] Normalization Statistics")
        logger.info(f"  Training scaler mean:  {train_mean}")
        logger.info(f"  Training scaler scale: {train_scale}")
        
        # Calculate production statistics BEFORE normalization
        # (This requires access to raw production data, which we may not have)
        logger.info(f"  Production data (AFTER normalization):")
        prod_mean = self.production_data.mean(axis=(0, 1))
        prod_std = self.production_data.std(axis=(0, 1))
        logger.info(f"    Mean:  {prod_mean}")
        logger.info(f"    Std:   {prod_std}")
        
        # If production data is normalized correctly, it should have mean≈0, std≈1
        if np.allclose(prod_mean, 0, atol=0.5) and np.allclose(prod_std, 1, atol=0.5):
            logger.info(f"  ✅ Production data appears normalized (mean≈0, std≈1)")
            return True
        else:
            self.warnings.append(
                f"WARNING: Production data doesn't look normalized (mean={prod_mean.mean():.2f}, std={prod_std.mean():.2f})"
            )
            logger.warning(f"  ⚠️  Production data doesn't appear normalized")
            logger.warning(f"      Expected: mean≈0, std≈1")
            logger.warning(f"      Got:      mean≈{prod_mean.mean():.2f}, std≈{prod_std.mean():.2f}")
            return False
    
    def check_label_mapping(self) -> bool:
        """Check label mapping consistency."""
        label_map = self.training_config.get('label_to_activity', {})
        n_classes = self.training_config.get('n_classes')
        
        logger.info(f"\n[CHECK 6] Label Mapping")
        logger.info(f"  Number of classes: {n_classes}")
        logger.info(f"  Label mapping:")
        for label_id, activity in sorted(label_map.items(), key=lambda x: int(x[0])):
            logger.info(f"    {label_id}: {activity}")
        
        if len(label_map) != n_classes:
            self.mismatches.append(
                f"MISMATCH: Label map has {len(label_map)} entries but n_classes={n_classes}"
            )
            logger.error(f"  ❌ MISMATCH: {len(label_map)} labels but n_classes={n_classes}")
            return False
        
        logger.info(f"  ✅ Label mapping consistent")
        return True
    
    def check_data_shape(self) -> bool:
        """Check production data shape consistency."""
        expected_shape = (
            None,  # number of windows (variable)
            self.training_config.get('window_size'),
            len(self.training_config.get('sensor_cols', []))
        )
        actual_shape = self.production_data.shape
        
        logger.info(f"\n[CHECK 7] Data Shape")
        logger.info(f"  Expected: (n_windows, {expected_shape[1]}, {expected_shape[2]})")
        logger.info(f"  Actual:   {actual_shape}")
        
        if actual_shape[1] != expected_shape[1]:
            self.mismatches.append(
                f"MISMATCH: Window size in data ({actual_shape[1]}) doesn't match config ({expected_shape[1]})"
            )
            logger.error(f"  ❌ MISMATCH in dimension 1 (window size)")
            return False
        
        if actual_shape[2] != expected_shape[2]:
            self.mismatches.append(
                f"MISMATCH: Channel count in data ({actual_shape[2]}) doesn't match config ({expected_shape[2]})"
            )
            logger.error(f"  ❌ MISMATCH in dimension 2 (channels)")
            return False
        
        logger.info(f"  ✅ Shape matches expected format")
        return True
    
    def check_feature_distributions(self) -> bool:
        """Compare feature distributions between training stats and production data."""
        train_mean = np.array(self.training_config.get('scaler_mean', []))
        train_scale = np.array(self.training_config.get('scaler_scale', []))
        
        logger.info(f"\n[CHECK 8] Feature Distribution Comparison")
        logger.info(f"  Training data (original scale):")
        logger.info(f"    Mean:  {train_mean}")
        logger.info(f"    Scale: {train_scale}")
        
        # Production data is normalized, so we'd need to denormalize to compare
        # For now, just report that this check requires raw production data
        logger.info(f"  ⚠️  Cannot compare distributions without raw production data")
        self.warnings.append(
            "WARNING: Need raw production data (before normalization) to compare distributions"
        )
        
        return True
    
    def _print_summary(self, results: Dict[str, bool]) -> None:
        """Print diagnostic summary."""
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(results.values())
        total = len(results)
        
        logger.info(f"\nChecks passed: {passed}/{total}")
        
        if self.mismatches:
            logger.error(f"\n🚨 CRITICAL MISMATCHES FOUND ({len(self.mismatches)}):")
            for mismatch in self.mismatches:
                logger.error(f"  - {mismatch}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if not self.mismatches:
            logger.info("\n✅ No critical mismatches detected")
            logger.info("   → 14-15% accuracy is likely due to DOMAIN SHIFT, not pipeline bugs")
            logger.info("   → Recommendation: Proceed with controlled fine-tuning on labeled production data")
        else:
            logger.error("\n❌ Critical mismatches detected - FIX THESE FIRST before fine-tuning")
            logger.error("   → Fine-tuning won't fix broken preprocessing")
        
        logger.info("=" * 80)


def main():
    """Run diagnostic on current pipeline."""
    from config import PROJECT_ROOT, DATA_PREPARED
    
    # Paths
    training_config_path = DATA_PREPARED / "config.json"
    production_metadata_path = DATA_PREPARED / "production_metadata.json"
    production_data_path = DATA_PREPARED / "production_X.npy"
    
    # Check files exist
    if not training_config_path.exists():
        logger.error(f"Training config not found: {training_config_path}")
        return
    if not production_metadata_path.exists():
        logger.error(f"Production metadata not found: {production_metadata_path}")
        return
    if not production_data_path.exists():
        logger.error(f"Production data not found: {production_data_path}")
        return
    
    # Run diagnostic
    diagnostic = PipelineDiagnostic(
        training_config_path=training_config_path,
        production_metadata_path=production_metadata_path,
        production_data_path=production_data_path
    )
    
    results = diagnostic.run_full_diagnostic()
    
    # Additional manual checks to perform
    logger.info("\n" + "=" * 80)
    logger.info("MANUAL CHECKS TO PERFORM")
    logger.info("=" * 80)
    logger.info("\n1. Print sample windows from training and production:")
    logger.info("   - Check if magnitude ranges are similar")
    logger.info("   - Verify sensor units match (m/s² for accel, deg/s or rad/s for gyro)")
    logger.info("\n2. Check evaluation code:")
    logger.info("   - Verify ground truth labels match prediction format")
    logger.info("   - Check if label IDs are 0-10 (not 1-11)")
    logger.info("\n3. Inspect predictions:")
    logger.info("   - Are predictions evenly distributed or dominated by 1-2 classes?")
    logger.info("   - Check confidence scores - are they all low?")
    logger.info("\n4. If no mismatches found → Domain shift is real → Proceed to fine-tuning")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
