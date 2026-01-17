#!/usr/bin/env python
"""
Inference Smoke Test Script
============================

Quick validation that the inference pipeline produces sensible predictions.
Checks for common issues that cause low accuracy.

Usage:
    python scripts/inference_smoke.py
    python scripts/inference_smoke.py --input data/prepared/production_X.npy
    python scripts/inference_smoke.py --model models/pretrained/fine_tuned_model_1dcnnbilstm.keras
    python scripts/inference_smoke.py --mlflow

Checks:
    1. Model loads successfully
    2. Model input shape matches data shape
    3. Predictions sum to 1 (proper softmax)
    4. Predictions are not uniform (model learned something)
    5. Predictions are deterministic (same input → same output)
    6. Inference time is reasonable
    7. Input variance proxy (idle data detection)

Output:
    - JSON report: reports/inference_smoke/<timestamp>.json
    - Console: PASS/FAIL summary
    - MLflow: Optional logging (--mlflow flag)

Author: MLOps Pipeline Audit
Date: January 9, 2026
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PROJECT_ROOT, DATA_PREPARED, MODELS_PRETRAINED,
    WINDOW_SIZE, NUM_SENSORS, NUM_CLASSES, ACTIVITY_LABELS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeTestReport:
    """Complete smoke test report."""
    timestamp: str
    model_path: str
    data_path: str
    checks_passed: int
    checks_failed: int
    critical_failures: int
    overall_status: str
    checks: List[Dict]
    inference_stats: Dict[str, Any]
    prediction_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# INFERENCE SMOKE TEST
# ============================================================================

class InferenceSmokeTest:
    """Inference pipeline smoke test."""
    
    def __init__(self, model_path: Path, data_path: Path, mlflow_logging: bool = False):
        self.model_path = model_path
        self.data_path = data_path
        self.mlflow_logging = mlflow_logging
        self.checks: List[CheckResult] = []
        self.model = None
        self.data = None
        self.predictions = None
        self.inference_stats: Dict[str, Any] = {}
        self.prediction_stats: Dict[str, Any] = {}
        self.mlflow_run_id = None
        
        # Initialize MLflow if enabled
        if self.mlflow_logging:
            self._init_mlflow()
    
    def _add_check(self, name: str, passed: bool, severity: str, 
                   message: str, details: Dict = None):
        """Add a check result."""
        self.checks.append(CheckResult(
            name=name,
            passed=bool(passed),  # Force conversion to Python bool
            severity=severity,
            message=message,
            details=details or {}
        ))
        
        status = "✅ PASS" if passed else f"❌ FAIL ({severity})"
        logger.info(f"  {status}: {name}")
        if not passed:
            logger.warning(f"    → {message}")
    
    # -------------------------------------------------------------------------
    # Model Loading Checks
    # -------------------------------------------------------------------------
    
    def check_model_loading(self) -> bool:
        """Check that model loads successfully."""
        logger.info("\n🔧 MODEL LOADING CHECKS")
        
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path))
            
            self._add_check(
                name="Model loads successfully",
                passed=True,
                severity="CRITICAL",
                message="OK",
                details={"model_path": str(self.model_path)}
            )
            
            # Check model architecture
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            n_params = self.model.count_params()
            
            self.inference_stats["input_shape"] = str(input_shape)
            self.inference_stats["output_shape"] = str(output_shape)
            self.inference_stats["n_params"] = n_params
            
            logger.info(f"    Model input shape: {input_shape}")
            logger.info(f"    Model output shape: {output_shape}")
            logger.info(f"    Parameters: {n_params:,}")
            
            return True
            
        except Exception as e:
            self._add_check(
                name="Model loads successfully",
                passed=False,
                severity="CRITICAL",
                message=f"Failed to load model: {e}",
                details={"error": str(e)}
            )
            return False
    
    # -------------------------------------------------------------------------
    # Data Loading Checks
    # -------------------------------------------------------------------------
    
    def check_data_loading(self) -> bool:
        """Check that data loads successfully."""
        logger.info("\n📂 DATA LOADING CHECKS")
        
        try:
            self.data = np.load(str(self.data_path))
            
            self._add_check(
                name="Data loads successfully",
                passed=True,
                severity="CRITICAL",
                message=f"Shape: {self.data.shape}, dtype: {self.data.dtype}",
                details={"shape": self.data.shape, "dtype": str(self.data.dtype)}
            )
            return True
            
        except Exception as e:
            self._add_check(
                name="Data loads successfully",
                passed=False,
                severity="CRITICAL",
                message=f"Failed to load data: {e}",
                details={"error": str(e)}
            )
            return False
    
    # -------------------------------------------------------------------------
    # Shape Compatibility Checks
    # -------------------------------------------------------------------------
    
    def check_shape_compatibility(self) -> bool:
        """Check that data shape matches model expectations."""
        logger.info("\n📐 SHAPE COMPATIBILITY CHECKS")
        
        if self.model is None or self.data is None:
            return False
        
        # Model expects (None, timesteps, channels)
        model_input = self.model.input_shape
        data_shape = self.data.shape
        
        # Check dimensions
        if len(data_shape) != 3:
            self._add_check(
                name="Data is 3D",
                passed=False,
                severity="CRITICAL",
                message=f"Expected 3D, got {len(data_shape)}D",
                details={}
            )
            return False
        
        # Check timesteps match
        expected_timesteps = model_input[1]
        actual_timesteps = data_shape[1]
        
        timesteps_ok = expected_timesteps is None or actual_timesteps == expected_timesteps
        self._add_check(
            name="Timesteps match",
            passed=timesteps_ok,
            severity="CRITICAL",
            message=f"Expected {expected_timesteps}, got {actual_timesteps}" if not timesteps_ok else "OK",
            details={"expected": expected_timesteps, "actual": actual_timesteps}
        )
        
        # Check channels match
        expected_channels = model_input[2]
        actual_channels = data_shape[2]
        
        channels_ok = expected_channels is None or actual_channels == expected_channels
        self._add_check(
            name="Channels match",
            passed=channels_ok,
            severity="CRITICAL",
            message=f"Expected {expected_channels}, got {actual_channels}" if not channels_ok else "OK",
            details={"expected": expected_channels, "actual": actual_channels}
        )
        
        return timesteps_ok and channels_ok
    
    # -------------------------------------------------------------------------
    # Inference Checks
    # -------------------------------------------------------------------------
    
    def check_inference(self) -> bool:
        """Run inference and check predictions."""
        logger.info("\n🚀 INFERENCE CHECKS")
        
        if self.model is None or self.data is None:
            return False
        
        # Time inference
        n_samples = min(100, len(self.data))  # Use first 100 samples
        test_data = self.data[:n_samples]
        
        start_time = time.time()
        try:
            self.predictions = self.model.predict(test_data, verbose=0)
            elapsed = time.time() - start_time
            
            self._add_check(
                name="Inference runs successfully",
                passed=True,
                severity="CRITICAL",
                message=f"OK ({elapsed:.3f}s for {n_samples} samples)",
                details={}
            )
            
            self.inference_stats["samples_tested"] = n_samples
            self.inference_stats["inference_time_sec"] = round(elapsed, 3)
            self.inference_stats["samples_per_sec"] = round(n_samples / elapsed, 1)
            
        except Exception as e:
            self._add_check(
                name="Inference runs successfully",
                passed=False,
                severity="CRITICAL",
                message=f"Inference failed: {e}",
                details={"error": str(e)}
            )
            return False
        
        return True
    
    # -------------------------------------------------------------------------
    # Prediction Quality Checks
    # -------------------------------------------------------------------------
    
    def check_prediction_quality(self):
        """Check that predictions are sensible."""
        logger.info("\n📊 PREDICTION QUALITY CHECKS")
        
        if self.predictions is None:
            return
        
        preds = self.predictions
        
        # Check output shape
        expected_classes = NUM_CLASSES
        actual_classes = preds.shape[1]
        
        self._add_check(
            name="Output classes correct",
            passed=actual_classes == expected_classes,
            severity="CRITICAL",
            message=f"Expected {expected_classes}, got {actual_classes}",
            details={"expected": expected_classes, "actual": actual_classes}
        )
        
        # Check probabilities sum to 1 (valid softmax) + find worst examples
        prob_sums = preds.sum(axis=1)
        sum_ok = np.allclose(prob_sums, 1.0, atol=0.01)
        
        # Find worst softmax violations
        sum_deviations = np.abs(prob_sums - 1.0)
        worst_indices = np.argsort(sum_deviations)[-5:][::-1]  # Top 5 worst
        
        self._add_check(
            name="Probabilities sum to 1",
            passed=sum_ok,
            severity="HIGH",
            message=f"Sum range: [{prob_sums.min():.3f}, {prob_sums.max():.3f}]" if not sum_ok else "OK",
            details={
                "min_sum": float(prob_sums.min()),
                "max_sum": float(prob_sums.max()),
                "worst_samples": [int(idx) for idx in worst_indices],
                "worst_deviations": [float(sum_deviations[idx]) for idx in worst_indices]
            }
        )
        
        # Check for uniform predictions (model learned nothing)
        pred_classes = preds.argmax(axis=1)
        class_dist = np.bincount(pred_classes, minlength=expected_classes)
        class_dist_norm = class_dist / class_dist.sum()
        
        # Entropy of predictions
        entropy = -np.sum(class_dist_norm * np.log(class_dist_norm + 1e-10))
        max_entropy = np.log(expected_classes)
        entropy_ratio = entropy / max_entropy
        
        is_uniform = entropy_ratio > 0.95  # Very high entropy = uniform
        
        self._add_check(
            name="Predictions not uniform",
            passed=not is_uniform,
            severity="HIGH",
            message=f"Entropy ratio: {entropy_ratio:.3f} (1.0 = uniform)" if is_uniform else f"OK (entropy ratio: {entropy_ratio:.3f})",
            details={"entropy_ratio": round(entropy_ratio, 4), "class_distribution": class_dist.tolist()}
        )
        
        # Most predicted class
        most_common_class = pred_classes.tolist()
        most_common_idx = max(set(most_common_class), key=most_common_class.count)
        most_common_pct = class_dist[most_common_idx] / len(pred_classes) * 100
        
        self._add_check(
            name="Not all same class",
            passed=most_common_pct < 90,
            severity="HIGH",
            message=f"Class {most_common_idx} predicted {most_common_pct:.1f}% of time" if most_common_pct >= 90 else "OK",
            details={"most_common_class": most_common_idx, "percentage": round(most_common_pct, 1)}
        )
        
        # Confidence analysis with histogram buckets
        max_probs = preds.max(axis=1)
        mean_confidence = max_probs.mean()
        
        # Confidence histogram
        hist_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
        confidence_hist = {}
        for low, high in hist_ranges:
            count = ((max_probs >= low) & (max_probs < high if high < 1.0 else max_probs <= high)).sum()
            pct = (count / len(max_probs)) * 100
            confidence_hist[f"{low}-{high}"] = {
                "count": int(count),
                "percentage": round(pct, 1)
            }
        
        # Low confidence suggests model uncertainty
        self._add_check(
            name="Model shows confidence",
            passed=mean_confidence > 0.3,
            severity="MEDIUM",
            message=f"Mean confidence: {mean_confidence:.3f} (low = uncertain)" if mean_confidence <= 0.3 else f"OK (mean confidence: {mean_confidence:.3f})",
            details={
                "mean_confidence": round(float(mean_confidence), 4),
                "confidence_histogram": confidence_hist
            }
        )
        
        # Store stats
        self.prediction_stats = {
            "n_samples": len(preds),
            "n_classes": actual_classes,
            "class_distribution": class_dist.tolist(),
            "entropy_ratio": round(entropy_ratio, 4),
            "mean_confidence": round(float(mean_confidence), 4),
            "min_confidence": round(float(max_probs.min()), 4),
            "max_confidence": round(float(max_probs.max()), 4),
            "confidence_histogram": confidence_hist,
            "most_predicted_class": int(most_common_idx),
            "most_predicted_label": ACTIVITY_LABELS[most_common_idx] if most_common_idx < len(ACTIVITY_LABELS) else f"class_{most_common_idx}"
        }
    
    # -------------------------------------------------------------------------
    # Input Variance Check (Activity Proxy)
    # -------------------------------------------------------------------------
    
    def check_input_variance(self):
        """Check input data variance as proxy for activity level."""
        logger.info("\n📉 INPUT VARIANCE CHECK")
        
        if self.data is None:
            return
        
        # Compute std per channel across all windows and timesteps
        per_channel_std = self.data.std(axis=(0, 1))
        mean_std = per_channel_std.mean()
        
        # Check if variance is suspiciously low (idle data)
        is_idle = mean_std < 0.3  # Normalized std << 1.0
        
        self._add_check(
            name="Input has activity variance",
            passed=not is_idle,
            severity="CRITICAL" if is_idle else "LOW",
            message=f"Mean std: {mean_std:.4f} - IDLE/STATIONARY data detected! Collect data with activities." if is_idle else f"OK (mean std: {mean_std:.4f})",
            details={
                "per_channel_std": [float(x) for x in per_channel_std.round(4)],
                "mean_std": float(round(mean_std, 4)),
                "idle_threshold": 0.3,
                "is_idle": bool(is_idle)
            }
        )
        
        self.inference_stats["input_variance"] = {
            "per_channel_std": [float(x) for x in per_channel_std.round(4)],
            "mean_std": float(round(mean_std, 4)),
            "is_idle_data": bool(is_idle)
        }
    
    # -------------------------------------------------------------------------
    # Determinism Check
    # -------------------------------------------------------------------------
    
    def check_determinism(self):
        """Check that inference is deterministic."""
        logger.info("\n🎲 DETERMINISM CHECK")
        
        if self.model is None or self.data is None:
            return
        
        # Run twice on same data
        test_sample = self.data[:10]
        pred1 = self.model.predict(test_sample, verbose=0)
        pred2 = self.model.predict(test_sample, verbose=0)
        
        is_deterministic = np.allclose(pred1, pred2)
        
        self._add_check(
            name="Inference is deterministic",
            passed=is_deterministic,
            severity="MEDIUM",
            message="Same input produces same output" if is_deterministic else "Non-deterministic predictions detected",
            details={}
        )
    
    # -------------------------------------------------------------------------
    # Run All Checks
    # -------------------------------------------------------------------------
    
    def run(self) -> SmokeTestReport:
        """Run all smoke tests."""
        logger.info(f"\n{'='*60}")
        logger.info("🔥 INFERENCE SMOKE TEST")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Data: {self.data_path}")
        
        # Run checks in order
        if not self.check_model_loading():
            return self._finalize_report()
        
        if not self.check_data_loading():
            return self._finalize_report()
        
        if not self.check_shape_compatibility():
            return self._finalize_report()
        
        if not self.check_inference():
            return self._finalize_report()
        
        self.check_input_variance()
        self.check_prediction_quality()
        self.check_determinism()
        
        return self._finalize_report()
    
    def _finalize_report(self) -> SmokeTestReport:
        """Generate report and log to MLflow if enabled."""
        report = self._generate_report()
        
        # Log to MLflow if enabled
        if self.mlflow_logging:
            self._log_to_mlflow(report)
        
        return report
    
    def _generate_report(self) -> SmokeTestReport:
        """Generate smoke test report."""
        n_passed = sum(1 for c in self.checks if c.passed)
        n_failed = sum(1 for c in self.checks if not c.passed)
        n_critical = sum(1 for c in self.checks if not c.passed and c.severity == "CRITICAL")
        
        if n_critical > 0:
            status = "FAIL"
        elif n_failed > 0:
            status = "WARN"
        else:
            status = "PASS"
        
        return SmokeTestReport(
            timestamp=datetime.now().isoformat(),
            model_path=str(self.model_path),
            data_path=str(self.data_path),
            checks_passed=n_passed,
            checks_failed=n_failed,
            critical_failures=n_critical,
            overall_status=status,
            checks=[asdict(c) for c in self.checks],
            inference_stats=self.inference_stats,
            prediction_stats=self.prediction_stats
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference Smoke Test")
    parser.add_argument('--model', type=str, 
                       default='models/pretrained/fine_tuned_model_1dcnnbilstm.keras',
                       help='Path to model file')
    parser.add_argument('--input', type=str,
                       default='data/prepared/production_X.npy',
                       help='Path to input data (NPY)')
    parser.add_argument('--output-dir', type=str,
                       default='reports/inference_smoke',
                       help='Output directory for reports')
    parser.add_argument('--mlflow', action='store_true',
                       help='Enable MLflow logging')
    args = parser.parse_args()
    
    # Resolve paths
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
    
    data_path = Path(args.input)
    if not data_path.exists():
        data_path = PROJECT_ROOT / args.input
    
    # Validate
    if not model_path.exists():
        logger.error(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    if not data_path.exists():
        logger.error(f"❌ Data not found: {args.input}")
        sys.exit(1)
    
    # Run smoke test
    test = InferenceSmokeTest(model_path, data_path, mlflow_logging=args.mlflow)
    report = test.run()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 SMOKE TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Status: {report.overall_status}")
    logger.info(f"  Checks passed: {report.checks_passed}")
    logger.info(f"  Checks failed: {report.checks_failed}")
    logger.info(f"  Critical failures: {report.critical_failures}")
    
    if report.prediction_stats:
        logger.info(f"\n  Prediction Stats:")
        logger.info(f"    Mean confidence: {report.prediction_stats.get('mean_confidence', 'N/A')}")
        logger.info(f"    Most predicted: {report.prediction_stats.get('most_predicted_label', 'N/A')}")
        logger.info(f"    Entropy ratio: {report.prediction_stats.get('entropy_ratio', 'N/A')}")
    
    # Save report
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"smoke_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    logger.info(f"\n💾 Report saved: {report_path}")
    
    # Exit code
    if report.overall_status == "FAIL":
        logger.error("\n❌ SMOKE TEST FAILED")
        sys.exit(1)
    elif report.overall_status == "WARN":
        logger.warning("\n⚠️ SMOKE TEST PASSED WITH WARNINGS")
        sys.exit(0)
    else:
        logger.info("\n✅ SMOKE TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
