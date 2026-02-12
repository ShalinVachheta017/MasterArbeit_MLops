#!/usr/bin/env python
"""
Post-Inference Monitoring Script
=================================

Comprehensive monitoring for unlabeled production data inference.
Implements the 3-layer monitoring framework:

Layer 1: Per-window confidence/uncertainty metrics
Layer 2: Sequence temporal plausibility metrics  
Layer 3: Batch-level drift detection vs training baseline

Usage:
    python scripts/post_inference_monitoring.py --predictions data/prepared/predictions/latest.csv
    python scripts/post_inference_monitoring.py --predictions data/prepared/predictions/latest.csv --baseline data/prepared/baseline_stats.json
    python scripts/post_inference_monitoring.py --predictions data/prepared/predictions/latest.csv --mlflow

Outputs:
    - reports/monitoring/<timestamp>/confidence_report.json
    - reports/monitoring/<timestamp>/temporal_report.json
    - reports/monitoring/<timestamp>/drift_report.json
    - reports/monitoring/<timestamp>/summary.json
    - MLflow metrics and artifacts (if --mlflow flag)

Author: Master Thesis MLOps Project
Date: January 15, 2026
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PROJECT_ROOT, DATA_PREPARED, OUTPUTS_DIR, LOGS_DIR,
    ACTIVITY_LABELS, NUM_CLASSES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """
    Configuration for post-inference monitoring.
    
    PSI Interpretation (standard thresholds):
        PSI < 0.10  → No/low shift (PASS)
        0.10-0.25   → Moderate shift (WARN, investigate)
        PSI > 0.25  → Major shift (likely drift, action needed)
    
    Note on Multiple Comparisons:
        Testing 6 channels independently inflates false positives.
        Options: (1) Bonferroni correction, (2) k-channel gating (default).
        We use k-channel gating: drift only flagged if ≥ max_drift_channels channels drift.
    
    Note on Dominance Mismatch:
        When watch is worn on non-dominant wrist, signal observability is reduced.
        Use relaxed thresholds via get_effective_thresholds(dominance_match=False).
    """
    
    # Reproducibility
    random_seed: int = 42  # Fixed seed for sampling reproducibility
    
    # Layer 1: Confidence thresholds (normal operation)
    confidence_threshold: float = 0.50  # Below this = uncertain
    entropy_threshold: float = 2.0       # Above this = high uncertainty
    margin_threshold: float = 0.10       # Below this = ambiguous
    
    # Layer 1: Relaxed thresholds for dominance mismatch
    # When watch is on non-dominant wrist, expect lower confidence
    mismatch_confidence_threshold: float = 0.35   # Relaxed from 0.50
    mismatch_entropy_threshold: float = 2.5       # Relaxed from 2.0
    mismatch_margin_threshold: float = 0.08       # Relaxed from 0.10
    
    # Layer 2: Temporal thresholds (normal operation)
    max_flip_rate: float = 0.30          # Above this = unstable
    min_dwell_time_seconds: float = 2.0  # Below this = too short
    window_duration_seconds: float = 4.0 # 200 samples at 50Hz (full window)
    window_overlap: float = 0.5          # 50% overlap between windows
    sampling_rate_hz: float = 50.0       # Sampling rate
    
    # Layer 2: Relaxed thresholds for dominance mismatch
    mismatch_max_flip_rate: float = 0.45  # Relaxed from 0.30
    
    # Computed: stride = window * (1 - overlap) = 4.0 * 0.5 = 2.0s
    # With 50% overlap, each new window advances by 2 seconds, not 4 seconds!
    @property
    def window_stride_seconds(self) -> float:
        """Time advance between consecutive windows (accounts for overlap)."""
        return self.window_duration_seconds * (1.0 - self.window_overlap)
    
    # Layer 3: Drift thresholds
    # Note: For 6 channels with Bonferroni, use ks_pvalue_threshold/6 = 0.00167
    ks_pvalue_threshold: float = 0.01    # Base p-value (before Bonferroni)
    use_bonferroni: bool = False         # If True, apply Bonferroni correction
    n_channels: int = 6                  # Number of channels for Bonferroni
    wasserstein_threshold: float = 0.5   # Effect size (preferred over p-value)
    mean_shift_threshold: float = 0.5    # Normalized mean shift
    variance_collapse_threshold: float = 0.1  # Below this = idle/failure
    
    # PSI thresholds (standard interpretation)
    # PSI < 0.10: no shift, 0.10-0.25: moderate, > 0.25: major shift
    psi_threshold: float = 0.25          # PSI > 0.25 indicates significant drift
    psi_warn_threshold: float = 0.10     # PSI > 0.10 triggers warning
    
    # Gating thresholds (k-channel approach for multiple comparisons)
    max_uncertain_ratio: float = 0.30    # Above this = WARN
    max_drift_channels: int = 2          # Above this = BLOCK
    
    @property
    def effective_ks_pvalue_threshold(self) -> float:
        """KS p-value threshold with optional Bonferroni correction."""
        if self.use_bonferroni:
            return self.ks_pvalue_threshold / self.n_channels
        return self.ks_pvalue_threshold
    
    def get_effective_thresholds(self, dominance_match: bool = True) -> Dict[str, float]:
        """
        Return adjusted thresholds based on dominance match status.
        
        When watch is on non-dominant wrist while user performs activities
        with dominant hand, expect lower confidence and higher uncertainty.
        
        Args:
            dominance_match: True if watch is on dominant wrist (normal),
                           False if watch is on non-dominant wrist (relaxed thresholds)
        
        Returns:
            Dictionary of effective thresholds for this session
        """
        if dominance_match:
            return {
                "confidence_threshold": self.confidence_threshold,
                "entropy_threshold": self.entropy_threshold,
                "margin_threshold": self.margin_threshold,
                "max_flip_rate": self.max_flip_rate,
                "mode": "normal",
            }
        else:
            return {
                "confidence_threshold": self.mismatch_confidence_threshold,
                "entropy_threshold": self.mismatch_entropy_threshold,
                "margin_threshold": self.mismatch_margin_threshold,
                "max_flip_rate": self.mismatch_max_flip_rate,
                "mode": "low_observability",
            }


# ============================================================================
# LOW OBSERVABILITY DETECTION
# ============================================================================

def detect_low_observability_pattern(
    confidence_mean: float,
    entropy_mean: float,
    flip_rate: float,
    idle_percentage: float,
    motion_energy: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Detect if session shows low-observability pattern consistent with
    non-dominant wrist wearing (watch on opposite wrist from activity hand).
    
    Pattern characteristics:
    - Low motion energy (weak signal from opposite hand)
    - High uncertainty (model struggles with ambiguous input)
    - High flip rate (unstable predictions from weak signal)
    - High idle percentage (defaults to sitting/standing when signal unclear)
    
    Args:
        confidence_mean: Mean prediction confidence across all windows
        entropy_mean: Mean entropy across all windows
        flip_rate: Fraction of windows where prediction changes from previous
        idle_percentage: Fraction of predictions that are sitting/standing
        motion_energy: Optional normalized motion energy (std of accelerometer)
    
    Returns:
        Tuple of (is_low_observability, score, explanation)
        - is_low_observability: True if pattern detected (score >= 0.5)
        - score: 0.0-1.0 indicating strength of low-observability pattern
        - explanation: Human-readable reasons
    """
    score = 0.0
    reasons = []
    
    # Low motion energy (weak signal from opposite hand)
    if motion_energy is not None and motion_energy < 0.5:
        score += 0.25
        reasons.append(f"Low motion energy ({motion_energy:.2f})")
    
    # High uncertainty (model struggles)
    if confidence_mean < 0.65:
        score += 0.25
        reasons.append(f"Low mean confidence ({confidence_mean:.1%})")
    
    # High entropy (probability spread)
    if entropy_mean > 1.2:
        score += 0.20
        reasons.append(f"High entropy ({entropy_mean:.2f})")
    
    # Unstable predictions
    if flip_rate > 0.25:
        score += 0.15
        reasons.append(f"High flip rate ({flip_rate:.1%})")
    
    # Dominated by idle predictions
    if idle_percentage > 0.40:
        score += 0.15
        reasons.append(f"High idle% ({idle_percentage:.1%})")
    
    is_low_obs = score >= 0.5
    explanation = "; ".join(reasons) if reasons else "Normal observability"
<<<<<<< HEAD
    
    return is_low_obs, score, explanation
=======
     
    return is_low_obs, score, explanation 
>>>>>>> 8632082 (Complete 10-stage MLOps pipeline with AdaBN domain adaptation)


# ============================================================================
# LAYER 1: CONFIDENCE/UNCERTAINTY METRICS
# ============================================================================

class ConfidenceAnalyzer:
    """
    Analyze per-window confidence and uncertainty.
    
    Metrics computed:
    - max_probability: Maximum softmax probability
    - entropy: Shannon entropy of probability distribution
    - margin: Difference between top-1 and top-2 probabilities
    - energy_score: Energy-based OOD score (if logits available)
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    def analyze(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze confidence from predictions DataFrame.
        
        Args:
            predictions_df: DataFrame with columns:
                - predicted_class or predicted_activity
                - confidence (max probability)
                - prob_0, prob_1, ..., prob_10 (optional, for entropy/margin)
        
        Returns:
            Dictionary with confidence analysis results
        """
        logger.info("=" * 60)
        logger.info("📊 LAYER 1: CONFIDENCE/UNCERTAINTY ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_windows": len(predictions_df),
            "metrics": {},
            "distributions": {},
            "flagged_windows": [],
            "summary": {}
        }
        
        # Get confidence values
        if 'confidence' in predictions_df.columns:
            confidence = predictions_df['confidence'].values
        else:
            # Try to compute from probability columns
            prob_cols = [c for c in predictions_df.columns if c.startswith('prob_')]
            if prob_cols:
                probs = predictions_df[prob_cols].values
                confidence = probs.max(axis=1)
            else:
                raise ValueError("No confidence or probability columns found")
        
        # Basic confidence statistics
        results["metrics"]["mean_confidence"] = float(np.mean(confidence))
        results["metrics"]["std_confidence"] = float(np.std(confidence))
        results["metrics"]["min_confidence"] = float(np.min(confidence))
        results["metrics"]["max_confidence"] = float(np.max(confidence))
        results["metrics"]["median_confidence"] = float(np.median(confidence))
        
        logger.info(f"  Mean confidence: {results['metrics']['mean_confidence']:.3f}")
        logger.info(f"  Std confidence:  {results['metrics']['std_confidence']:.3f}")
        logger.info(f"  Min confidence:  {results['metrics']['min_confidence']:.3f}")
        logger.info(f"  Max confidence:  {results['metrics']['max_confidence']:.3f}")
        
        # Confidence level distribution
        n_high = np.sum(confidence >= 0.90)
        n_moderate = np.sum((confidence >= 0.70) & (confidence < 0.90))
        n_low = np.sum((confidence >= 0.50) & (confidence < 0.70))
        n_uncertain = np.sum(confidence < 0.50)
        
        results["distributions"]["confidence_levels"] = {
            "high_90plus": {"count": int(n_high), "ratio": float(n_high / len(confidence))},
            "moderate_70_90": {"count": int(n_moderate), "ratio": float(n_moderate / len(confidence))},
            "low_50_70": {"count": int(n_low), "ratio": float(n_low / len(confidence))},
            "uncertain_below_50": {"count": int(n_uncertain), "ratio": float(n_uncertain / len(confidence))}
        }
        
        uncertain_ratio = n_uncertain / len(confidence)
        results["metrics"]["uncertain_ratio"] = float(uncertain_ratio)
        
        logger.info(f"\n  Confidence distribution:")
        logger.info(f"    HIGH (≥90%):      {n_high:5d} ({100*n_high/len(confidence):.1f}%)")
        logger.info(f"    MODERATE (70-90%): {n_moderate:5d} ({100*n_moderate/len(confidence):.1f}%)")
        logger.info(f"    LOW (50-70%):     {n_low:5d} ({100*n_low/len(confidence):.1f}%)")
        logger.info(f"    UNCERTAIN (<50%): {n_uncertain:5d} ({100*n_uncertain/len(confidence):.1f}%)")
        
        # Compute entropy and margin if probability columns available
        prob_cols = [c for c in predictions_df.columns if c.startswith('prob_')]
        if prob_cols:
            probs = predictions_df[prob_cols].values
            
            # Entropy: H = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            results["metrics"]["mean_entropy"] = float(np.mean(entropy))
            results["metrics"]["max_entropy"] = float(np.max(entropy))
            
            # Margin: top1 - top2
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # Descending
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            results["metrics"]["mean_margin"] = float(np.mean(margin))
            results["metrics"]["min_margin"] = float(np.min(margin))
            
            # Count high entropy and low margin
            n_high_entropy = np.sum(entropy > self.config.entropy_threshold)
            n_low_margin = np.sum(margin < self.config.margin_threshold)
            
            results["metrics"]["high_entropy_count"] = int(n_high_entropy)
            results["metrics"]["low_margin_count"] = int(n_low_margin)
            
            logger.info(f"\n  Entropy analysis:")
            logger.info(f"    Mean entropy: {results['metrics']['mean_entropy']:.3f}")
            logger.info(f"    High entropy (>{self.config.entropy_threshold}): {n_high_entropy}")
            logger.info(f"\n  Margin analysis:")
            logger.info(f"    Mean margin: {results['metrics']['mean_margin']:.3f}")
            logger.info(f"    Low margin (<{self.config.margin_threshold}): {n_low_margin}")
        
        # Flag uncertain windows
        uncertain_mask = confidence < self.config.confidence_threshold
        uncertain_indices = np.where(uncertain_mask)[0]
        
        for idx in uncertain_indices[:100]:  # Limit to first 100
            window_info = {
                "window_id": int(idx),
                "confidence": float(confidence[idx]),
                "predicted_class": predictions_df.iloc[idx].get('predicted_class', 
                                    predictions_df.iloc[idx].get('predicted_activity', 'unknown'))
            }
            if prob_cols:
                window_info["entropy"] = float(entropy[idx])
                window_info["margin"] = float(margin[idx])
                # Top 2 classes
                top2_idx = np.argsort(probs[idx])[::-1][:2]
                window_info["top2_classes"] = [int(i) for i in top2_idx]
                window_info["top2_probs"] = [float(probs[idx][i]) for i in top2_idx]
            
            results["flagged_windows"].append(window_info)
        
        # Summary status
        if uncertain_ratio > self.config.max_uncertain_ratio:
            status = "WARN"
            message = f"High uncertainty: {100*uncertain_ratio:.1f}% windows below {self.config.confidence_threshold} confidence"
        elif uncertain_ratio > 0.1:
            status = "INFO"
            message = f"Some uncertainty: {100*uncertain_ratio:.1f}% windows uncertain"
        else:
            status = "PASS"
            message = f"Good confidence: only {100*uncertain_ratio:.1f}% windows uncertain"
        
        results["summary"]["status"] = status
        results["summary"]["message"] = message
        results["summary"]["uncertain_ratio"] = float(uncertain_ratio)
        results["summary"]["threshold"] = self.config.confidence_threshold
        
        logger.info(f"\n  Status: {status} - {message}")
        
        return results


# ============================================================================
# LAYER 2: TEMPORAL PLAUSIBILITY METRICS
# ============================================================================

class TemporalAnalyzer:
    """
    Analyze temporal plausibility of prediction sequences.
    
    Metrics computed:
    - flip_rate: Ratio of class changes between consecutive windows
    - dwell_times: Duration of each continuous activity bout
    - transition_matrix: Counts of transitions between classes
    - stability_score: Overall temporal stability metric
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Define impossible/unlikely transitions (domain knowledge)
        # These are transitions that shouldn't happen in < 1 window
        self.unlikely_transitions = [
            # Rapid oscillations between opposite states
            ("sitting", "standing", "sitting"),
            ("standing", "sitting", "standing"),
        ]
    
    def analyze(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in prediction sequence.
        
        Args:
            predictions_df: DataFrame with predictions in sequential order
        
        Returns:
            Dictionary with temporal analysis results
        """
        logger.info("=" * 60)
        logger.info("⏱️  LAYER 2: TEMPORAL PLAUSIBILITY ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_windows": len(predictions_df),
            "metrics": {},
            "bouts": [],
            "transition_matrix": {},
            "warnings": [],
            "summary": {}
        }
        
        # Get prediction sequence
        if 'predicted_class' in predictions_df.columns:
            classes = predictions_df['predicted_class'].values
        elif 'predicted_activity' in predictions_df.columns:
            classes = predictions_df['predicted_activity'].values
        else:
            raise ValueError("No predicted_class or predicted_activity column found")
        
        n_windows = len(classes)
        
        # 1. Flip rate
        if n_windows > 1:
            transitions = np.sum(classes[1:] != classes[:-1])
            flip_rate = transitions / (n_windows - 1)
        else:
            transitions = 0
            flip_rate = 0.0
        
        results["metrics"]["n_transitions"] = int(transitions)
        results["metrics"]["flip_rate"] = float(flip_rate)
        
        logger.info(f"  Total windows: {n_windows}")
        logger.info(f"  Transitions: {transitions}")
        logger.info(f"  Flip rate: {100*flip_rate:.1f}%")
        
        if flip_rate > self.config.max_flip_rate:
            results["warnings"].append({
                "type": "high_flip_rate",
                "message": f"Flip rate {100*flip_rate:.1f}% exceeds threshold {100*self.config.max_flip_rate:.1f}%",
                "severity": "MEDIUM"
            })
            logger.warning(f"  ⚠️ High flip rate detected!")
        
        # 2. Activity bouts (continuous segments of same class)
        # IMPORTANT: Use stride, not window duration!
        # With 50% overlap, stride = 2.0s, not 4.0s
        stride_seconds = self.config.window_stride_seconds
        
        bouts = []
        current_class = classes[0]
        bout_start = 0
        
        for i in range(1, n_windows):
            if classes[i] != current_class:
                # Duration = number of windows * stride + (window_duration - stride) for last window
                # Simplified: n_windows * stride gives the START-to-START time
                # Add (window_duration - stride) to account for last window's full coverage
                n_bout_windows = i - bout_start
                bout_duration = n_bout_windows * stride_seconds
                bouts.append({
                    "class": str(current_class),
                    "start_window": int(bout_start),
                    "end_window": int(i - 1),
                    "n_windows": int(n_bout_windows),
                    "duration_seconds": float(bout_duration)
                })
                current_class = classes[i]
                bout_start = i
        
        # Final bout
        n_bout_windows = n_windows - bout_start
        bout_duration = n_bout_windows * stride_seconds
        bouts.append({
            "class": str(current_class),
            "start_window": int(bout_start),
            "end_window": int(n_windows - 1),
            "n_windows": int(n_bout_windows),
            "duration_seconds": float(bout_duration)
        })
        
        results["bouts"] = bouts
        results["metrics"]["n_bouts"] = len(bouts)
        
        # Dwell time statistics
        dwell_times = [b["duration_seconds"] for b in bouts]
        results["metrics"]["mean_dwell_time"] = float(np.mean(dwell_times))
        results["metrics"]["min_dwell_time"] = float(np.min(dwell_times))
        results["metrics"]["max_dwell_time"] = float(np.max(dwell_times))
        
        logger.info(f"\n  Activity bouts: {len(bouts)}")
        logger.info(f"  Mean dwell time: {results['metrics']['mean_dwell_time']:.1f}s")
        logger.info(f"  Min dwell time: {results['metrics']['min_dwell_time']:.1f}s")
        logger.info(f"  Max dwell time: {results['metrics']['max_dwell_time']:.1f}s")
        
        # Check for unrealistically short bouts
        short_bouts = [b for b in bouts if b["duration_seconds"] < self.config.min_dwell_time_seconds]
        if short_bouts:
            results["metrics"]["n_short_bouts"] = len(short_bouts)
            results["warnings"].append({
                "type": "short_bouts",
                "message": f"{len(short_bouts)} bouts shorter than {self.config.min_dwell_time_seconds}s",
                "severity": "LOW",
                "details": short_bouts[:10]  # First 10
            })
            logger.warning(f"  ⚠️ {len(short_bouts)} unrealistically short bouts detected")
        
        # 3. Transition matrix
        unique_classes = sorted(set(classes))
        n_classes = len(unique_classes)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        
        trans_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(classes) - 1):
            from_idx = class_to_idx[classes[i]]
            to_idx = class_to_idx[classes[i + 1]]
            trans_matrix[from_idx, to_idx] += 1
        
        # Store as nested dict
        trans_dict = {}
        for i, from_class in enumerate(unique_classes):
            trans_dict[str(from_class)] = {}
            for j, to_class in enumerate(unique_classes):
                if trans_matrix[i, j] > 0:
                    trans_dict[str(from_class)][str(to_class)] = int(trans_matrix[i, j])
        
        results["transition_matrix"] = trans_dict
        
        # Activity distribution
        class_counts = {}
        for c in classes:
            class_counts[str(c)] = class_counts.get(str(c), 0) + 1
        
        results["activity_distribution"] = class_counts
        
        logger.info(f"\n  Activity distribution:")
        for c, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {c}: {count} windows ({100*count/n_windows:.1f}%)")
        
        # 4. Summary status
        warning_count = len(results["warnings"])
        if flip_rate > self.config.max_flip_rate:
            status = "WARN"
            message = f"High instability: {100*flip_rate:.1f}% flip rate"
        elif warning_count > 0:
            status = "INFO"
            message = f"{warning_count} minor temporal warnings"
        else:
            status = "PASS"
            message = "Temporally plausible predictions"
        
        results["summary"]["status"] = status
        results["summary"]["message"] = message
        results["summary"]["flip_rate"] = float(flip_rate)
        results["summary"]["n_warnings"] = warning_count
        
        logger.info(f"\n  Status: {status} - {message}")
        
        return results


# ============================================================================
# LAYER 3: DRIFT DETECTION
# ============================================================================

class DriftDetector:
    """
    Detect distribution drift between production data and training baseline.
    
    Methods:
    - KS test (Kolmogorov-Smirnov): Non-parametric test for distribution difference
    - Wasserstein distance: Earth mover's distance between distributions
    - Mean/std shift: Simple statistics comparison
    - Variance collapse detection: Identify idle or sensor failure
    """
    
    SENSOR_NAMES = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    def load_baseline(self, baseline_path: Path) -> Dict[str, Any]:
        """Load training baseline statistics."""
        with open(baseline_path, 'r') as f:
            return json.load(f)
    
    def analyze(self, 
                production_data: np.ndarray,
                baseline: Optional[Dict[str, Any]] = None,
                baseline_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Analyze drift between production data and training baseline.
        
        Args:
            production_data: NumPy array of shape (n_windows, timesteps, channels)
            baseline: Baseline statistics dict (or load from baseline_path)
            baseline_path: Path to baseline_stats.json
        
        Returns:
            Dictionary with drift analysis results
        """
        logger.info("=" * 60)
        logger.info("📈 LAYER 3: DRIFT DETECTION ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "production_shape": list(production_data.shape),
            "metrics": {},
            "per_channel": {},
            "warnings": [],
            "summary": {}
        }
        
        # Load baseline if needed
        if baseline is None and baseline_path is not None:
            baseline = self.load_baseline(baseline_path)
        
        # Set RNG seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Compute production statistics
        n_windows, n_timesteps, n_channels = production_data.shape
        
        # Flatten to (n_samples, n_channels) for per-channel analysis
        prod_flat = production_data.reshape(-1, n_channels)
        
        logger.info(f"  Production data: {n_windows} windows × {n_timesteps} timesteps × {n_channels} channels")
        
        # Per-channel analysis
        drift_flags = []
        
        for ch_idx, ch_name in enumerate(self.SENSOR_NAMES[:n_channels]):
            ch_data = prod_flat[:, ch_idx]
            ch_results = {
                "mean": float(np.mean(ch_data)),
                "std": float(np.std(ch_data)),
                "min": float(np.min(ch_data)),
                "max": float(np.max(ch_data)),
                "percentile_5": float(np.percentile(ch_data, 5)),
                "percentile_95": float(np.percentile(ch_data, 95))
            }
            
            # Variance collapse check
            if ch_results["std"] < self.config.variance_collapse_threshold:
                ch_results["variance_collapse"] = True
                drift_flags.append(ch_name)
                results["warnings"].append({
                    "type": "variance_collapse",
                    "channel": ch_name,
                    "message": f"Very low variance ({ch_results['std']:.4f}) - possible idle data or sensor failure",
                    "severity": "HIGH"
                })
                logger.warning(f"  ⚠️ {ch_name}: Variance collapse detected (std={ch_results['std']:.4f})")
            else:
                ch_results["variance_collapse"] = False
            
            # Compare to baseline if available
            if baseline and "per_channel" in baseline:
                baseline_ch = baseline["per_channel"].get(ch_name, {})
                
                if baseline_ch:
                    # Mean shift
                    baseline_mean = baseline_ch.get("mean", 0)
                    baseline_std = baseline_ch.get("std", 1)
                    mean_shift = abs(ch_results["mean"] - baseline_mean)
                    normalized_shift = mean_shift / (baseline_std + 1e-6)
                    
                    ch_results["baseline_mean"] = baseline_mean
                    ch_results["baseline_std"] = baseline_std
                    ch_results["mean_shift"] = float(mean_shift)
                    ch_results["normalized_mean_shift"] = float(normalized_shift)
                    
                    # Sample from production for comparison
                    sample_size = min(10000, len(ch_data))
                    prod_sample = np.random.choice(ch_data, sample_size, replace=False)
                    
                    # Check if real baseline samples are available (preferred method)
                    baseline_samples = baseline_ch.get("samples", None)
                    
                    if baseline_samples is not None:
                        # Use REAL stored samples - scientifically defensible!
                        baseline_sample = np.array(baseline_samples)
                        # Subsample if needed
                        if len(baseline_sample) > sample_size:
                            baseline_sample = np.random.choice(baseline_sample, sample_size, replace=False)
                        
                        ks_stat, ks_pvalue = stats.ks_2samp(prod_sample, baseline_sample)
                        wasserstein = stats.wasserstein_distance(prod_sample, baseline_sample)
                        ch_results["comparison_method"] = "real_samples"
                    else:
                        # Fallback: Use histogram-based PSI (Population Stability Index)
                        # This is more robust than synthetic Normal approximation
                        baseline_hist = baseline_ch.get("histogram", None)
                        
                        if baseline_hist is not None:
                            psi = self._compute_psi(
                                prod_sample, 
                                np.array(baseline_hist["bin_edges"]),
                                np.array(baseline_hist["counts"])
                            )
                            ch_results["psi"] = float(psi)
                            ch_results["comparison_method"] = "histogram_psi"
                            
                            # For KS/Wasserstein, use histogram-reconstructed samples
                            baseline_sample = self._sample_from_histogram(
                                np.array(baseline_hist["bin_edges"]),
                                np.array(baseline_hist["counts"]),
                                sample_size
                            )
                        else:
                            # Last resort: Normal approximation (document the limitation)
                            logger.warning(f"  ⚠️ {ch_name}: Using Normal approximation (no samples/histogram)")
                            baseline_sample = np.random.normal(baseline_mean, baseline_std, sample_size)
                            ch_results["comparison_method"] = "normal_approximation"
                        
                        ks_stat, ks_pvalue = stats.ks_2samp(prod_sample, baseline_sample)
                        wasserstein = stats.wasserstein_distance(prod_sample, baseline_sample)
                    
                    ch_results["ks_statistic"] = float(ks_stat)
                    ch_results["ks_pvalue"] = float(ks_pvalue)
                    ch_results["wasserstein_distance"] = float(wasserstein)
                    
                    # Check drift thresholds
                    # Use PSI if available, otherwise KS/Wasserstein
                    # Note: We prefer effect sizes (Wasserstein, PSI) over p-values
                    # because KS p-values become tiny at scale even for minor changes
                    psi_value = ch_results.get("psi", 0)
                    psi_drift = psi_value > self.config.psi_threshold
                    psi_warn = psi_value > self.config.psi_warn_threshold
                    
                    # Use Bonferroni-corrected threshold if enabled
                    effective_p_threshold = self.config.effective_ks_pvalue_threshold
                    stat_drift = (
                        ks_pvalue < effective_p_threshold or
                        wasserstein > self.config.wasserstein_threshold or
                        normalized_shift > self.config.mean_shift_threshold
                    )
                    
                    drift_detected = psi_drift or stat_drift
                    ch_results["drift_detected"] = bool(drift_detected)
                    ch_results["psi_level"] = "major" if psi_drift else ("moderate" if psi_warn else "low")
                    
                    if drift_detected:
                        drift_flags.append(ch_name)
                        psi_str = f", PSI={ch_results.get('psi', 'N/A'):.3f}" if 'psi' in ch_results else ""
                        results["warnings"].append({
                            "type": "distribution_drift",
                            "channel": ch_name,
                            "message": f"Drift detected: KS p={ks_pvalue:.4f}, Wasserstein={wasserstein:.3f}{psi_str}",
                            "severity": "MEDIUM",
                            "method": ch_results.get("comparison_method", "unknown")
                        })
                        logger.warning(f"  ⚠️ {ch_name}: Drift detected (KS p={ks_pvalue:.4f})")
                    else:
                        logger.info(f"  ✓ {ch_name}: No significant drift (KS p={ks_pvalue:.4f})")
            
            results["per_channel"][ch_name] = ch_results
        
        # Aggregate metrics
        results["metrics"]["n_channels_with_drift"] = len(drift_flags)
        results["metrics"]["drift_channels"] = drift_flags
        
        # Gravity check - only valid for raw data (not normalized)
        # If data is normalized (mean≈0, std≈1), skip this check
        if n_channels >= 3:
            az_mean = results["per_channel"]["Az"]["mean"]
            az_std = results["per_channel"]["Az"]["std"]
            
            # Detect if data appears normalized (std near 1, mean near 0)
            is_normalized = (abs(az_std - 1.0) < 1.5 and abs(az_mean) < 5.0)
            
            if is_normalized:
                # For normalized data, check if mean is within reasonable range
                results["metrics"]["gravity_check"] = {
                    "az_mean": float(az_mean),
                    "data_type": "normalized",
                    "note": "Gravity check skipped for normalized data",
                    "within_tolerance": True  # Skip this check for normalized data
                }
            else:
                # For raw data in m/s², check gravity
                gravity_expected = -9.8
                gravity_tolerance = 3.0  # Allow some variation
                
                gravity_ok = abs(az_mean - gravity_expected) < gravity_tolerance
                results["metrics"]["gravity_check"] = {
                    "az_mean": float(az_mean),
                    "data_type": "raw_m/s²",
                    "expected": gravity_expected,
                    "within_tolerance": bool(gravity_ok)
                }
                
                if not gravity_ok:
                    results["warnings"].append({
                        "type": "gravity_anomaly",
                        "message": f"Az mean ({az_mean:.2f}) differs from expected gravity ({gravity_expected})",
                        "severity": "HIGH"
                    })
                    logger.warning(f"  ⚠️ Gravity check failed: Az mean = {az_mean:.2f}")
        
        # Summary status
        n_drift_channels = len(drift_flags)
        variance_collapse = any(w["type"] == "variance_collapse" for w in results["warnings"])
        
        if variance_collapse:
            status = "BLOCK"
            message = "Variance collapse detected - sensor failure or idle data"
        elif n_drift_channels > self.config.max_drift_channels:
            status = "BLOCK"
            message = f"Significant drift in {n_drift_channels} channels"
        elif n_drift_channels > 0:
            status = "WARN"
            message = f"Minor drift in {n_drift_channels} channel(s): {drift_flags}"
        else:
            status = "PASS"
            message = "No significant drift detected"
        
        results["summary"]["status"] = status
        results["summary"]["message"] = message
        results["summary"]["n_drift_channels"] = n_drift_channels
        results["summary"]["drift_channels"] = drift_flags
        
        logger.info(f"\n  Status: {status} - {message}")
        
        return results
    
    def _compute_psi(self, 
                     production_sample: np.ndarray,
                     baseline_bin_edges: np.ndarray,
                     baseline_counts: np.ndarray) -> float:
        """
        Compute Population Stability Index (PSI) between production and baseline.
        
        PSI is a measure of how much a distribution has shifted. It's widely used
        in credit scoring and model monitoring because:
        - It works with histograms (no need to store raw samples)
        - It's symmetric and interpretable
        - PSI < 0.10: No significant change
        - PSI 0.10-0.25: Moderate change, investigate
        - PSI > 0.25: Significant change, likely need action
        
        Args:
            production_sample: Array of production values
            baseline_bin_edges: Bin edges from baseline histogram
            baseline_counts: Bin counts from baseline histogram
        
        Returns:
            PSI value (float)
        """
        # Compute production histogram using same bins
        prod_counts, _ = np.histogram(production_sample, bins=baseline_bin_edges)
        
        # Normalize to proportions
        baseline_prop = baseline_counts / (baseline_counts.sum() + 1e-10)
        prod_prop = prod_counts / (prod_counts.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_prop = np.clip(baseline_prop, epsilon, 1.0)
        prod_prop = np.clip(prod_prop, epsilon, 1.0)
        
        # PSI = sum( (prod - baseline) * ln(prod / baseline) )
        psi = np.sum((prod_prop - baseline_prop) * np.log(prod_prop / baseline_prop))
        
        return float(psi)
    
    def _sample_from_histogram(self,
                               bin_edges: np.ndarray,
                               counts: np.ndarray,
                               n_samples: int) -> np.ndarray:
        """
        Generate samples that approximate the distribution represented by a histogram.
        
        This allows using stored histograms for KS/Wasserstein tests when raw
        samples aren't available.
        
        Args:
            bin_edges: Histogram bin edges
            counts: Histogram bin counts
            n_samples: Number of samples to generate
        
        Returns:
            Array of samples approximating the histogram distribution
        """
        # Normalize counts to probabilities
        probs = counts / (counts.sum() + 1e-10)
        
        # Sample bin indices according to probabilities
        bin_indices = np.random.choice(len(counts), size=n_samples, p=probs)
        
        # Generate uniform samples within each selected bin
        bin_widths = np.diff(bin_edges)
        samples = bin_edges[bin_indices] + np.random.uniform(0, 1, n_samples) * bin_widths[bin_indices]
        
        return samples


# ============================================================================
# LAYER 4 (OPTIONAL): EMBEDDING DRIFT DETECTION
# ============================================================================

class EmbeddingDriftDetector:
    """
    Detect drift in model embedding space (optional but strong for thesis).
    
    This is one of the strongest "label-free evaluation" arguments for deep HAR:
    - Embeddings capture learned representations
    - Drift in embedding space indicates input distribution shift
    - Works even when raw sensor statistics look similar
    
    Methods:
    - Mean/covariance shift in embedding space
    - Mahalanobis distance from baseline
    - Cosine similarity shift
    - Wasserstein distance in embedding space
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    def load_baseline(self, baseline_path: Path) -> Dict[str, Any]:
        """Load embedding baseline from NPZ file."""
        data = np.load(baseline_path)
        return {
            "mean": data["mean"],
            "std": data["std"],
            "n_samples": int(data["n_samples"]),
            "sample_embeddings": data.get("sample_embeddings", None)
        }
    
    def extract_embeddings(self, 
                          model_path: Path, 
                          X: np.ndarray,
                          layer_name: str = "bidirectional") -> np.ndarray:
        """Extract embeddings from model's intermediate layer."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for embedding extraction")
        
        model = tf.keras.models.load_model(model_path)
        
        # Find embedding layer
        embedding_layer = None
        for layer in model.layers:
            if layer_name.lower() in layer.name.lower():
                embedding_layer = layer
                break
        
        if embedding_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        # Create embedding extractor
        embedding_model = tf.keras.Model(
            inputs=model.input,
            outputs=embedding_layer.output
        )
        
        # Extract in batches
        batch_size = 256
        embeddings = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_emb = embedding_model.predict(batch, verbose=0)
            # Global average pooling if sequence output
            if len(batch_emb.shape) == 3:
                batch_emb = np.mean(batch_emb, axis=1)
            embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def analyze(self,
                production_embeddings: np.ndarray,
                baseline: Optional[Dict[str, Any]] = None,
                baseline_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Analyze embedding drift.
        
        Args:
            production_embeddings: (n_samples, embedding_dim)
            baseline: Baseline dict with mean/std/sample_embeddings
            baseline_path: Path to baseline_embeddings.npz
        
        Returns:
            Dictionary with embedding drift analysis
        """
        logger.info("=" * 60)
        logger.info("🧠 LAYER 4: EMBEDDING DRIFT ANALYSIS")
        logger.info("=" * 60)
        
        np.random.seed(self.config.random_seed)
        
        if baseline is None and baseline_path is not None:
            baseline = self.load_baseline(baseline_path)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "production_shape": list(production_embeddings.shape),
            "metrics": {},
            "warnings": [],
            "summary": {}
        }
        
        prod_mean = np.mean(production_embeddings, axis=0)
        prod_std = np.std(production_embeddings, axis=0)
        
        results["metrics"]["production_embedding_dim"] = production_embeddings.shape[1]
        
        if baseline:
            baseline_mean = baseline["mean"]
            baseline_std = baseline["std"]
            
            # 1. Mean shift (L2 distance)
            mean_shift = np.linalg.norm(prod_mean - baseline_mean)
            normalized_mean_shift = mean_shift / (np.linalg.norm(baseline_std) + 1e-6)
            
            results["metrics"]["embedding_mean_shift"] = float(mean_shift)
            results["metrics"]["embedding_normalized_mean_shift"] = float(normalized_mean_shift)
            
            # 2. Cosine similarity
            cosine_sim = np.dot(prod_mean, baseline_mean) / (
                np.linalg.norm(prod_mean) * np.linalg.norm(baseline_mean) + 1e-6
            )
            results["metrics"]["embedding_cosine_similarity"] = float(cosine_sim)
            
            # 3. Per-dimension drift
            dim_shifts = np.abs(prod_mean - baseline_mean) / (baseline_std + 1e-6)
            results["metrics"]["max_dim_shift"] = float(np.max(dim_shifts))
            results["metrics"]["mean_dim_shift"] = float(np.mean(dim_shifts))
            n_drifted_dims = np.sum(dim_shifts > 2.0)  # > 2 std
            results["metrics"]["n_drifted_dimensions"] = int(n_drifted_dims)
            
            # 4. Wasserstein on sample embeddings (if available)
            if baseline.get("sample_embeddings") is not None:
                sample_emb = baseline["sample_embeddings"]
                # Use first principal component for 1D Wasserstein
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                try:
                    baseline_1d = pca.fit_transform(sample_emb).flatten()
                    prod_1d = pca.transform(production_embeddings[:1000]).flatten()
                    wasserstein = stats.wasserstein_distance(prod_1d, baseline_1d)
                    results["metrics"]["embedding_wasserstein_1d"] = float(wasserstein)
                except Exception as e:
                    logger.warning(f"Could not compute embedding Wasserstein: {e}")
            
            # Determine drift status
            drift_detected = (
                normalized_mean_shift > 1.0 or  # > 1 std shift
                cosine_sim < 0.95 or            # < 95% similarity
                n_drifted_dims > production_embeddings.shape[1] * 0.1  # > 10% dims drifted
            )
            
            results["metrics"]["drift_detected"] = drift_detected
            
            if drift_detected:
                results["warnings"].append({
                    "type": "embedding_drift",
                    "message": f"Embedding drift: mean_shift={normalized_mean_shift:.3f}, cosine={cosine_sim:.3f}",
                    "severity": "MEDIUM"
                })
                status = "WARN"
                message = f"Embedding drift detected (shift={normalized_mean_shift:.2f}σ)"
            else:
                status = "PASS"
                message = "No significant embedding drift"
            
            logger.info(f"  Mean shift: {normalized_mean_shift:.3f}σ")
            logger.info(f"  Cosine similarity: {cosine_sim:.4f}")
            logger.info(f"  Drifted dimensions: {n_drifted_dims}/{production_embeddings.shape[1]}")
        else:
            status = "SKIP"
            message = "No embedding baseline available"
            logger.info("  No baseline - skipping comparison")
        
        results["summary"]["status"] = status
        results["summary"]["message"] = message
        
        logger.info(f"\n  Status: {status} - {message}")
        
        return results


# ============================================================================
# MAIN MONITORING ORCHESTRATOR
# ============================================================================

@dataclass
class MonitoringReport:
    """Complete monitoring report."""
    timestamp: str
    batch_id: str
    layer1_confidence: Dict
    layer2_temporal: Dict
    layer3_drift: Dict
    layer4_embedding: Dict  # Optional: may be empty if not computed
    overall_status: str
    overall_message: str
    gating_decision: str
    needs_review: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PostInferenceMonitor:
    """
    Orchestrate all monitoring layers (3 core + 1 optional).
    
    Layers:
        1. Confidence/Uncertainty (always run)
        2. Temporal Plausibility (always run)
        3. Sensor Drift (if production data available)
        4. Embedding Drift (optional, if model + embeddings available)
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.confidence_analyzer = ConfidenceAnalyzer(self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.embedding_detector = EmbeddingDriftDetector(self.config)
    
    def run(self,
            predictions_path: Path,
            production_data_path: Optional[Path] = None,
            baseline_path: Optional[Path] = None,
            embedding_baseline_path: Optional[Path] = None,
            model_path: Optional[Path] = None,
            output_dir: Optional[Path] = None) -> MonitoringReport:
        """
        Run complete monitoring pipeline.
        
        Args:
            predictions_path: Path to predictions CSV
            production_data_path: Path to production_X.npy (for drift detection)
            baseline_path: Path to baseline_stats.json
            embedding_baseline_path: Path to baseline_embeddings.npz (optional)
            model_path: Path to model (for embedding extraction, optional)
            output_dir: Directory to save reports
        
        Returns:
            MonitoringReport with all results
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        batch_id = predictions_path.stem.replace("predictions_", "")
        
        logger.info(f"\n{'='*60}")
        logger.info("🔍 POST-INFERENCE MONITORING")
        logger.info(f"{'='*60}")
        logger.info(f"Predictions: {predictions_path}")
        logger.info(f"Batch ID: {batch_id}")
        
        # Load predictions
        predictions_df = pd.read_csv(predictions_path)
        logger.info(f"Loaded {len(predictions_df)} predictions")
        
        # Layer 1: Confidence
        layer1_results = self.confidence_analyzer.analyze(predictions_df)
        
        # Layer 2: Temporal
        layer2_results = self.temporal_analyzer.analyze(predictions_df)
        
        # Layer 3: Sensor Drift (if data available)
        production_data = None
        if production_data_path and production_data_path.exists():
            production_data = np.load(production_data_path)
            layer3_results = self.drift_detector.analyze(
                production_data,
                baseline_path=baseline_path
            )
        else:
            logger.info("\n⏭️  Skipping Layer 3 (no production data provided)")
            layer3_results = {"summary": {"status": "SKIP", "message": "No data provided"}}
        
        # Layer 4: Embedding Drift (optional - strongest label-free argument)
        layer4_results = {"summary": {"status": "SKIP", "message": "Not computed"}}
        if (embedding_baseline_path and embedding_baseline_path.exists() and 
            model_path and model_path.exists() and production_data is not None):
            try:
                logger.info("\n🧠 Computing embedding drift (Layer 4)...")
                production_embeddings = self.embedding_detector.extract_embeddings(
                    model_path, production_data
                )
                layer4_results = self.embedding_detector.analyze(
                    production_embeddings,
                    baseline_path=embedding_baseline_path
                )
            except Exception as e:
                logger.warning(f"⚠️ Embedding drift analysis failed: {e}")
                layer4_results = {"summary": {"status": "SKIP", "message": f"Error: {e}"}}
        else:
            logger.info("\n⏭️  Skipping Layer 4 (embedding baseline or model not provided)")
        
        # Determine overall status (from all layers including Layer 4)
        statuses = [
            layer1_results["summary"]["status"],
            layer2_results["summary"]["status"],
            layer3_results["summary"]["status"],
            layer4_results["summary"]["status"]
        ]
        
        # Filter out SKIP statuses for decision
        active_statuses = [s for s in statuses if s != "SKIP"]
        
        if "BLOCK" in active_statuses:
            overall_status = "BLOCK"
            gating_decision = "BLOCK"
        elif "WARN" in active_statuses:
            overall_status = "WARN"
            gating_decision = "PASS_WITH_REVIEW"
        else:
            overall_status = "PASS"
            gating_decision = "PASS"
        
        needs_review = overall_status in ["WARN", "BLOCK"]
        
        # Create messages
        messages = []
        if layer1_results["summary"]["status"] != "PASS":
            messages.append(f"Layer1: {layer1_results['summary']['message']}")
        if layer2_results["summary"]["status"] != "PASS":
            messages.append(f"Layer2: {layer2_results['summary']['message']}")
        if layer3_results["summary"]["status"] not in ["PASS", "SKIP"]:
            messages.append(f"Layer3: {layer3_results['summary']['message']}")
        if layer4_results["summary"]["status"] not in ["PASS", "SKIP"]:
            messages.append(f"Layer4: {layer4_results['summary']['message']}")
        
        overall_message = "; ".join(messages) if messages else "All checks passed"
        
        # Create report
        report = MonitoringReport(
            timestamp=timestamp,
            batch_id=batch_id,
            layer1_confidence=layer1_results,
            layer2_temporal=layer2_results,
            layer3_drift=layer3_results,
            layer4_embedding=layer4_results,
            overall_status=overall_status,
            overall_message=overall_message,
            gating_decision=gating_decision,
            needs_review=needs_review
        )
        
        # Save reports
        if output_dir:
            output_dir = Path(output_dir)
            batch_dir = output_dir / f"{timestamp}_{batch_id}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Individual layer reports
            with open(batch_dir / "confidence_report.json", 'w') as f:
                json.dump(layer1_results, f, indent=2, cls=NumpyEncoder)
            
            with open(batch_dir / "temporal_report.json", 'w') as f:
                json.dump(layer2_results, f, indent=2, cls=NumpyEncoder)
            
            with open(batch_dir / "drift_report.json", 'w') as f:
                json.dump(layer3_results, f, indent=2, cls=NumpyEncoder)
            
            if layer4_results["summary"]["status"] != "SKIP":
                with open(batch_dir / "embedding_report.json", 'w') as f:
                    json.dump(layer4_results, f, indent=2, cls=NumpyEncoder)
            
            # Summary report
            with open(batch_dir / "summary.json", 'w') as f:
                json.dump(report.to_dict(), f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"\n📁 Reports saved to: {batch_dir}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 MONITORING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Layer 1 (Confidence): {layer1_results['summary']['status']}")
        logger.info(f"  Layer 2 (Temporal):   {layer2_results['summary']['status']}")
        logger.info(f"  Layer 3 (Drift):      {layer3_results['summary']['status']}")
        logger.info(f"  Layer 4 (Embedding):  {layer4_results['summary']['status']}")
        logger.info(f"  ---")
        logger.info(f"  Overall Status:   {overall_status}")
        logger.info(f"  Gating Decision:  {gating_decision}")
        logger.info(f"  Needs Review:     {needs_review}")
        
        return report


# ============================================================================
# MLFLOW INTEGRATION
# ============================================================================

def log_to_mlflow(
    report: MonitoringReport, 
    run_name: Optional[str] = None,
    config: Optional[MonitoringConfig] = None,
    model_version: Optional[str] = None,
    data_version: Optional[str] = None,
    baseline_version: Optional[str] = None
):
    """
    Log monitoring results to MLflow with standardized metrics/tags/artifacts.
    
    This creates a production-ready MLflow run with:
    - Tags: status, versions, config
    - Metrics: all 3 layers
    - Artifacts: JSON reports, CSVs
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not available - skipping logging")
        return
    
    config = config or MonitoringConfig()
    
    with mlflow.start_run(run_name=run_name or f"monitoring_{report.batch_id}"):
        # ==================== TAGS ====================
        # Status tags (most important for dashboards)
        mlflow.set_tag("monitoring_status", report.overall_status)
        mlflow.set_tag("gating_decision", report.gating_decision)
        mlflow.set_tag("needs_review", str(report.needs_review))
        mlflow.set_tag("batch_id", report.batch_id)
        
        # Version tags (reproducibility)
        if model_version:
            mlflow.set_tag("model_version", model_version)
        if data_version:
            mlflow.set_tag("data_version", data_version)
        if baseline_version:
            mlflow.set_tag("baseline_version", baseline_version)
        
        # Config tags (for debugging)
        mlflow.set_tag("window_size_seconds", str(config.window_duration_seconds))
        mlflow.set_tag("window_overlap", str(config.window_overlap))
        mlflow.set_tag("stride_seconds", str(config.window_stride_seconds))
        mlflow.set_tag("use_bonferroni", str(config.use_bonferroni))
        
        # ==================== METRICS ====================
        # Layer 1: Confidence metrics
        l1 = report.layer1_confidence.get("metrics", {})
        mlflow.log_metric("confidence/mean", l1.get("mean_confidence", 0))
        mlflow.log_metric("confidence/std", l1.get("std_confidence", 0))
        mlflow.log_metric("confidence/median", l1.get("median_confidence", 0))
        mlflow.log_metric("confidence/uncertain_ratio", l1.get("uncertain_ratio", 0))
        if "mean_entropy" in l1:
            mlflow.log_metric("confidence/entropy_mean", l1["mean_entropy"])
        if "mean_margin" in l1:
            mlflow.log_metric("confidence/margin_mean", l1["mean_margin"])
        
        # Layer 2: Temporal metrics
        l2 = report.layer2_temporal.get("metrics", {})
        mlflow.log_metric("temporal/flip_rate", l2.get("flip_rate", 0))
        mlflow.log_metric("temporal/n_bouts", l2.get("n_bouts", 0))
        mlflow.log_metric("temporal/median_dwell_time", l2.get("median_dwell_time", 0))
        mlflow.log_metric("temporal/mean_dwell_time", l2.get("mean_dwell_time", 0))
        
        # Layer 3: Drift metrics
        l3 = report.layer3_drift.get("metrics", {})
        if l3:
            mlflow.log_metric("drift/n_channels", l3.get("n_channels_with_drift", 0))
            
            # Find max Wasserstein and PSI across channels
            max_wasserstein = 0
            max_psi = 0
            for ch_name, ch_data in report.layer3_drift.get("per_channel", {}).items():
                w = ch_data.get("wasserstein_distance", 0)
                p = ch_data.get("psi", 0)
                if w > max_wasserstein:
                    max_wasserstein = w
                if p > max_psi:
                    max_psi = p
            
            mlflow.log_metric("drift/max_wasserstein", max_wasserstein)
            mlflow.log_metric("drift/max_psi", max_psi)
        
        # Layer 4: Embedding metrics (optional but strong for thesis)
        l4 = report.layer4_embedding.get("metrics", {})
        if l4:
            if "embedding_normalized_mean_shift" in l4:
                mlflow.log_metric("embedding/mean_shift", l4["embedding_normalized_mean_shift"])
            if "embedding_cosine_similarity" in l4:
                mlflow.log_metric("embedding/cosine_similarity", l4["embedding_cosine_similarity"])
            if "n_drifted_dimensions" in l4:
                mlflow.log_metric("embedding/n_drifted_dims", l4["n_drifted_dimensions"])
            if "embedding_wasserstein_1d" in l4:
                mlflow.log_metric("embedding/wasserstein_1d", l4["embedding_wasserstein_1d"])
        
        # ==================== ARTIFACTS ====================
        import tempfile
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 1. Full monitoring report JSON
            summary_path = tmpdir / "monitoring_report.json"
            with open(summary_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, cls=NumpyEncoder)
            mlflow.log_artifact(str(summary_path))
            
            # 2. Drift summary CSV (easy to read)
            if report.layer3_drift.get("per_channel"):
                drift_csv = tmpdir / "drift_summary.csv"
                with open(drift_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["channel", "mean", "std", "baseline_mean", 
                                    "ks_pvalue", "wasserstein", "psi", "drift_detected", "method"])
                    for ch, data in report.layer3_drift["per_channel"].items():
                        writer.writerow([
                            ch,
                            f"{data.get('mean', 0):.4f}",
                            f"{data.get('std', 0):.4f}",
                            f"{data.get('baseline_mean', 0):.4f}",
                            f"{data.get('ks_pvalue', 0):.6f}",
                            f"{data.get('wasserstein_distance', 0):.4f}",
                            f"{data.get('psi', 'N/A')}",
                            data.get('drift_detected', False),
                            data.get('comparison_method', 'unknown')
                        ])
                mlflow.log_artifact(str(drift_csv))
            
            # 3. Transition matrix CSV
            if report.layer2_temporal.get("transition_matrix"):
                trans_csv = tmpdir / "transition_matrix.csv"
                trans = report.layer2_temporal["transition_matrix"]
                classes = sorted(trans.keys())
                with open(trans_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["from_class"] + classes)
                    for from_class in classes:
                        row = [from_class] + [trans.get(from_class, {}).get(to, 0) for to in classes]
                        writer.writerow(row)
                mlflow.log_artifact(str(trans_csv))
        
        logger.info(f"📊 Logged monitoring results to MLflow run: {mlflow.active_run().info.run_id}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-Inference Monitoring")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to production_X.npy (for drift detection)')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline_stats.json')
    parser.add_argument('--embedding-baseline', type=str, default=None,
                       help='Path to baseline_embeddings.npz (for Layer 4)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model (for embedding extraction)')
    parser.add_argument('--output-dir', type=str, default='reports/monitoring',
                       help='Output directory for reports')
    parser.add_argument('--mlflow', action='store_true',
                       help='Log results to MLflow')
    parser.add_argument('--use-bonferroni', action='store_true',
                       help='Apply Bonferroni correction for multiple comparisons')
    args = parser.parse_args()
    
    # Resolve paths
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        predictions_path = PROJECT_ROOT / args.predictions
    
    if not predictions_path.exists():
        logger.error(f"❌ Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    production_data_path = None
    if args.data:
        production_data_path = Path(args.data)
        if not production_data_path.exists():
            production_data_path = PROJECT_ROOT / args.data
    
    baseline_path = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            baseline_path = PROJECT_ROOT / args.baseline
    elif (DATA_PREPARED / "baseline_stats.json").exists():
        baseline_path = DATA_PREPARED / "baseline_stats.json"
    
    # Embedding paths (optional Layer 4)
    embedding_baseline_path = None
    if args.embedding_baseline:
        embedding_baseline_path = Path(args.embedding_baseline)
        if not embedding_baseline_path.exists():
            embedding_baseline_path = PROJECT_ROOT / args.embedding_baseline
    elif (DATA_PREPARED / "baseline_embeddings.npz").exists():
        embedding_baseline_path = DATA_PREPARED / "baseline_embeddings.npz"
    
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            model_path = PROJECT_ROOT / args.model
    elif (PROJECT_ROOT / "models/pretrained/fine_tuned_model_1dcnnbilstm.keras").exists():
        model_path = PROJECT_ROOT / "models/pretrained/fine_tuned_model_1dcnnbilstm.keras"
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir
    
    # Create config with optional Bonferroni
    config = MonitoringConfig(use_bonferroni=args.use_bonferroni)
    
    # Run monitoring
    monitor = PostInferenceMonitor(config=config)
    report = monitor.run(
        predictions_path=predictions_path,
        production_data_path=production_data_path,
        baseline_path=baseline_path,
        embedding_baseline_path=embedding_baseline_path,
        model_path=model_path,
        output_dir=output_dir
    )
    
    # Log to MLflow if requested
    if args.mlflow:
        log_to_mlflow(report, config=config)
    
    # Exit code based on status
    if report.overall_status == "BLOCK":
        logger.error("❌ MONITORING FAILED - Blocking pipeline")
        sys.exit(1)
    elif report.overall_status == "WARN":
        logger.warning("⚠️ MONITORING WARNINGS - Review recommended")
        sys.exit(0)
    else:
        logger.info("✅ MONITORING PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
