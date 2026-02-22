#!/usr/bin/env python3
"""
Energy-Based Out-of-Distribution Detection
===========================================

Implements energy-based OOD detection for the HAR monitoring pipeline.
Based on the NeurIPS 2020 paper: "Energy-based Out-of-distribution Detection"

Energy Score: E(x) = -log(sum(exp(f_i(x))))

Lower energy = in-distribution (model confident)
Higher energy = out-of-distribution (potentially novel/anomalous)

Usage:
    from ood_detection import EnergyOODDetector
    
    detector = EnergyOODDetector(model)
    energy_scores = detector.compute_energy(logits)
    is_ood = detector.detect_ood(energy_scores)

Author: HAR MLOps Pipeline
Date: January 30, 2026
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class OODConfig:
    """Configuration for OOD detection."""

    # Energy thresholds (calibrated on validation data)
    energy_threshold_warn: float = -5.0  # Above this = warning
    energy_threshold_critical: float = -2.0  # Above this = critical OOD

    # Temperature for energy calculation
    temperature: float = 1.0

    # Ensemble approach
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = None

    # Reconstruction-based (if autoencoder available)
    reconstruction_threshold: float = 0.5

    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {"energy": 0.4, "entropy": 0.3, "confidence": 0.3}


# ============================================================================
# ENERGY SCORE COMPUTATION
# ============================================================================


class EnergyOODDetector:
    """
    Energy-based Out-of-Distribution detector.

    Uses the negative log-sum-exp of logits as an energy score.
    This has been shown to be more effective than softmax confidence
    for detecting OOD samples.
    """

    def __init__(self, config: OODConfig = None):
        self.config = config or OODConfig()
        self.logger = logging.getLogger(f"{__name__}.EnergyOOD")

        # Calibration statistics (set from validation data)
        self.in_dist_energy_mean: Optional[float] = None
        self.in_dist_energy_std: Optional[float] = None

    def compute_energy(self, logits: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Compute energy scores from logits.

        Energy = -T * log(sum(exp(f_i / T)))

        Args:
            logits: Raw model outputs before softmax, shape (N, C)
            temperature: Temperature scaling (default from config)

        Returns:
            Energy scores, shape (N,). Lower = more in-distribution.
        """
        T = temperature or self.config.temperature

        # Numerical stability: subtract max
        logits_scaled = logits / T
        max_logits = np.max(logits_scaled, axis=1, keepdims=True)

        # Log-sum-exp trick
        energy = -T * (
            max_logits.squeeze() + np.log(np.sum(np.exp(logits_scaled - max_logits), axis=1))
        )

        return energy

    def compute_energy_from_probs(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute approximate energy from softmax probabilities.

        When logits aren't available, we can approximate by:
        E ≈ -log(sum(p_i)) = -log(1) = 0 for true probs

        But we can use max-logit approximation:
        E ≈ -log(max(p))

        Args:
            probabilities: Softmax outputs, shape (N, C)

        Returns:
            Approximate energy scores
        """
        # Max probability as proxy
        max_prob = np.max(probabilities, axis=1)

        # -log(max_prob) as energy proxy (higher for uncertain)
        energy = -np.log(max_prob + 1e-10)

        return energy

    def calibrate(self, in_dist_energies: np.ndarray):
        """
        Calibrate thresholds using in-distribution validation data.

        Args:
            in_dist_energies: Energy scores from validation set
        """
        self.in_dist_energy_mean = float(np.mean(in_dist_energies))
        self.in_dist_energy_std = float(np.std(in_dist_energies))

        # Set thresholds based on statistics
        # Warning: 2 std above mean
        # Critical: 3 std above mean
        self.config.energy_threshold_warn = self.in_dist_energy_mean + 2 * self.in_dist_energy_std
        self.config.energy_threshold_critical = (
            self.in_dist_energy_mean + 3 * self.in_dist_energy_std
        )

        self.logger.info(
            f"Calibrated: mean={self.in_dist_energy_mean:.3f}, std={self.in_dist_energy_std:.3f}"
        )
        self.logger.info(
            f"Thresholds: warn={self.config.energy_threshold_warn:.3f}, critical={self.config.energy_threshold_critical:.3f}"
        )

    def detect_ood(self, energy_scores: np.ndarray, return_mask: bool = False) -> Dict[str, Any]:
        """
        Detect OOD samples based on energy scores.

        Args:
            energy_scores: Energy scores to evaluate
            return_mask: If True, return boolean mask of OOD samples

        Returns:
            Dictionary with OOD detection results
        """
        n_samples = len(energy_scores)

        # Classify based on thresholds
        warn_mask = energy_scores > self.config.energy_threshold_warn
        critical_mask = energy_scores > self.config.energy_threshold_critical

        n_warn = np.sum(warn_mask & ~critical_mask)
        n_critical = np.sum(critical_mask)
        n_normal = n_samples - n_warn - n_critical

        results = {
            "n_samples": n_samples,
            "n_normal": int(n_normal),
            "n_warning": int(n_warn),
            "n_critical": int(n_critical),
            "ood_ratio": float((n_warn + n_critical) / n_samples),
            "critical_ratio": float(n_critical / n_samples),
            "mean_energy": float(np.mean(energy_scores)),
            "std_energy": float(np.std(energy_scores)),
            "max_energy": float(np.max(energy_scores)),
            "min_energy": float(np.min(energy_scores)),
            "thresholds": {
                "warn": self.config.energy_threshold_warn,
                "critical": self.config.energy_threshold_critical,
            },
        }

        if return_mask:
            results["ood_mask"] = critical_mask | warn_mask
            results["critical_mask"] = critical_mask

        return results


# ============================================================================
# ENSEMBLE OOD DETECTOR
# ============================================================================


class EnsembleOODDetector:
    """
    Ensemble OOD detector combining multiple signals:
    - Energy score
    - Entropy
    - Max confidence
    - (Optional) Reconstruction error

    Multiple signals reduce false positives.
    """

    def __init__(self, config: OODConfig = None):
        self.config = config or OODConfig()
        self.energy_detector = EnergyOODDetector(config)
        self.logger = logging.getLogger(f"{__name__}.EnsembleOOD")

    def compute_all_scores(
        self, probabilities: np.ndarray, logits: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute all OOD scores.

        Args:
            probabilities: Softmax outputs
            logits: Raw logits (optional, for true energy)

        Returns:
            Dictionary of score arrays
        """
        scores = {}

        # Energy score
        if logits is not None:
            scores["energy"] = self.energy_detector.compute_energy(logits)
        else:
            scores["energy"] = self.energy_detector.compute_energy_from_probs(probabilities)

        # Entropy
        scores["entropy"] = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        # Negative confidence (higher = more OOD)
        scores["neg_confidence"] = 1.0 - np.max(probabilities, axis=1)

        # Margin (difference between top-2 predictions)
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        scores["neg_margin"] = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])

        return scores

    def compute_ensemble_score(
        self, scores: Dict[str, np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """
        Compute weighted ensemble OOD score.

        Args:
            scores: Dictionary of individual scores
            normalize: Whether to normalize scores before combining

        Returns:
            Combined OOD score (higher = more likely OOD)
        """
        # Normalize each score to [0, 1] range
        normalized_scores = {}

        for name, values in scores.items():
            if normalize:
                min_val, max_val = values.min(), values.max()
                if max_val - min_val > 1e-6:
                    normalized_scores[name] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_scores[name] = np.zeros_like(values)
            else:
                normalized_scores[name] = values

        # Weighted combination
        weights = self.config.ensemble_weights
        ensemble_score = np.zeros(len(next(iter(scores.values()))))

        weight_sum = 0
        for name, weight in weights.items():
            if name in normalized_scores:
                ensemble_score += weight * normalized_scores[name]
                weight_sum += weight

        if weight_sum > 0:
            ensemble_score /= weight_sum

        return ensemble_score

    def detect(
        self, probabilities: np.ndarray, logits: Optional[np.ndarray] = None, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Full OOD detection pipeline.

        Args:
            probabilities: Model softmax outputs
            logits: Raw logits (optional)
            threshold: Ensemble score threshold for OOD

        Returns:
            Detection results
        """
        self.logger.info("Running ensemble OOD detection...")

        # Compute all scores
        scores = self.compute_all_scores(probabilities, logits)

        # Compute ensemble
        ensemble_score = self.compute_ensemble_score(scores)

        # Detect OOD
        ood_mask = ensemble_score > threshold
        n_ood = np.sum(ood_mask)

        results = {
            "n_samples": len(probabilities),
            "n_ood": int(n_ood),
            "ood_ratio": float(n_ood / len(probabilities)),
            "mean_ensemble_score": float(np.mean(ensemble_score)),
            "individual_scores": {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "max": float(np.max(values)),
                }
                for name, values in scores.items()
            },
            "ood_indices": np.where(ood_mask)[0].tolist()[:100],  # Limit output
            "threshold": threshold,
        }

        # Get severity breakdown using energy
        energy_results = self.energy_detector.detect_ood(scores["energy"])
        results["energy_breakdown"] = {
            "n_normal": energy_results["n_normal"],
            "n_warning": energy_results["n_warning"],
            "n_critical": energy_results["n_critical"],
        }

        return results


# ============================================================================
# INTEGRATION WITH MONITORING
# ============================================================================


def add_ood_metrics_to_monitoring(
    monitoring_report: Dict, probabilities: np.ndarray, logits: Optional[np.ndarray] = None
) -> Dict:
    """
    Add OOD detection metrics to existing monitoring report.

    This function integrates with post_inference_monitoring.py

    Args:
        monitoring_report: Existing monitoring report dictionary
        probabilities: Model softmax outputs
        logits: Raw model logits (optional)

    Returns:
        Updated monitoring report with OOD metrics
    """
    detector = EnsembleOODDetector()

    # Run detection
    ood_results = detector.detect(probabilities, logits)

    # Add to report
    monitoring_report["ood_detection"] = {
        "ensemble_ood_ratio": ood_results["ood_ratio"],
        "n_ood_samples": ood_results["n_ood"],
        "mean_ensemble_score": ood_results["mean_ensemble_score"],
        "energy_breakdown": ood_results["energy_breakdown"],
        "individual_scores": ood_results["individual_scores"],
    }

    # Update summary status if OOD is significant
    if ood_results["ood_ratio"] > 0.2:
        if "summary" not in monitoring_report:
            monitoring_report["summary"] = {}
        monitoring_report["summary"]["ood_alert"] = True
        monitoring_report["summary"]["ood_ratio"] = ood_results["ood_ratio"]

    return monitoring_report


# ============================================================================
# CLI
# ============================================================================


def main():
    """Demo of OOD detection."""
    import argparse

    parser = argparse.ArgumentParser(description="OOD Detection Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("ENERGY-BASED OOD DETECTION DEMO")
        print("=" * 60)

        np.random.seed(42)

        # Simulate in-distribution data (confident predictions)
        n_in_dist = 100
        in_dist_probs = np.random.dirichlet(np.ones(11) * 5, size=n_in_dist)

        # Simulate OOD data (uncertain predictions)
        n_ood = 20
        ood_probs = np.random.dirichlet(np.ones(11) * 0.5, size=n_ood)

        # Combine
        all_probs = np.vstack([in_dist_probs, ood_probs])

        # Run detection
        detector = EnsembleOODDetector()
        results = detector.detect(all_probs)

        print(f"\nTotal samples: {results['n_samples']}")
        print(f"Detected OOD: {results['n_ood']} ({results['ood_ratio']:.1%})")
        print(f"\nEnergy breakdown:")
        print(f"  Normal: {results['energy_breakdown']['n_normal']}")
        print(f"  Warning: {results['energy_breakdown']['n_warning']}")
        print(f"  Critical: {results['energy_breakdown']['n_critical']}")

        print(f"\nIndividual score means:")
        for name, stats in results["individual_scores"].items():
            print(f"  {name}: {stats['mean']:.3f}")

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
