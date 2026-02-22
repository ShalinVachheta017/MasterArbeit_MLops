"""
Robustness & Noise Injection Test Suite
========================================

Systematic robustness evaluation framework for the HAR pipeline.
Tests model performance under realistic deployment conditions:

    1. Gaussian noise injection at varying SNR levels
    2. Missing data simulation (window-level and sample-level)
    3. Sampling rate jitter (simulating imprecise device clocks)
    4. Sensor saturation / clipping
    5. Combined degradation profiles

Reference papers:
    - "Comparative Study on the Effects of Noise in HAR"
    - "Resilience of ML Models in Anxiety Detection: Assessing
       the Impact of Gaussian Noise on Wearable Sensors"

Usage:
    from src.robustness import RobustnessEvaluator

    evaluator = RobustnessEvaluator(model)
    report = evaluator.full_evaluation(X_test, y_test)
    evaluator.save_report(report, "reports/robustness/")

Author: HAR MLOps Pipeline
Date: February 2026
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class RobustnessConfig:
    """Configuration for robustness evaluation."""

    # Gaussian noise levels (fraction of signal std)
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.50])

    # Missing data ratios
    missing_ratios: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.20])

    # Sampling rate jitter (fraction deviation from 50Hz)
    jitter_levels: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.10])

    # Sensor saturation thresholds
    saturation_thresholds: List[float] = field(default_factory=lambda: [50.0, 20.0, 10.0, 5.0])

    # Batch size for inference
    batch_size: int = 64

    # Number of random seeds for stability
    n_seeds: int = 3

    # Output
    save_degradation_curves: bool = True


# ============================================================================
# NOISE INJECTORS
# ============================================================================


class GaussianNoiseInjector:
    """Add Gaussian noise at a specified fraction of signal std."""

    def inject(
        self,
        X: np.ndarray,
        noise_level: float,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Add Gaussian noise to sensor data.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)
        noise_level : float
            Fraction of per-channel std to use as noise std.
        seed : int

        Returns
        -------
        X_noisy : np.ndarray, same shape
        """
        if noise_level <= 0:
            return X.copy()

        rng = np.random.RandomState(seed)
        X_noisy = X.copy()

        for ch in range(X.shape[2]):
            ch_std = np.std(X[:, :, ch])
            noise_std = noise_level * ch_std
            noise = rng.normal(0, noise_std, size=X[:, :, ch].shape)
            X_noisy[:, :, ch] += noise

        return X_noisy


class MissingDataInjector:
    """Simulate missing data by zeroing random segments."""

    def inject(
        self,
        X: np.ndarray,
        missing_ratio: float,
        mode: str = "sample",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate missing data.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)
        missing_ratio : float — fraction of data to zero out
        mode : str — "sample" (random samples) or "window" (entire windows)
        seed : int

        Returns
        -------
        X_missing : np.ndarray
        """
        if missing_ratio <= 0:
            return X.copy()

        rng = np.random.RandomState(seed)
        X_missing = X.copy()

        if mode == "window":
            n_drop = int(len(X) * missing_ratio)
            drop_idx = rng.choice(len(X), n_drop, replace=False)
            X_missing[drop_idx] = 0.0
        else:
            mask = rng.random(X.shape) < missing_ratio
            X_missing[mask] = 0.0

        return X_missing


class SamplingJitterInjector:
    """Simulate sampling rate jitter by resampling with slight distortion."""

    def inject(
        self,
        X: np.ndarray,
        jitter_fraction: float,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate sampling rate jitter.

        Randomly stretches/compresses the time axis per window.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)
        jitter_fraction : float — max relative deviation (e.g. 0.05 = ±5%)
        seed : int

        Returns
        -------
        X_jittered : np.ndarray, same shape (resampled back to original length)
        """
        if jitter_fraction <= 0:
            return X.copy()

        rng = np.random.RandomState(seed)
        N, T, C = X.shape
        X_jittered = np.zeros_like(X)

        for i in range(N):
            # Random stretch factor per window
            stretch = 1.0 + rng.uniform(-jitter_fraction, jitter_fraction)
            new_T = int(T * stretch)
            new_T = max(new_T, 3)  # Avoid degenerate

            for ch in range(C):
                # Resample to new_T then back to T
                original = X[i, :, ch]
                x_old = np.linspace(0, 1, T)
                x_new = np.linspace(0, 1, new_T)
                resampled = np.interp(x_new, x_old, original)
                x_final = np.linspace(0, 1, T)
                X_jittered[i, :, ch] = np.interp(x_final, x_new, resampled)

        return X_jittered


class SaturationInjector:
    """Simulate sensor saturation / clipping."""

    def inject(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Clip values to [-threshold, threshold]."""
        return np.clip(X, -threshold, threshold)


# ============================================================================
# ROBUSTNESS EVALUATOR
# ============================================================================


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation framework.

    Systematically tests model accuracy under various degradation types,
    producing degradation curves for the thesis.
    """

    def __init__(self, model=None, config: RobustnessConfig = None):
        self.model = model
        self.config = config or RobustnessConfig()

        self.noise_injector = GaussianNoiseInjector()
        self.missing_injector = MissingDataInjector()
        self.jitter_injector = SamplingJitterInjector()
        self.saturation_injector = SaturationInjector()

    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return (predictions, confidences)."""
        probs = self.model.predict(
            X.astype(np.float32),
            batch_size=self.config.batch_size,
            verbose=0,
        )
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return predictions, confidences

    def _evaluate_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Compute accuracy and confidence metrics."""
        preds, confs = self._predict(X)
        accuracy = float(np.mean(preds == y))
        return {
            "accuracy": accuracy,
            "mean_confidence": float(np.mean(confs)),
            "low_confidence_ratio": float(np.mean(confs < 0.65)),
        }

    def evaluate_noise_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Test accuracy across Gaussian noise levels."""
        results = {}
        for level in self.config.noise_levels:
            seed_results = []
            for seed in range(self.config.n_seeds):
                X_noisy = self.noise_injector.inject(X, level, seed=seed + 42)
                metrics = self._evaluate_accuracy(X_noisy, y)
                seed_results.append(metrics)

            # Average across seeds
            results[f"noise_{level:.2f}"] = {
                "noise_level": level,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in seed_results])),
                "accuracy_std": float(np.std([r["accuracy"] for r in seed_results])),
                "confidence_mean": float(np.mean([r["mean_confidence"] for r in seed_results])),
            }
            logger.info(
                "Noise %.2f → accuracy=%.4f (±%.4f)",
                level,
                results[f"noise_{level:.2f}"]["accuracy_mean"],
                results[f"noise_{level:.2f}"]["accuracy_std"],
            )
        return results

    def evaluate_missing_data_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Test accuracy across missing data ratios."""
        results = {}
        for ratio in self.config.missing_ratios:
            seed_results = []
            for seed in range(self.config.n_seeds):
                X_missing = self.missing_injector.inject(X, ratio, seed=seed + 42)
                metrics = self._evaluate_accuracy(X_missing, y)
                seed_results.append(metrics)

            results[f"missing_{ratio:.2f}"] = {
                "missing_ratio": ratio,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in seed_results])),
                "accuracy_std": float(np.std([r["accuracy"] for r in seed_results])),
            }
        return results

    def evaluate_jitter_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Test accuracy across sampling rate jitter levels."""
        results = {}
        for level in self.config.jitter_levels:
            X_jittered = self.jitter_injector.inject(X, level)
            metrics = self._evaluate_accuracy(X_jittered, y)
            results[f"jitter_{level:.2f}"] = {
                "jitter_level": level,
                **metrics,
            }
        return results

    def full_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run complete robustness evaluation across all degradation types.

        Returns a comprehensive report suitable for thesis.
        """
        logger.info("=" * 60)
        logger.info("ROBUSTNESS EVALUATION")
        logger.info("  Samples: %d  |  Window shape: %s", len(X), X.shape[1:])
        logger.info("=" * 60)

        # Baseline (clean)
        baseline = self._evaluate_accuracy(X, y)
        logger.info("Baseline accuracy: %.4f", baseline["accuracy"])

        return {
            "baseline": baseline,
            "noise": self.evaluate_noise_robustness(X, y),
            "missing_data": self.evaluate_missing_data_robustness(X, y),
            "jitter": self.evaluate_jitter_robustness(X, y),
            "summary": {
                "n_test_samples": int(len(X)),
                "baseline_accuracy": baseline["accuracy"],
                "n_noise_levels": len(self.config.noise_levels),
                "n_missing_levels": len(self.config.missing_ratios),
                "n_jitter_levels": len(self.config.jitter_levels),
            },
        }

    def save_report(self, report: Dict, output_dir: str):
        """Save robustness report as JSON."""
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / "robustness_report.json"

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Robustness report saved: %s", path)

    def save_degradation_curves(self, report: Dict, output_dir: str):
        """Save degradation curves as PNG plots."""
        try:
            import matplotlib.pyplot as plt

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Noise degradation
            if "noise" in report:
                levels = [v["noise_level"] for v in report["noise"].values()]
                accs = [v["accuracy_mean"] for v in report["noise"].values()]
                stds = [v["accuracy_std"] for v in report["noise"].values()]
                axes[0].errorbar(levels, accs, yerr=stds, marker="o", capsize=5)
                axes[0].set_xlabel("Noise Level (fraction of signal std)")
                axes[0].set_ylabel("Accuracy")
                axes[0].set_title("Gaussian Noise Robustness")
                axes[0].axhline(
                    y=report["baseline"]["accuracy"], color="r", linestyle="--", label="Baseline"
                )
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Missing data degradation
            if "missing_data" in report:
                ratios = [v["missing_ratio"] for v in report["missing_data"].values()]
                accs = [v["accuracy_mean"] for v in report["missing_data"].values()]
                axes[1].plot(ratios, accs, marker="s", color="green")
                axes[1].set_xlabel("Missing Data Ratio")
                axes[1].set_ylabel("Accuracy")
                axes[1].set_title("Missing Data Robustness")
                axes[1].axhline(
                    y=report["baseline"]["accuracy"], color="r", linestyle="--", label="Baseline"
                )
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            # Jitter degradation
            if "jitter" in report:
                levels = [v["jitter_level"] for v in report["jitter"].values()]
                accs = [v["accuracy"] for v in report["jitter"].values()]
                axes[2].plot(levels, accs, marker="^", color="orange")
                axes[2].set_xlabel("Sampling Rate Jitter (fraction)")
                axes[2].set_ylabel("Accuracy")
                axes[2].set_title("Sampling Jitter Robustness")
                axes[2].axhline(
                    y=report["baseline"]["accuracy"], color="r", linestyle="--", label="Baseline"
                )
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            plt.suptitle("Model Robustness Under Sensor Degradation", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_path / "degradation_curves.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Degradation curves saved: %s", output_path)

        except ImportError:
            logger.warning("matplotlib not available — skipping plots.")
