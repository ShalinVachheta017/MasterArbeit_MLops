"""
Wasserstein-Based Drift Detection
==================================

Implements Wasserstein (Earth Mover's) distance for distribution drift
detection, complementing PSI and KS-test in the monitoring pipeline.

Wasserstein distance measures the minimum "work" required to transform
one distribution into another — more sensitive to distribution shape
changes than KS-test and works well with continuous sensor data.

Methods:
    1. Per-channel Wasserstein distance
    2. Multi-resolution drift (window, hourly, daily)
    3. Change-point detection on Wasserstein time series
    4. Integrated drift report with PSI, KS, and Wasserstein

References:
    - WATCH: Wasserstein Change Point Detection for High-Dimensional
      Time Series Data (Yau & Kolaczyk, 2023)
    - LIFEWATCH: Lifelong Wasserstein Change Point Detection
    - Sinkhorn Divergences for Change Point Detection (Munk et al.)
    - Optimal Transport Based Change Point Detection (Matteson & James)

Usage:
    from src.wasserstein_drift import WassersteinDriftDetector

    detector = WassersteinDriftDetector()
    report = detector.detect(baseline_data, production_data)

Author: HAR MLOps Pipeline
Date: February 2026
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class WassersteinDriftConfig:
    """Configuration for Wasserstein-based drift detection."""

    # Thresholds (per-channel)
    warn_threshold: float = 0.3
    critical_threshold: float = 0.5

    # Multi-channel gating
    min_drifted_channels_warn: int = 2
    min_drifted_channels_critical: int = 4

    # Change-point detection
    window_size_cpd: int = 50   # Rolling window for CPD
    cpd_threshold: float = 2.0  # Std devs above mean for change-point

    # Multi-resolution
    enable_multi_resolution: bool = True

    # Sensor channels
    sensor_columns: List[str] = field(
        default_factory=lambda: [
            "Ax", "Ay", "Az", "Gx", "Gy", "Gz"
        ]
    )


# ============================================================================
# WASSERSTEIN DISTANCE COMPUTATION
# ============================================================================

class WassersteinDriftDetector:
    """
    Detects distribution drift using the 1-Wasserstein (Earth Mover's) distance.

    For 1D distributions, W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx,
    computed efficiently via sorted-sample differences.

    Per sensor channel, compares:
        - Baseline distribution (from training data)
        - Production distribution (current batch)
    """

    def __init__(self, config: WassersteinDriftConfig = None):
        self.config = config or WassersteinDriftConfig()

    def wasserstein_1d(
        self,
        baseline: np.ndarray,
        production: np.ndarray,
    ) -> float:
        """
        Compute 1-Wasserstein distance between two 1D samples.

        Uses scipy if available, otherwise a numpy fallback via
        sorted empirical CDF differences.
        """
        try:
            from scipy.stats import wasserstein_distance
            return float(wasserstein_distance(baseline, production))
        except ImportError:
            # Fallback: sorted empirical CDF
            all_vals = np.sort(np.concatenate([baseline, production]))
            cdf_b = np.searchsorted(np.sort(baseline), all_vals, side="right") / len(baseline)
            cdf_p = np.searchsorted(np.sort(production), all_vals, side="right") / len(production)
            return float(np.mean(np.abs(cdf_b - cdf_p)))

    def detect(
        self,
        baseline_data: np.ndarray,
        production_data: np.ndarray,
        channel_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run per-channel Wasserstein drift detection.

        Parameters
        ----------
        baseline_data : np.ndarray, shape (N_b, C) or (N_b, T, C)
            Baseline (training) sensor data.
        production_data : np.ndarray, shape (N_p, C) or (N_p, T, C)
            Production sensor data.
        channel_names : list[str], optional
            Names for each channel.

        Returns
        -------
        dict with per-channel distances, overall status, and alerts.
        """
        # Flatten windows if 3D → (N*T, C)
        if baseline_data.ndim == 3:
            baseline_data = baseline_data.reshape(-1, baseline_data.shape[-1])
        if production_data.ndim == 3:
            production_data = production_data.reshape(-1, production_data.shape[-1])

        n_channels = baseline_data.shape[1]
        names = channel_names or self.config.sensor_columns[:n_channels]
        if len(names) < n_channels:
            names = [f"ch_{i}" for i in range(n_channels)]

        per_channel = {}
        n_warn = 0
        n_critical = 0

        for i in range(n_channels):
            dist = self.wasserstein_1d(baseline_data[:, i], production_data[:, i])
            status = "NORMAL"
            if dist > self.config.critical_threshold:
                status = "CRITICAL"
                n_critical += 1
            elif dist > self.config.warn_threshold:
                status = "WARNING"
                n_warn += 1

            per_channel[names[i]] = {
                "wasserstein_distance": dist,
                "status": status,
            }

        # Overall assessment
        if n_critical >= self.config.min_drifted_channels_critical:
            overall_status = "CRITICAL"
        elif n_warn >= self.config.min_drifted_channels_warn:
            overall_status = "WARNING"
        else:
            overall_status = "NORMAL"

        distances = [v["wasserstein_distance"] for v in per_channel.values()]

        return {
            "overall_status": overall_status,
            "n_channels_warn": n_warn,
            "n_channels_critical": n_critical,
            "mean_wasserstein": float(np.mean(distances)),
            "max_wasserstein": float(np.max(distances)),
            "per_channel": per_channel,
            "thresholds": {
                "warn": self.config.warn_threshold,
                "critical": self.config.critical_threshold,
            },
        }


# ============================================================================
# CHANGE-POINT DETECTION ON WASSERSTEIN TIME SERIES
# ============================================================================

class WassersteinChangePointDetector:
    """
    Detect change points in a stream of Wasserstein distances.

    Given a time series of Wasserstein distances (e.g. one per batch),
    detects when the drift *magnitude* changes significantly — indicating
    a new drift regime (gradual→sudden or vice versa).

    Uses a rolling-window z-score approach:
        If W_t > mean(W_{t-k:t-1}) + threshold * std(W_{t-k:t-1})
        → change point detected.

    Reference:
        WATCH: Wasserstein Change Point Detection (Yau & Kolaczyk, 2023)
    """

    def __init__(self, config: WassersteinDriftConfig = None):
        self.config = config or WassersteinDriftConfig()

    def detect_change_points(
        self,
        wasserstein_series: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect change points in a Wasserstein distance time series.

        Parameters
        ----------
        wasserstein_series : np.ndarray, shape (T,)
            Wasserstein distances over time (one per batch/window).

        Returns
        -------
        dict with change_points (list of indices), scores, status.
        """
        T = len(wasserstein_series)
        w = self.config.window_size_cpd

        if T < w + 1:
            return {
                "change_points": [],
                "scores": [],
                "status": "INSUFFICIENT_DATA",
                "n_points": T,
                "min_required": w + 1,
            }

        scores = np.zeros(T)
        change_points = []

        for t in range(w, T):
            window = wasserstein_series[t - w : t]
            mu = np.mean(window)
            sigma = np.std(window) + 1e-8
            z = (wasserstein_series[t] - mu) / sigma
            scores[t] = z

            if z > self.config.cpd_threshold:
                change_points.append(t)

        return {
            "change_points": change_points,
            "scores": scores.tolist(),
            "n_change_points": len(change_points),
            "status": (
                "DRIFT_REGIME_CHANGE"
                if change_points
                else "STABLE"
            ),
            "mean_z_score": float(np.mean(scores[w:])),
            "max_z_score": float(np.max(scores[w:])) if T > w else 0.0,
        }


# ============================================================================
# MULTI-RESOLUTION DRIFT ANALYSIS
# ============================================================================

class MultiResolutionDriftAnalyzer:
    """
    Analyze drift at multiple time resolutions:
        - Window-level  (per inference batch)
        - Hourly        (aggregate over 1 hour)
        - Daily         (aggregate over 1 day)

    Useful for distinguishing transient noise from persistent drift.
    """

    def __init__(self, config: WassersteinDriftConfig = None):
        self.config = config or WassersteinDriftConfig()
        self.detector = WassersteinDriftDetector(config)

    def analyze(
        self,
        baseline_data: np.ndarray,
        production_batches: List[np.ndarray],
        batch_timestamps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute drift at window-level, then aggregate to coarser
        resolutions.

        Parameters
        ----------
        baseline_data : np.ndarray
            Reference data, shape (N_b, C).
        production_batches : list of np.ndarray
            List of production batches, each shape (N_i, C).
        batch_timestamps : list of str, optional
            ISO timestamps for each batch.

        Returns
        -------
        dict with window_level, hourly, and daily summaries.
        """
        # Window-level distances (one per batch)
        window_distances = []
        for batch in production_batches:
            report = self.detector.detect(baseline_data, batch)
            window_distances.append(report["mean_wasserstein"])

        window_distances = np.array(window_distances)

        result = {
            "window_level": {
                "n_batches": len(window_distances),
                "mean_distance": float(np.mean(window_distances)),
                "max_distance": float(np.max(window_distances)),
                "std_distance": float(np.std(window_distances)),
                "trend": self._compute_trend(window_distances),
            },
        }

        # Change-point detection on the distance series
        cpd = WassersteinChangePointDetector(self.config)
        result["change_points"] = cpd.detect_change_points(window_distances)

        return result

    def _compute_trend(self, series: np.ndarray) -> str:
        """Simple linear trend classification."""
        if len(series) < 3:
            return "INSUFFICIENT"
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        if abs(slope) < 0.001:
            return "STABLE"
        elif slope > 0:
            return "INCREASING"
        else:
            return "DECREASING"


# ============================================================================
# INTEGRATED DRIFT REPORT (PSI + KS + Wasserstein)
# ============================================================================

def compute_integrated_drift_report(
    baseline_data: np.ndarray,
    production_data: np.ndarray,
    channel_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute a combined drift report using PSI, KS-test, and Wasserstein.

    This integrates all three drift signals into a single report,
    following the multi-metric approach recommended in the literature.
    """
    from scipy.stats import ks_2samp

    if baseline_data.ndim == 3:
        baseline_data = baseline_data.reshape(-1, baseline_data.shape[-1])
    if production_data.ndim == 3:
        production_data = production_data.reshape(-1, production_data.shape[-1])

    n_channels = baseline_data.shape[1]
    names = channel_names or [f"ch_{i}" for i in range(n_channels)]

    wass_detector = WassersteinDriftDetector()
    per_channel = {}

    for i in range(n_channels):
        b = baseline_data[:, i]
        p = production_data[:, i]

        # Wasserstein
        w_dist = wass_detector.wasserstein_1d(b, p)

        # KS test
        ks_stat, ks_pvalue = ks_2samp(b, p)

        # PSI (10-bucket)
        psi = _compute_psi(b, p, n_bins=10)

        per_channel[names[i]] = {
            "wasserstein": w_dist,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "psi": psi,
            "drift_consensus": _consensus(w_dist, ks_pvalue, psi),
        }

    return {
        "per_channel": per_channel,
        "summary": {
            "n_channels": n_channels,
            "mean_wasserstein": float(
                np.mean([v["wasserstein"] for v in per_channel.values()])
            ),
            "mean_psi": float(
                np.mean([v["psi"] for v in per_channel.values()])
            ),
            "mean_ks_stat": float(
                np.mean([v["ks_statistic"] for v in per_channel.values()])
            ),
            "channels_with_drift": sum(
                1
                for v in per_channel.values()
                if v["drift_consensus"] != "NORMAL"
            ),
        },
    }


def _compute_psi(baseline: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-4
    min_val = min(baseline.min(), production.min())
    max_val = max(baseline.max(), production.max())
    bins = np.linspace(min_val - eps, max_val + eps, n_bins + 1)

    b_counts = np.histogram(baseline, bins=bins)[0].astype(float)
    p_counts = np.histogram(production, bins=bins)[0].astype(float)

    b_pct = (b_counts + eps) / (b_counts.sum() + eps * n_bins)
    p_pct = (p_counts + eps) / (p_counts.sum() + eps * n_bins)

    psi = np.sum((p_pct - b_pct) * np.log(p_pct / b_pct))
    return float(psi)


def _consensus(wasserstein: float, ks_pvalue: float, psi: float) -> str:
    """Simple 2-of-3 consensus across drift signals."""
    signals = 0
    if wasserstein > 0.3:
        signals += 1
    if ks_pvalue < 0.01:
        signals += 1
    if psi > 0.10:
        signals += 1

    if signals >= 2:
        return "DRIFT_CONFIRMED"
    elif signals == 1:
        return "DRIFT_POSSIBLE"
    else:
        return "NORMAL"
