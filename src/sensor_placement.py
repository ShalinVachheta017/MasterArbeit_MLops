"""
Sensor Placement Robustness Module
====================================

Handles the problem of dominant vs. non-dominant hand sensor placement
in wrist-worn HAR devices.  63% of users wear their watch on the
non-dominant hand (Case B), but training data is often collected on
the dominant hand (Case A).

This module provides:
    1. Axis mirroring augmentation during training
    2. Hand-side detection heuristic
    3. Per-hand performance tracking
    4. Augmentation pipeline integration

ABCD Hand Cases:
    A: Dominant hand, trained on dominant    (baseline)
    B: Non-dominant hand, trained on dominant (63% of users)
    C: Dominant hand, trained on non-dominant (mirror of B)
    D: Both hands available                  (sensor fusion)

References:
    - 7 sensor placement papers in the paper collection
    - "Enhancing HAR in Wrist-Worn Sensor Data Through Compensation
       Strategies for Sensor Displacement" (Wang et al., 2024)
    - THESIS_QUESTIONS_AND_ANSWERS_2026-01-30.md — ABCD hand cases

Usage:
    from src.sensor_placement import AxisMirrorAugmenter, HandDetector

    augmenter = AxisMirrorAugmenter()
    X_aug, y_aug = augmenter.augment(X_train, y_train)

    detector = HandDetector()
    hand_info = detector.detect(sensor_data)

Author: HAR MLOps Pipeline
Date: February 2026
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SensorPlacementConfig:
    """Configuration for sensor placement robustness."""

    # Axis mirroring — which axes to flip for hand swap
    # For wrist IMU: Y and Z are typically mirrored when swapping hands
    mirror_axes: List[int] = field(default_factory=lambda: [1, 2, 4, 5])
    # Indices: 0=Ax, 1=Ay, 2=Az, 3=Gx, 4=Gy, 5=Gz
    # Ay, Az, Gy, Gz are mirrored for left↔right hand swap

    # Augmentation probability
    mirror_probability: float = 0.5

    # Hand detection
    dominant_accel_threshold: float = 1.2  # Dominant hand has ~20% higher variance

    # Sensor columns
    sensor_columns: List[str] = field(default_factory=lambda: ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"])

    # Accel-only column indices
    accel_indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    gyro_indices: List[int] = field(default_factory=lambda: [3, 4, 5])


# ============================================================================
# AXIS MIRRORING AUGMENTER
# ============================================================================


class AxisMirrorAugmenter:
    """
    Augment training data by simulating hand-swap via axis mirroring.

    When a user wears the watch on the non-dominant hand, certain
    axes (Y, Z for both accel and gyro) are effectively negated
    compared to dominant-hand placement.

    This augmenter creates additional training samples with flipped
    axes to make the model robust to hand placement.
    """

    def __init__(self, config: SensorPlacementConfig = None):
        self.config = config or SensorPlacementConfig()

    def mirror(self, X: np.ndarray) -> np.ndarray:
        """
        Apply axis mirroring to a batch of windows.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)
            Sensor windows (time_steps × channels).

        Returns
        -------
        X_mirrored : np.ndarray, same shape
            Mirrored version of the input.
        """
        X_mirrored = X.copy()
        for axis_idx in self.config.mirror_axes:
            if axis_idx < X_mirrored.shape[2]:
                X_mirrored[:, :, axis_idx] *= -1
        return X_mirrored

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probability: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset by randomly applying axis mirroring.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, C)
        y : np.ndarray, shape (N,)
        probability : float
            Probability of mirroring each sample (default from config).

        Returns
        -------
        X_aug : np.ndarray — original + mirrored samples
        y_aug : np.ndarray — corresponding labels
        """
        prob = probability if probability is not None else self.config.mirror_probability

        # Select samples to mirror
        mask = np.random.random(len(X)) < prob
        X_to_mirror = X[mask]
        y_to_mirror = y[mask]

        if len(X_to_mirror) == 0:
            logger.info("No samples selected for mirroring.")
            return X, y

        X_mirrored = self.mirror(X_to_mirror)

        # Combine original + mirrored
        X_aug = np.concatenate([X, X_mirrored], axis=0)
        y_aug = np.concatenate([y, y_to_mirror], axis=0)

        # Shuffle
        perm = np.random.permutation(len(X_aug))
        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        logger.info(
            "Axis mirroring: %d original + %d mirrored = %d total " "(p=%.2f, axes=%s)",
            len(X),
            len(X_mirrored),
            len(X_aug),
            prob,
            self.config.mirror_axes,
        )

        return X_aug, y_aug


# ============================================================================
# HAND DETECTION HEURISTIC
# ============================================================================


class HandDetector:
    """
    Detect whether the sensor is on the dominant or non-dominant hand.

    Heuristic: dominant hand typically shows ~20% higher accelerometer
    variance (more active movement) and different axis distribution
    patterns compared to non-dominant hand.

    This is a heuristic — not ground truth. Used for:
    - Logging metadata
    - Selecting the right preprocessing path
    - Adapting inference confidence thresholds
    """

    def __init__(self, config: SensorPlacementConfig = None):
        self.config = config or SensorPlacementConfig()

    def detect(
        self,
        sensor_data: np.ndarray,
        reference_stats: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Attempt to classify whether sensor is on dominant or non-dominant hand.

        Parameters
        ----------
        sensor_data : np.ndarray, shape (N, C) or (N, T, C)
            Sensor data (windows or flat).
        reference_stats : dict, optional
            Reference statistics from training data (dominant hand).

        Returns
        -------
        dict with detection result, confidence, and stats.
        """
        if sensor_data.ndim == 3:
            sensor_data = sensor_data.reshape(-1, sensor_data.shape[-1])

        accel_idx = self.config.accel_indices
        gyro_idx = self.config.gyro_indices

        accel_data = sensor_data[:, accel_idx] if max(accel_idx) < sensor_data.shape[1] else None
        gyro_data = sensor_data[:, gyro_idx] if max(gyro_idx) < sensor_data.shape[1] else None

        result = {
            "detected_hand": "UNKNOWN",
            "detection_confidence": 0.0,
            "features": {},
        }

        if accel_data is None:
            return result

        # Feature: overall accelerometer variance
        accel_var = np.var(accel_data, axis=0)
        total_var = float(np.sum(accel_var))

        # Feature: Y-axis sign bias (negative bias → likely non-dominant)
        y_mean = float(np.mean(accel_data[:, 1]))  # Ay

        # Feature: Z-axis variance ratio to X-axis
        var_ratio_zx = float(accel_var[2] / (accel_var[0] + 1e-8))

        result["features"] = {
            "accel_variance_per_axis": accel_var.tolist(),
            "total_accel_variance": total_var,
            "ay_mean": y_mean,
            "var_ratio_zx": var_ratio_zx,
        }

        # Heuristic classification
        if reference_stats:
            ref_var = reference_stats.get("total_accel_variance", total_var)
            ratio = total_var / (ref_var + 1e-8)

            if ratio > self.config.dominant_accel_threshold:
                result["detected_hand"] = "DOMINANT"
                result["detection_confidence"] = min(ratio - 1.0, 1.0)
            elif ratio < 1.0 / self.config.dominant_accel_threshold:
                result["detected_hand"] = "NON_DOMINANT"
                result["detection_confidence"] = min(1.0 - ratio, 1.0)
            else:
                result["detected_hand"] = "AMBIGUOUS"
                result["detection_confidence"] = 0.3
        else:
            # Without reference, use Y-axis heuristic
            if abs(y_mean) > 0.5:
                result["detected_hand"] = "DOMINANT" if y_mean > 0 else "NON_DOMINANT"
                result["detection_confidence"] = min(abs(y_mean), 1.0)
            else:
                result["detected_hand"] = "AMBIGUOUS"
                result["detection_confidence"] = 0.2

        logger.info(
            "Hand detection: %s (confidence=%.2f)",
            result["detected_hand"],
            result["detection_confidence"],
        )
        return result


# ============================================================================
# PER-HAND PERFORMANCE REPORTER
# ============================================================================


class HandPerformanceReporter:
    """
    Track and report model performance split by detected hand placement.

    Useful for:
    - Identifying if the model degrades on non-dominant hand data
    - Validating that axis mirroring augmentation helps
    - Thesis evaluation showing robustness across placement cases
    """

    def report(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        hand_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generate per-hand performance report.

        Parameters
        ----------
        predictions : np.ndarray, shape (N,)
        confidences : np.ndarray, shape (N,)
        hand_labels : np.ndarray, shape (N,) — "DOMINANT" or "NON_DOMINANT"
        true_labels : np.ndarray, shape (N,), optional

        Returns
        -------
        dict with per-hand metrics.
        """
        unique_hands = np.unique(hand_labels)
        report = {}

        for hand in unique_hands:
            mask = hand_labels == hand
            n = int(mask.sum())
            conf_subset = confidences[mask]

            hand_report = {
                "n_samples": n,
                "mean_confidence": float(np.mean(conf_subset)),
                "std_confidence": float(np.std(conf_subset)),
                "low_confidence_ratio": float(np.mean(conf_subset < 0.65)),
            }

            if true_labels is not None:
                preds_subset = predictions[mask]
                labels_subset = true_labels[mask]
                accuracy = float(np.mean(preds_subset == labels_subset))
                hand_report["accuracy"] = accuracy

            report[str(hand)] = hand_report

        # Compare dominant vs non-dominant
        if "DOMINANT" in report and "NON_DOMINANT" in report:
            dom = report["DOMINANT"]
            ndom = report["NON_DOMINANT"]
            report["confidence_gap"] = dom["mean_confidence"] - ndom["mean_confidence"]
            if "accuracy" in dom and "accuracy" in ndom:
                report["accuracy_gap"] = dom["accuracy"] - ndom["accuracy"]

        return report
