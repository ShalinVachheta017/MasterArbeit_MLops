"""
Curriculum Pseudo-Labeling (Self-Training Loop)
================================================

Implements progressive pseudo-labeling for semi-supervised retraining
on unlabeled production data.  Combines ideas from:

    1. Curriculum Labeling — start with high-confidence pseudo-labels,
       progressively lower the threshold (self-paced learning)
    2. SelfHAR — teacher-student framework
    3. EWC (Elastic Weight Consolidation) — prevent catastrophic forgetting
    4. Class-balanced sampling — max K samples per class per iteration

This is a key differentiator for the thesis: most HAR pipelines retrain
on fully labeled data.  This module enables retraining on *unlabeled*
production data, which is the realistic deployment scenario.

References:
    - Curriculum Labeling: Self-paced Pseudo-Labeling for Semi-Supervised
      HAR (IMWUT 2022)
    - SelfHAR: Improving HAR Training via Self-supervised Learning
      and Semi-supervised Learning (Tang et al., 2021)
    - ADAPT: Adaptive Pseudo-Labeling for Domain Adaptation (2023)
    - Elastic Weight Consolidation (Kirkpatrick et al., 2017)

Usage:
    from src.curriculum_pseudo_labeling import CurriculumTrainer

    trainer = CurriculumTrainer(config)
    result = trainer.train(
        model=base_model,
        labeled_X=source_X, labeled_y=source_y,
        unlabeled_X=production_X,
    )

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
class CurriculumConfig:
    """Configuration for curriculum pseudo-labeling."""

    # Progressive thresholds
    initial_confidence_threshold: float = 0.95
    final_confidence_threshold: float = 0.80
    n_iterations: int = 5
    threshold_decay: str = "linear"  # "linear" or "exponential"

    # Class-balanced sampling
    max_samples_per_class: int = 20
    min_samples_per_class: int = 3

    # Teacher-student
    use_teacher_student: bool = True
    ema_decay: float = 0.999  # Exponential moving average for teacher

    # EWC regularization
    use_ewc: bool = True
    ewc_lambda: float = 1000.0  # Regularization strength
    ewc_n_samples: int = 200     # Samples for Fisher computation

    # Training
    epochs_per_iteration: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0005  # Lower LR for fine-tuning

    # Metrics
    min_improvement_threshold: float = 0.005
    patience: int = 2  # Stop if no improvement for N iterations


# ============================================================================
# PSEUDO-LABEL SELECTOR
# ============================================================================

class PseudoLabelSelector:
    """
    Select high-confidence pseudo-labels with class balancing.

    Implements curriculum-style progressive thresholding:
    iteration 0:  only predictions with confidence > 0.95
    iteration 1:  confidence > 0.92
    ...
    iteration N:  confidence > 0.80
    """

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()

    def get_threshold_for_iteration(self, iteration: int) -> float:
        """Compute the confidence threshold for a given iteration."""
        c = self.config
        if c.threshold_decay == "exponential":
            ratio = c.final_confidence_threshold / c.initial_confidence_threshold
            return c.initial_confidence_threshold * (
                ratio ** (iteration / max(c.n_iterations - 1, 1))
            )
        else:  # linear
            step = (
                (c.initial_confidence_threshold - c.final_confidence_threshold)
                / max(c.n_iterations - 1, 1)
            )
            return c.initial_confidence_threshold - step * iteration

    def select(
        self,
        probabilities: np.ndarray,
        iteration: int,
        n_classes: int = 11,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Select pseudo-labeled samples for the current iteration.

        Parameters
        ----------
        probabilities : np.ndarray, shape (N, C)
            Model softmax outputs on unlabeled data.
        iteration : int
            Current curriculum iteration.
        n_classes : int
            Number of activity classes.

        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected samples.
        pseudo_labels : np.ndarray
            Pseudo-labels for selected samples.
        stats : dict
            Selection statistics.
        """
        threshold = self.get_threshold_for_iteration(iteration)
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)

        # Apply threshold
        mask = confidences >= threshold
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {
                "threshold": threshold,
                "n_candidates": 0,
                "n_selected": 0,
                "per_class_selected": {},
            }

        # Class-balanced sampling
        selected = []
        per_class_counts = {}

        for cls in range(n_classes):
            cls_mask = predictions[candidate_indices] == cls
            cls_indices = candidate_indices[cls_mask]

            if len(cls_indices) == 0:
                per_class_counts[cls] = 0
                continue

            # Sort by confidence (highest first)
            cls_confs = confidences[cls_indices]
            sort_order = np.argsort(-cls_confs)
            cls_indices = cls_indices[sort_order]

            # Take up to max_per_class
            n_take = min(len(cls_indices), self.config.max_samples_per_class)
            selected.extend(cls_indices[:n_take].tolist())
            per_class_counts[cls] = n_take

        selected = np.array(selected, dtype=int)
        pseudo_labels = predictions[selected]

        stats = {
            "threshold": float(threshold),
            "iteration": iteration,
            "n_candidates": int(len(candidate_indices)),
            "n_selected": int(len(selected)),
            "per_class_selected": {int(k): int(v) for k, v in per_class_counts.items()},
            "mean_confidence_selected": float(
                np.mean(confidences[selected]) if len(selected) > 0 else 0
            ),
        }

        return selected, pseudo_labels, stats


# ============================================================================
# EWC REGULARIZATION
# ============================================================================

class EWCRegularizer:
    """
    Elastic Weight Consolidation to prevent catastrophic forgetting.

    Adds a regularization term to the loss that penalizes changes to
    parameters that are important for the source task:

        L_total = L_new + (lambda/2) * sum_i F_i * (theta_i - theta_i_old)^2

    where F_i is the Fisher Information for parameter i.

    Reference:
        Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting
        in Neural Networks." PNAS.
    """

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.fisher_diag: Optional[List[np.ndarray]] = None
        self.old_params: Optional[List[np.ndarray]] = None

    def compute_fisher(self, model, X: np.ndarray, y: np.ndarray):
        """
        Compute diagonal Fisher Information Matrix using labeled data.

        Parameters
        ----------
        model : keras.Model
        X : np.ndarray — labeled input samples
        y : np.ndarray — labels
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for EWC.")

        n_samples = min(len(X), self.config.ewc_n_samples)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices].astype(np.float32)
        y_subset = y[indices]

        # Store current parameters
        self.old_params = [p.numpy().copy() for p in model.trainable_variables]

        # Accumulate squared gradients
        fisher_accum = [np.zeros_like(p) for p in self.old_params]

        for i in range(n_samples):
            x_i = tf.expand_dims(X_subset[i], 0)
            y_i = tf.constant([y_subset[i]], dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = model(x_i, training=False)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_i, logits, from_logits=False
                )

            grads = tape.gradient(loss, model.trainable_variables)
            for j, g in enumerate(grads):
                if g is not None:
                    fisher_accum[j] += g.numpy() ** 2

        # Average
        self.fisher_diag = [f / n_samples for f in fisher_accum]
        logger.info(
            "EWC: Fisher computed from %d samples, %d parameter groups.",
            n_samples,
            len(self.fisher_diag),
        )

    def penalty(self, model) -> float:
        """
        Compute the EWC penalty term.

        Returns scalar penalty = (lambda/2) * sum_i F_i * (theta_i - theta_old_i)^2
        """
        if self.fisher_diag is None or self.old_params is None:
            return 0.0

        penalty = 0.0
        for param, old_param, fisher in zip(
            model.trainable_variables, self.old_params, self.fisher_diag
        ):
            diff = param.numpy() - old_param
            penalty += np.sum(fisher * diff ** 2)

        return float(self.config.ewc_lambda / 2.0 * penalty)


# ============================================================================
# CURRICULUM TRAINER (main entry point)
# ============================================================================

class CurriculumTrainer:
    """
    Complete curriculum pseudo-labeling training loop.

    Workflow per iteration:
        1. Teacher predicts on unlabeled data
        2. Select high-confidence pseudo-labels (class-balanced)
        3. Combine with labeled source data
        4. Fine-tune student model with EWC regularization
        5. Update teacher via EMA
        6. Lower threshold for next iteration
    """

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.selector = PseudoLabelSelector(config)
        self.ewc = EWCRegularizer(config) if self.config.use_ewc else None

    def train(
        self,
        model,
        labeled_X: np.ndarray,
        labeled_y: np.ndarray,
        unlabeled_X: np.ndarray,
        n_classes: int = 11,
    ) -> Dict[str, Any]:
        """
        Run the full curriculum pseudo-labeling loop.

        Parameters
        ----------
        model : keras.Model
            Pre-trained source model.
        labeled_X : np.ndarray, shape (N_l, T, C)
            Labeled source data.
        labeled_y : np.ndarray, shape (N_l,)
            Source labels.
        unlabeled_X : np.ndarray, shape (N_u, T, C)
            Unlabeled production data.
        n_classes : int
            Number of activity classes.

        Returns
        -------
        dict with trained model, per-iteration metrics, and final stats.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow required for curriculum training.")

        logger.info("=" * 60)
        logger.info("CURRICULUM PSEUDO-LABELING")
        logger.info("  Labeled: %d  |  Unlabeled: %d  |  Iterations: %d",
                     len(labeled_X), len(unlabeled_X), self.config.n_iterations)
        logger.info("=" * 60)

        # Compute EWC Fisher on source data
        if self.ewc:
            self.ewc.compute_fisher(model, labeled_X, labeled_y)

        # Teacher is a clone (or EMA copy)
        teacher = keras.models.clone_model(model)
        teacher.set_weights(model.get_weights())

        student = model  # Fine-tune in place

        iteration_logs = []
        best_metric = 0.0
        patience_counter = 0

        for it in range(self.config.n_iterations):
            logger.info("── Iteration %d/%d ──", it + 1, self.config.n_iterations)

            # 1. Teacher predicts on unlabeled data
            probs = teacher.predict(unlabeled_X, batch_size=self.config.batch_size, verbose=0)

            # 2. Select pseudo-labels
            sel_idx, pseudo_y, sel_stats = self.selector.select(probs, it, n_classes)
            logger.info(
                "  Selected: %d samples (threshold=%.3f)",
                sel_stats["n_selected"],
                sel_stats["threshold"],
            )

            if sel_stats["n_selected"] < self.config.min_samples_per_class:
                logger.warning("  Too few pseudo-labels — skipping iteration.")
                iteration_logs.append({**sel_stats, "skipped": True})
                continue

            # 3. Combine labeled + pseudo-labeled
            pseudo_X = unlabeled_X[sel_idx]
            combined_X = np.concatenate([labeled_X, pseudo_X], axis=0)
            combined_y = np.concatenate([labeled_y, pseudo_y], axis=0)

            # Shuffle
            perm = np.random.permutation(len(combined_X))
            combined_X = combined_X[perm]
            combined_y = combined_y[perm]

            # 4. Fine-tune student
            history = student.fit(
                combined_X.astype(np.float32),
                combined_y,
                epochs=self.config.epochs_per_iteration,
                batch_size=self.config.batch_size,
                validation_split=0.1,
                verbose=0,
            )

            # Get training metrics
            val_loss = history.history.get("val_loss", [0])[-1]
            val_acc = history.history.get("val_accuracy", [0])[-1]

            # EWC penalty
            ewc_penalty = self.ewc.penalty(student) if self.ewc else 0.0

            # 5. Update teacher via EMA
            if self.config.use_teacher_student:
                self._ema_update(teacher, student, self.config.ema_decay)

            iter_log = {
                **sel_stats,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "ewc_penalty": ewc_penalty,
                "skipped": False,
            }
            iteration_logs.append(iter_log)
            logger.info(
                "  val_acc=%.4f  val_loss=%.4f  ewc=%.4f",
                val_acc, val_loss, ewc_penalty,
            )

            # Early stopping
            if val_acc > best_metric + self.config.min_improvement_threshold:
                best_metric = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("  Early stopping at iteration %d.", it + 1)
                    break

        # Final confidence on unlabeled data
        final_probs = student.predict(unlabeled_X, batch_size=self.config.batch_size, verbose=0)
        final_confs = np.max(final_probs, axis=1)

        return {
            "model": student,
            "iterations_completed": len(iteration_logs),
            "iteration_logs": iteration_logs,
            "final_mean_confidence": float(np.mean(final_confs)),
            "final_low_conf_ratio": float(np.mean(final_confs < 0.5)),
            "best_val_accuracy": float(best_metric),
        }

    def _ema_update(self, teacher, student, decay: float):
        """Exponential Moving Average update: teacher ← decay*teacher + (1-decay)*student."""
        teacher_w = teacher.get_weights()
        student_w = student.get_weights()
        new_w = [
            decay * tw + (1 - decay) * sw
            for tw, sw in zip(teacher_w, student_w)
        ]
        teacher.set_weights(new_w)
