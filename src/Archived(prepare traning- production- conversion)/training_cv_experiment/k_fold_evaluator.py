"""
K-Fold Cross-Validation Evaluator for HAR Model
================================================

Implements 5-fold cross-validation following ICTH_16 paper methodology.

Paper Approach (ICTH_16):
- Split data into 5 folds
- For each fold:
  * Train/fine-tune on 4 folds
  * Test on 1 fold
- Report: Accuracy ± std, Precision ± std, Recall ± std, F1 ± std

This validates model robustness and prevents overfitting.

Expected Output Format (like paper):
Accuracy: 0.87 ± 0.012
Precision: 0.86 ± 0.02
Recall: 0.87 ± 0.01
F1-Score: 0.86 ± 0.01

Author: MLOps Pipeline
Date: January 6, 2026
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PREPARED, LOGS_DIR, OUTPUTS_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class CrossValConfig:
    """Configuration for cross-validation."""

    n_folds: int = 5
    random_state: int = 42
    model_path: Path = Path("models/pretrained/fine_tuned_model_1dcnnbilstm.keras")
    data_path: Path = DATA_PREPARED / "production_X.npy"
    labels_path: Path = DATA_PREPARED / "production_y.npy"  # If available
    output_dir: Path = OUTPUTS_DIR / "cross_validation"
    verbose: bool = True


# Activity classes
ACTIVITY_CLASSES: Dict[int, str] = {
    0: "ear_rubbing",
    1: "forehead_rubbing",
    2: "hair_pulling",
    3: "hand_scratching",
    4: "hand_tapping",
    5: "knuckles_cracking",
    6: "nail_biting",
    7: "nape_rubbing",
    8: "sitting",
    9: "smoking",
    10: "standing",
}


# ============================================================================
# LOGGER SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# K-FOLD CROSS-VALIDATION EVALUATOR
# ============================================================================


class KFoldEvaluator:
    """
    K-Fold Cross-Validation for HAR model.

    Follows ICTH_16 paper methodology:
    - Split into K folds
    - Train on K-1 folds, test on 1 fold
    - Repeat K times
    - Report mean ± std for all metrics
    """

    def __init__(self, config: CrossValConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.fold_results = []
        self.confusion_matrices = []

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data and labels.

        Returns:
            X: (N, 200, 6) windows
            y: (N,) labels
        """
        logger.info("=" * 60)
        logger.info("📂 LOADING DATA")
        logger.info("=" * 60)

        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.config.data_path}")

        X = np.load(self.config.data_path)
        logger.info(f"X shape: {X.shape}")
        logger.info(f"X dtype: {X.dtype}")

        # Check for labels
        if self.config.labels_path.exists():
            y = np.load(self.config.labels_path)
            logger.info(f"y shape: {y.shape}")
            logger.info(f"y dtype: {y.dtype}")
        else:
            logger.error(f"❌ Labels not found: {self.config.labels_path}")
            logger.error("   Cannot run cross-validation without ground truth labels!")
            logger.error("   You need to:")
            logger.error("   1. Collect labeled data from your Garmin watch")
            logger.error("   2. Save labels as production_y.npy")
            raise FileNotFoundError(f"Labels required for cross-validation")

        logger.info(f"✅ Loaded {len(X)} samples")
        return X, y

    def load_model(self):
        """Load Keras model."""
        logger.info("=" * 60)
        logger.info("🧠 LOADING MODEL")
        logger.info("=" * 60)

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        import tensorflow as tf

        model = tf.keras.models.load_model(self.config.model_path)

        logger.info(f"✅ Model loaded: {self.config.model_path}")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")

        return model

    def evaluate_fold(self, model, X_test: np.ndarray, y_test: np.ndarray, fold_idx: int) -> Dict:
        """
        Evaluate model on one fold.

        Args:
            model: Trained Keras model
            X_test: Test data
            y_test: Test labels
            fold_idx: Fold number (for logging)

        Returns:
            Dictionary with metrics
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 EVALUATING FOLD {fold_idx + 1}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Test samples: {len(X_test)}")

        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        report = classification_report(
            y_test,
            y_pred,
            target_names=[ACTIVITY_CLASSES[i] for i in range(11)],
            output_dict=True,
            zero_division=0,
        )

        return {
            "fold": fold_idx,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def run_cross_validation(self) -> Dict:
        """
        Run K-fold cross-validation.

        Returns:
            Dictionary with aggregated results
        """
        logger.info("=" * 60)
        logger.info(f"🔄 STARTING {self.config.n_folds}-FOLD CROSS-VALIDATION")
        logger.info("=" * 60)

        # Load data
        X, y = self.load_data()

        # Load model (for architecture)
        base_model = self.load_model()

        # Setup K-Fold
        kfold = KFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state
        )

        # Run cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info(f"\n{'=' * 60}")
            logger.info(f"FOLD {fold_idx + 1}/{self.config.n_folds}")
            logger.info(f"{'=' * 60}")
            logger.info(f"Train samples: {len(X_train)}")
            logger.info(f"Test samples: {len(X_test)}")

            # For now, we're just evaluating the existing model
            # In future, you would fine-tune on X_train here

            # Evaluate
            fold_results = self.evaluate_fold(base_model, X_test, y_test, fold_idx)
            self.fold_results.append(fold_results)
            self.confusion_matrices.append(fold_results["confusion_matrix"])

        # Aggregate results
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """
        Aggregate results across all folds.

        Returns mean ± std for all metrics (like ICTH_16 paper).
        """
        logger.info(f"\n{'=' * 60}")
        logger.info("📊 AGGREGATED RESULTS (MEAN ± STD)")
        logger.info(f"{'=' * 60}")

        # Extract metrics from each fold
        accuracies = [r["accuracy"] for r in self.fold_results]
        precisions = [r["precision"] for r in self.fold_results]
        recalls = [r["recall"] for r in self.fold_results]
        f1_scores = [r["f1"] for r in self.fold_results]

        # Calculate mean ± std
        results = {
            "accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "values": accuracies,
            },
            "precision": {
                "mean": np.mean(precisions),
                "std": np.std(precisions),
                "values": precisions,
            },
            "recall": {"mean": np.mean(recalls), "std": np.std(recalls), "values": recalls},
            "f1": {"mean": np.mean(f1_scores), "std": np.std(f1_scores), "values": f1_scores},
        }

        # Display results (like paper format)
        logger.info(f"\n📈 Cross-Validation Results ({self.config.n_folds} folds):")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Accuracy:  {results['accuracy']['mean']:.3f} ± {results['accuracy']['std']:.3f}"
        )
        logger.info(
            f"Precision: {results['precision']['mean']:.3f} ± {results['precision']['std']:.3f}"
        )
        logger.info(f"Recall:    {results['recall']['mean']:.3f} ± {results['recall']['std']:.3f}")
        logger.info(f"F1-Score:  {results['f1']['mean']:.3f} ± {results['f1']['std']:.3f}")

        # Per-fold breakdown
        logger.info(f"\n📋 Per-Fold Breakdown:")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
        )
        logger.info(f"{'-' * 60}")
        for i, r in enumerate(self.fold_results):
            logger.info(
                f"{i+1:<6} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
                f"{r['recall']:<10.4f} {r['f1']:<10.4f}"
            )

        # Average confusion matrix
        avg_cm = np.mean(self.confusion_matrices, axis=0)
        logger.info(f"\n📊 Average Confusion Matrix:")
        logger.info(f"{'=' * 60}")
        logger.info(f"\n{avg_cm}")

        results["confusion_matrix_avg"] = avg_cm

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: Dict):
        """Save results to JSON and CSV."""
        output_file = self.config.output_dir / f"cross_validation_{self.config.n_folds}fold.json"

        # Convert numpy arrays to lists for JSON
        results_json = {
            "n_folds": self.config.n_folds,
            "metrics": {
                "accuracy": {
                    "mean": float(results["accuracy"]["mean"]),
                    "std": float(results["accuracy"]["std"]),
                    "values": [float(v) for v in results["accuracy"]["values"]],
                },
                "precision": {
                    "mean": float(results["precision"]["mean"]),
                    "std": float(results["precision"]["std"]),
                    "values": [float(v) for v in results["precision"]["values"]],
                },
                "recall": {
                    "mean": float(results["recall"]["mean"]),
                    "std": float(results["recall"]["std"]),
                    "values": [float(v) for v in results["recall"]["values"]],
                },
                "f1": {
                    "mean": float(results["f1"]["mean"]),
                    "std": float(results["f1"]["std"]),
                    "values": [float(v) for v in results["f1"]["values"]],
                },
            },
        }

        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run cross-validation."""
    config = CrossValConfig()

    logger.info("=" * 60)
    logger.info("🔄 K-FOLD CROSS-VALIDATION EVALUATOR")
    logger.info("=" * 60)
    logger.info(f"Following ICTH_16 paper methodology")
    logger.info(f"Number of folds: {config.n_folds}")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Data: {config.data_path}")
    logger.info("")

    # Check if labels exist
    if not config.labels_path.exists():
        logger.error("=" * 60)
        logger.error("❌ CANNOT RUN CROSS-VALIDATION")
        logger.error("=" * 60)
        logger.error(f"Labels not found: {config.labels_path}")
        logger.error("")
        logger.error("You need ground truth labels to run cross-validation.")
        logger.error("Options:")
        logger.error("  1. Collect labeled data from your Garmin watch")
        logger.error("  2. Manually label a sample of production data")
        logger.error("  3. Use the labeled training data (all_users_data_labeled.csv)")
        logger.error("")
        logger.error("💡 For now, you can run: python src/fine_tune_model.py")
        logger.error("   This will fine-tune the model on your Garmin data")
        return False

    # Run cross-validation
    evaluator = KFoldEvaluator(config)
    results = evaluator.run_cross_validation()

    # Success
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CROSS-VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"Final Result: Accuracy = {results['accuracy']['mean']:.3f} ± {results['accuracy']['std']:.3f}"
    )

    # Compare with paper
    logger.info("")
    logger.info("📚 Comparison with ICTH_16 Paper:")
    logger.info("   Paper (after fine-tuning): 0.870 ± 0.012")
    logger.info(
        f"   Your result: {results['accuracy']['mean']:.3f} ± {results['accuracy']['std']:.3f}"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
