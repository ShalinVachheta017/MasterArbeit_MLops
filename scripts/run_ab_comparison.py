#!/usr/bin/env python3
"""
=============================================================================
Normalization A/B/C Comparison  —  HAR MLOps Thesis
=============================================================================

Controlled experiment comparing three normalization strategies end-to-end.
Each variant is evaluated with stratified k-fold cross-validation so that
the same random splits are reused, making metric differences attributable
ONLY to the normalization strategy.

VARIANTS
--------
  A  zscore   GlobalStandardScaler fit on training data (current baseline)
  B  none     No amplitude normalization  (model retrained from scratch)
  C  robust   GlobalRobustScaler (median/IQR) fit on training data

REPORTED METRICS
----------------
  * Val accuracy  (mean ± std over folds)
  * F1-macro      (mean ± std)
  * F1-weighted   (mean ± std)
  * Cohen's κ     (mean ± std)
  * ECE           (Expected Calibration Error; 10 equal-width bins)

USAGE
-----
  # Run from the project root with the thesis-mlops conda env active:
  python scripts/run_ab_comparison.py

  python scripts/run_ab_comparison.py --variants A B          # skip Variant C
  python scripts/run_ab_comparison.py --n-folds 3 --epochs 30 # quick smoke test
  python scripts/run_ab_comparison.py \\
      --data-path data/raw/all_users_data_labeled.csv \\
      --output-dir reports/ab_comparison

OUTPUT
------
  reports/ab_comparison/ab_comparison_<timestamp>.json
  reports/ab_comparison/ab_comparison_<timestamp>.csv
  reports/ab_comparison/reliability_<timestamp>.png  (if matplotlib available)

=============================================================================
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configure logging before any src imports (they set up loggers at import time)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # suppress verbose train.py output during comparison
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Show only our comparison logger at INFO level
_log = logging.getLogger("ab_comparison")
_log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
)
_log.addHandler(_handler)


# ---------------------------------------------------------------------------
# Deferred heavy imports (tensorflow/keras slow to import)
# ---------------------------------------------------------------------------
def _lazy_imports():
    """Import ML stack once (called after args are parsed)."""
    global np, StratifiedKFold, f1_score, cohen_kappa_score
    global keras, StandardScaler, RobustScaler
    global TrainingConfig, DataLoader, HARModelBuilder, _make_scaler

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, cohen_kappa_score
    from sklearn.preprocessing import StandardScaler, RobustScaler

    # Suppress TF logs unless user sets TF_CPP_MIN_LOG_LEVEL
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow.keras as keras  # noqa: F401  (alias for type hints)

    from train import TrainingConfig, DataLoader, HARModelBuilder, _make_scaler  # noqa: E402


# ============================================================================
# Expected Calibration Error (ECE)
# ============================================================================


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = Σ_m (|B_m| / n) * |acc(B_m) − conf(B_m)|

    Args:
        y_true:  1-D integer array of ground-truth class indices
        probs:   2-D float array of shape (n_samples, n_classes)
        n_bins:  number of equal-width confidence bins

    Returns:
        ECE as a float in [0, 1]
    """
    confidences = probs.max(axis=1)  # predicted probability of top class
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        acc_bin = correctness[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, mean_accuracy, mean_confidence) for reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_acc[i] = correctness[in_bin].mean()
            bin_conf[i] = confidences[in_bin].mean()

    return bin_centres, bin_acc, bin_conf


# ============================================================================
# Single variant cross-validation loop
# ============================================================================


def run_cv_for_variant(
    variant: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_sensors: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    cv_random_seed: int,
    early_stopping_patience: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Train and evaluate one normalization variant using stratified k-fold CV.

    Returns:
        fold_results    : list of per-fold metric dicts
        all_y_true      : concatenated ground-truth labels
        all_probs       : concatenated softmax probabilities (for ECE)
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, cohen_kappa_score
    import tensorflow as tf
    import tensorflow.keras as keras

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_seed)
    fold_results: List[Dict[str, Any]] = []
    all_y_true: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    # Build a minimal TrainingConfig for the model builder
    cfg = TrainingConfig(
        window_size=X.shape[1],
        n_sensors=n_sensors,
        n_classes=n_classes,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        normalization_variant=variant,
        # Disable MLflow during comparison run to avoid polluting experiment
        experiment_name="ab_comparison",
    )

    _log.info(f"  [Variant {variant.upper()}]  Starting {n_folds}-fold CV …")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()
        _log.info(f"    Fold {fold}/{n_folds}  train={len(train_idx):,}  val={len(val_idx):,}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ----- Normalization (fit on TRAIN only, apply to both) ----------
        scaler = _make_scaler(variant)
        if scaler is not None:
            X_train_flat = X_train.reshape(-1, n_sensors)
            X_val_flat = X_val.reshape(-1, n_sensors)
            scaler.fit(X_train_flat)
            X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
            X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
        else:
            # Variant B: no amplitude normalization
            X_train_norm = X_train.copy()
            X_val_norm = X_val.copy()

        # ----- Build model -----------------------------------------------
        model_builder = HARModelBuilder(cfg)
        model = model_builder.create_1dcnn_bilstm()

        # Callbacks (early stopping + LR reduction; NO checkpoint for speed)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, early_stopping_patience // 3),
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        # ----- Train -------------------------------------------------------
        history = model.fit(
            X_train_norm,
            y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # ----- Evaluate ----------------------------------------------------
        val_loss, val_acc = model.evaluate(X_val_norm, y_val, verbose=0)
        probs = model.predict(X_val_norm, verbose=0)  # (n_val, n_classes)
        pred_classes = probs.argmax(axis=1)

        f1_mac = f1_score(y_val, pred_classes, average="macro")
        f1_wt = f1_score(y_val, pred_classes, average="weighted")
        kappa = cohen_kappa_score(y_val, pred_classes)
        ece = compute_ece(y_val, probs)

        elapsed = time.time() - fold_start
        _log.info(
            f"    Fold {fold} done in {elapsed:.0f}s — "
            f"acc={val_acc:.4f}  F1-mac={f1_mac:.4f}  ECE={ece:.4f}"
        )

        fold_results.append(
            {
                "fold": fold,
                "accuracy": float(val_acc),
                "f1_macro": float(f1_mac),
                "f1_weighted": float(f1_wt),
                "kappa": float(kappa),
                "ece": float(ece),
                "val_loss": float(val_loss),
                "best_epoch": int(len(history.history.get("accuracy", [0]))),
            }
        )

        all_y_true.append(y_val)
        all_probs.append(probs)

        # Free memory between folds
        del model
        keras.backend.clear_session()

    return (
        fold_results,
        np.concatenate(all_y_true),
        np.concatenate(all_probs, axis=0),
    )


# ============================================================================
# Aggregation + reporting helpers
# ============================================================================


def aggregate_folds(fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Return mean ± std for each numeric metric across folds."""
    metrics = [k for k in fold_results[0] if k != "fold"]
    agg = {}
    for m in metrics:
        vals = np.array([r[m] for r in fold_results])
        agg[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return agg


VARIANT_LABELS = {
    "zscore": "A  z-score  (StandardScaler, training stats)",
    "none": "B  none     (no amplitude normalization)",
    "robust": "C  robust   (RobustScaler, training stats)",
}

PRINT_METRICS = ["accuracy", "f1_macro", "f1_weighted", "kappa", "ece"]


def print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 20
    header_metrics = ["accuracy", "f1_macro", "f1_weighted", "kappa", "ece"]
    header = f"{'Variant':<42}" + "".join(f"{m.upper():>{col_w}}" for m in header_metrics)
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("  NORMALIZATION A/B/C COMPARISON  —  means ± std over folds")
    print("=" * len(header))
    print(header)
    print(sep)

    for variant, data in results.items():
        agg = data["aggregate"]
        label = VARIANT_LABELS.get(variant, variant)
        row = f"{label:<42}"
        for m in header_metrics:
            mu = agg[m]["mean"]
            std = agg[m]["std"]
            row += f"{mu:.4f}±{std:.4f}".rjust(col_w)
        print(row)

    print("=" * len(header))
    print()


# ============================================================================
# Reliability diagram (optional)
# ============================================================================


def save_reliability_diagram(results: Dict[str, Dict], output_path: Path) -> None:
    """Plot reliability diagrams for all variants side-by-side."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib not available — skipping reliability diagram")
        return

    variants = list(results.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 5))
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        y_true = results[variant]["all_y_true"]
        probs = results[variant]["all_probs"]
        centres, bin_acc, bin_conf = compute_reliability_data(y_true, probs)
        ece_val = results[variant]["aggregate"]["ece"]["mean"]

        ax.bar(
            centres,
            bin_acc,
            width=0.09,
            alpha=0.7,
            label="Accuracy",
            color="steelblue",
            edgecolor="black",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Variant {variant.upper()}\nECE={ece_val:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Reliability Diagrams — Normalization A/B/C Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"Reliability diagram saved: {output_path}")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run normalization A/B/C comparison for HAR MLOps thesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["A", "B", "C", "zscore", "none", "robust"],
        default=["A", "B", "C"],
        help="Variants to evaluate (default: A B C). A=zscore, B=none, C=robust.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to all_users_data_labeled.csv. Defaults to data/raw/all_users_data_labeled.csv.",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5). Use 3 for a quick smoke test.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs per fold (default: 100; early-stopping applies).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early-stopping patience (default: 15).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ab_comparison",
        help="Directory to save results (JSON, CSV, PNG).",
    )
    p.add_argument(
        "--sensor-columns",
        nargs="+",
        default=["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"],
        help="Sensor column names in the labeled CSV.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Sliding window size in samples (default: 200 @ 50 Hz = 4 s).",
    )
    p.add_argument(
        "--step-size",
        type=int,
        default=100,
        help="Step between windows (default: 100 → 50%% overlap).",
    )
    return p.parse_args()


def resolve_variants(raw: List[str]) -> List[str]:
    """Normalise variant names: 'A' → 'zscore', 'B' → 'none', 'C' → 'robust'."""
    mapping = {"A": "zscore", "B": "none", "C": "robust"}
    out, seen = [], set()
    for v in raw:
        v_norm = mapping.get(v.upper(), v.lower())
        if v_norm not in seen:
            out.append(v_norm)
            seen.add(v_norm)
    return out


def main() -> None:
    args = parse_args()

    _log.info("Loading heavy imports (TensorFlow, sklearn) …")
    _lazy_imports()

    # Import here (after lazy imports resolved globals)
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    data_path = args.data_path or (PROJECT_ROOT / "data" / "raw" / "all_users_data_labeled.csv")
    if not data_path.exists():
        _log.error(f"Training data not found: {data_path}")
        _log.error("Pass --data-path to specify the location of all_users_data_labeled.csv")
        sys.exit(1)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = resolve_variants(args.variants)
    _log.info(f"Variants to compare: {', '.join(v.upper() for v in variants)}")
    _log.info(f"Data: {data_path}")
    _log.info(f"CV folds: {args.n_folds}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")

    # -----------------------------------------------------------------------
    # Load and window training data
    # -----------------------------------------------------------------------
    _log.info("Loading training data …")
    df = pd.read_csv(data_path)
    _log.info(
        f"  {len(df):,} samples  |  users: {df['User'].nunique()}  |  "
        f"activities: {df['activity'].nunique()}"
    )

    sensor_cols = [c for c in args.sensor_columns if c in df.columns]
    if not sensor_cols:
        _log.error(
            f"None of {args.sensor_columns} found in CSV. "
            f"Available: {df.columns.tolist()}. Use --sensor-columns."
        )
        sys.exit(1)

    # Encode labels
    le = LabelEncoder()
    y_raw = le.fit_transform(df["activity"].values)
    X_raw = df[sensor_cols].values
    n_classes = len(le.classes_)
    n_sensors = len(sensor_cols)

    _log.info(f"  Classes ({n_classes}): {', '.join(le.classes_)}")
    _log.info("Creating sliding windows …")

    # Sliding window creation
    windows_X, windows_y = [], []
    for i in range(0, len(X_raw) - args.window_size + 1, args.step_size):
        w = X_raw[i : i + args.window_size]
        w_labels = y_raw[i : i + args.window_size]
        majority = Counter(w_labels).most_common(1)[0][0]
        windows_X.append(w)
        windows_y.append(majority)

    X = np.array(windows_X, dtype=np.float32)  # (N, window_size, n_sensors)
    y = np.array(windows_y, dtype=np.int32)

    _log.info(
        f"  Windows: {X.shape}  |  class distribution: " f"{dict(zip(le.classes_, np.bincount(y)))}"
    )

    # -----------------------------------------------------------------------
    # Run each variant
    # -----------------------------------------------------------------------
    experiment_results: Dict[str, Any] = {}
    run_start = time.time()

    for variant in variants:
        label = VARIANT_LABELS.get(variant, variant)
        _log.info(f"\n{'━' * 70}")
        _log.info(f"  VARIANT {label}")
        _log.info(f"{'━' * 70}")
        v_start = time.time()

        fold_results, all_y_true, all_probs = run_cv_for_variant(
            variant=variant,
            X=X,
            y=y,
            n_folds=args.n_folds,
            n_sensors=n_sensors,
            n_classes=n_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cv_random_seed=42,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Global ECE over all folds concatenated (more stable estimate)
        global_ece = compute_ece(all_y_true, all_probs)
        agg = aggregate_folds(fold_results)
        agg["ece"]["mean"] = global_ece  # override per-fold average with global estimate

        _log.info(
            f"  Variant {variant.upper()} summary — "
            f"acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}  "
            f"F1-mac={agg['f1_macro']['mean']:.4f}±{agg['f1_macro']['std']:.4f}  "
            f"ECE={global_ece:.4f} "
            f"[{time.time()-v_start:.0f}s]"
        )

        experiment_results[variant] = {
            "label": label,
            "fold_results": fold_results,
            "aggregate": agg,
            "all_y_true": all_y_true,  # numpy – kept in memory for reliability diagram
            "all_probs": all_probs,
        }

    total_elapsed = time.time() - run_start
    _log.info(f"\nTotal runtime: {total_elapsed / 60:.1f} min")

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print_comparison_table(experiment_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — serialise numpy arrays to lists
    json_path = output_dir / f"ab_comparison_{ts}.json"
    json_results: Dict[str, Any] = {}
    for v, data in experiment_results.items():
        json_results[v] = {
            "label": data["label"],
            "fold_results": data["fold_results"],
            "aggregate": data["aggregate"],
            "global_ece": float(compute_ece(data["all_y_true"], data["all_probs"])),
        }
    json_results["meta"] = {
        "timestamp": ts,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "sensor_columns": sensor_cols,
        "n_classes": int(n_classes),
        "class_names": le.classes_.tolist(),
        "n_windows": int(len(X)),
        "runtime_secs": float(total_elapsed),
    }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    _log.info(f"Results JSON: {json_path}")

    # CSV — one row per variant
    rows = []
    for v, data in experiment_results.items():
        agg = data["aggregate"]
        rows.append(
            {
                "variant": v,
                "label": data["label"],
                **{f"{m}_mean": agg[m]["mean"] for m in PRINT_METRICS},
                **{f"{m}_std": agg[m]["std"] for m in PRINT_METRICS},
            }
        )
    csv_path = output_dir / f"ab_comparison_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    _log.info(f"Results CSV:  {csv_path}")

    # Reliability diagram
    png_path = output_dir / f"reliability_{ts}.png"
    save_reliability_diagram(experiment_results, png_path)

    _log.info("\nDone.")

    # -----------------------------------------------------------------------
    # Post-run advisory: how to apply the winning variant in production
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  NEXT STEP — applying the winning variant to the production pipeline")
    print("=" * 72)
    print(
        """
After selecting the best variant, update data/prepared/config.json:

  \"normalization_variant\": \"<zscore|none|robust>\",

  For zscore  → copy scaler_mean / scaler_scale from the training run
  For robust  → add scaler_center (medians) + scaler_scale (IQR)
  For none    → no scaler keys needed; set normalization_variant=none

The production pipeline (preprocess_data.py) reads this field automatically.
Training-inference parity is enforced by reading the variant from this file.
"""
    )
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
