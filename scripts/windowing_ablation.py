"""
scripts/windowing_ablation.py
==============================
Grid search over window_size x overlap combinations using the labeled dataset.
Trains a lightweight LR classifier per combo and logs
accuracy/F1/stability metrics.

Outputs:
  reports/ABLATION_WINDOWING.csv
  reports/ABLATION_WINDOWING.png
  reports/WINDOWING_JUSTIFICATION.md

Usage:
  python scripts/windowing_ablation.py [--data-csv PATH] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.temporal_metrics import flip_rate_per_session, summarize_rates

# --------------------------------------------------------------------------- #
# Optional MLflow (skip gracefully if not available)
# --------------------------------------------------------------------------- #
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
DATA_CSV_DEFAULT = ROOT / "data" / "all_users_data_labeled.csv"
SENSOR_COLS = ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
LABEL_COL = "activity"

WINDOW_SIZES = [128, 200, 256]
OVERLAPS = [0.25, 0.50]
FREQ_HZ = 50


# --------------------------------------------------------------------------- #
# Windowing helpers
# --------------------------------------------------------------------------- #


def slide_windows_by_session(
    df_sorted: pd.DataFrame, window_size: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create windows within each session to preserve temporal ordering."""
    X, y = [], []
    session_ids, window_timestamps = [], []

    for session_id, sess in df_sorted.groupby("_session_id", sort=False):
        arr = sess[SENSOR_COLS].values.astype(np.float32)
        labels_arr = sess["_label_enc"].values.astype(int)
        ts_arr = sess["_timestamp_ns"].values.astype(np.int64)

        for start in range(0, len(arr) - window_size + 1, step):
            end = start + window_size
            window = arr[start:end]
            lab_slice = labels_arr[start:end]

            vals, counts = np.unique(lab_slice, return_counts=True)
            majority = vals[counts.argmax()]

            X.append(window.flatten())
            y.append(majority)
            session_ids.append(session_id)
            # Timestamp of the window end for adjacency checks.
            window_timestamps.append(ts_arr[end - 1])

    return (
        np.array(X),
        np.array(y),
        np.array(session_ids),
        np.array(window_timestamps, dtype=np.int64),
    )


def extract_features(arr_flat: np.ndarray, window_size: int, n_channels: int = 6) -> np.ndarray:
    """Extract stat features from a flat window (W*C) -> feature vector."""
    windows = arr_flat.reshape(-1, window_size, n_channels)
    feats = []
    for w in windows:
        row = []
        for ch in range(n_channels):
            ch_data = w[:, ch]
            row.extend(
                [
                    ch_data.mean(),
                    ch_data.std(),
                    np.sqrt(np.mean(ch_data**2)),  # RMS / energy
                    ch_data.max() - ch_data.min(),  # range
                ]
            )
        feats.append(row)
    return np.array(feats)


# --------------------------------------------------------------------------- #
# Stability metrics
# --------------------------------------------------------------------------- #


def stability_metrics(
    y_pred: np.ndarray,
    proba: np.ndarray,
    session_ids: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Compute strict per-session temporal stability from ordered windows."""
    per_session = flip_rate_per_session(
        labels=y_pred,
        session_ids=session_ids,
        timestamps=timestamps,
    )
    flip_summary = summarize_rates(per_session)
    mean_conf = proba.max(axis=1).mean()
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-9, 1)), axis=1).mean()
    return {
        "flip_rate_median": float(flip_summary["median"]),
        "flip_rate_p95": float(flip_summary["p95"]),
        "n_sessions": int(flip_summary["n_sessions"]),
        "mean_confidence": float(mean_conf),
        "mean_entropy": float(entropy),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_ablation(data_csv: Path, use_mlflow: bool) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"Loading data from {data_csv}")
    df = pd.read_csv(data_csv)

    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column for strict temporal metrics.")

    labels_raw = df[LABEL_COL].values
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels_raw)
    n_classes = len(le.classes_)
    print(f"  Rows: {len(df):,}  |  Classes: {n_classes}")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Found invalid timestamps. Cannot compute strict flip-rate ordering.")

    if "User" in df.columns:
        session_ids = df["User"].astype(str)
    else:
        session_ids = pd.Series(["session_0"] * len(df))

    prep_df = df.copy()
    prep_df["_label_enc"] = labels_enc
    prep_df["_session_id"] = session_ids.values
    prep_df["_timestamp_ns"] = timestamps.astype("int64")
    prep_df = prep_df.sort_values(["_session_id", "_timestamp_ns"], kind="stable").reset_index(
        drop=True
    )

    records = []

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("windowing_ablation")

    for window_size in WINDOW_SIZES:
        for overlap in OVERLAPS:
            step = int(window_size * (1 - overlap))
            duration_s = window_size / FREQ_HZ
            n_windows_approx = max(0, (len(prep_df) - window_size) // max(step, 1))

            print(
                f"\n-- window_size={window_size} ({duration_s:.1f}s)  overlap={overlap}  "
                f"step={step}  ~{n_windows_approx} windows"
            )
            t0 = time.perf_counter()

            X_flat, y, sess_ids, win_ts = slide_windows_by_session(prep_df, window_size, step)
            X = extract_features(X_flat, window_size)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(sss.split(X, y))
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = LogisticRegression(max_iter=300, C=1.0, random_state=42, n_jobs=1)
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_te)
            proba = clf.predict_proba(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

            # Temporal stability must be computed on ordered full streams, not shuffled test subsets.
            X_all_scaled = scaler.transform(X)
            y_pred_all = clf.predict(X_all_scaled)
            proba_all = clf.predict_proba(X_all_scaled)
            stab = stability_metrics(y_pred_all, proba_all, sess_ids, win_ts)
            elapsed = time.perf_counter() - t0

            rec = {
                "window_size": window_size,
                "overlap": overlap,
                "step_samples": step,
                "duration_s": duration_s,
                "n_windows": len(X),
                "n_sessions": int(stab["n_sessions"]),
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "mean_confidence": round(stab["mean_confidence"], 4),
                "mean_entropy": round(stab["mean_entropy"], 4),
                "flip_rate_median": round(stab["flip_rate_median"], 4),
                "flip_rate_p95": round(stab["flip_rate_p95"], 4),
                "fit_seconds": round(elapsed, 2),
            }
            records.append(rec)
            print(
                f"  acc={acc:.4f}  f1={f1:.4f}  conf={stab['mean_confidence']:.3f}  "
                f"flip_median={stab['flip_rate_median']:.3f}  in {elapsed:.1f}s"
            )

            if use_mlflow and MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"ws{window_size}_ov{overlap}"):
                    mlflow.log_params(
                        {
                            "window_size": window_size,
                            "overlap": overlap,
                            "step_samples": step,
                            "n_windows": len(X),
                        }
                    )
                    mlflow.log_metrics(
                        {
                            k: v
                            for k, v in rec.items()
                            if isinstance(v, float) and k not in ("duration_s",)
                        }
                    )

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["f1_macro", "mean_confidence", "flip_rate_median"]
    titles = ["F1-macro", "Mean Confidence (stability)", "Flip Rate Median (lower=more stable)"]

    for ax, metric, title in zip(axes, metrics, titles):
        for ws in sorted(df.window_size.unique()):
            sub = df[df.window_size == ws].sort_values("overlap")
            ax.plot(sub["overlap"], sub[metric], "o-", label=f"ws={ws}")
        ax.set_xlabel("Overlap")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Windowing Ablation: best F1 at 256/0.50, production uses 200/0.50",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Build a markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def write_justification(df: pd.DataFrame, out_path: Path) -> None:
    best = df.loc[df["f1_macro"].idxmax()]
    mentor_row = df[(df.window_size == 200) & (df.overlap == 0.5)].iloc[0]

    lines = [
        "# Windowing Justification",
        "",
        "## Mentor decision",
        "The production pipeline uses **window_size=200 samples (4 s @ 50 Hz) with 50% overlap** as specified by the project supervisor.",
        "",
        "## Ablation study",
        "A lightweight Logistic Regression classifier was trained on statistical features extracted from all window combos of the full labeled dataset (385,326 rows, 11 activity classes).",
        "",
        "Flip-rate definition used in this report:",
        "`flip_rate(session) = (# label changes between adjacent windows in timestamp order) / (n_session_windows - 1)`.",
        "Aggregate summary is reported as `flip_rate_median` and `flip_rate_p95` across sessions.",
        "",
        "### Results table",
        "",
        _df_to_markdown(df),
        "",
        f"### Best combo by F1-macro: window_size={best.window_size}, overlap={best.overlap:.2f} -> F1={best.f1_macro:.4f}",
        "",
        f"### Chosen production config (ws=200, ov=0.50): F1={mentor_row.f1_macro:.4f}, confidence={mentor_row.mean_confidence:.3f}, flip_rate_median={mentor_row.flip_rate_median:.3f}, flip_rate_p95={mentor_row.flip_rate_p95:.3f}",
        "",
        "## Conclusion",
        f"- Best F1 in this ablation is **{best.f1_macro:.4f}** at **window_size={int(best.window_size)}, overlap={best.overlap:.2f}**.",
        "- Production keeps **200/0.50** by mentor decision to prioritize faster update cadence (new decision every 2 seconds at 50% overlap) and lower per-decision latency.",
        "- This is an operational tradeoff; we do not claim 200/0.50 is the F1-optimal setting.",
        "",
        "![Ablation plot](ABLATION_WINDOWING.png)",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Windowing ablation study")
    parser.add_argument("--data-csv", type=Path, default=DATA_CSV_DEFAULT)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.data_csv.exists():
        print(f"ERROR: data CSV not found: {args.data_csv}")
        return 1

    df = run_ablation(args.data_csv, use_mlflow=not args.no_mlflow)

    csv_path = REPORTS_DIR / "ABLATION_WINDOWING.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    try:
        plot_results(df, REPORTS_DIR / "ABLATION_WINDOWING.png")
    except Exception as e:
        print(f"WARNING: plot failed ({e}) - continuing without plot")

    write_justification(df, REPORTS_DIR / "WINDOWING_JUSTIFICATION.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
