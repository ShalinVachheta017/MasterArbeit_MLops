"""
Per-Dataset Inference Analysis
==============================
Run the production model on each raw accelerometer/gyroscope pair
individually and report confidence stats, uncertain %, and activity
distribution per dataset.  Identifies datasets the model struggles with.

Usage:
    python scripts/per_dataset_inference.py
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────
WINDOW_SIZE = 200
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
CONFIDENCE_THRESHOLD = 0.50

ACTIVITY_LABELS = [
    "hand_tapping", "ear_rubbing", "forehead_rubbing", "smoking",
    "hand_scratching", "nail_biting", "nape_rubbing", "hair_pulling",
    "knuckles_cracking", "sitting", "standing",
]


def load_model():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model_path = Path("models/pretrained/fine_tuned_model_1dcnnbilstm.keras")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    return tf.keras.models.load_model(model_path)


def load_scaler():
    config_path = Path("data/prepared/config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return np.array(config["scaler_mean"]), np.array(config["scaler_scale"])


def fuse_and_normalize(accel_file, gyro_file, scaler_mean, scaler_scale):
    """Load accel+gyro pair, fuse into 6-channel array, normalize."""
    if accel_file.endswith(".csv"):
        a = pd.read_csv(accel_file)
        g = pd.read_csv(gyro_file)
    else:
        a = pd.read_excel(accel_file)
        g = pd.read_excel(gyro_file)

    if len(a) < WINDOW_SIZE or len(g) < WINDOW_SIZE:
        return None

    # Detect column names
    accel_cols = [c for c in a.columns if any(k in c.lower() for k in ["accel_x", "accel_y", "accel_z"])]
    if not accel_cols:
        accel_cols = [c for c in a.columns if c in ["Ax", "Ay", "Az", "x", "y", "z"]]
    gyro_cols = [c for c in g.columns if any(k in c.lower() for k in ["gyro_x", "gyro_y", "gyro_z"])]
    if not gyro_cols:
        gyro_cols = [c for c in g.columns if c in ["Gx", "Gy", "Gz", "x", "y", "z"]]

    if len(accel_cols) < 3 or len(gyro_cols) < 3:
        return None

    a_vals = a[accel_cols[:3]].values.astype(float)
    g_vals = g[gyro_cols[:3]].values.astype(float)
    min_len = min(len(a_vals), len(g_vals))
    data_6ch = np.hstack([a_vals[:min_len], g_vals[:min_len]])

    # Unit conversion: milliG → m/s² if needed
    if np.abs(data_6ch[:, :3]).max() > 20:
        data_6ch[:, :3] *= 0.00981

    # Normalize
    data_norm = (data_6ch - scaler_mean) / scaler_scale
    return data_norm.astype(np.float32)


def create_windows(data):
    """Sliding window over (N, 6) data → (n_windows, 200, 6)."""
    n_samples = len(data)
    n_windows = (n_samples - WINDOW_SIZE) // STEP_SIZE + 1
    if n_windows <= 0:
        return None
    windows = np.array([
        data[i * STEP_SIZE: i * STEP_SIZE + WINDOW_SIZE]
        for i in range(n_windows)
    ])
    # Remove windows with NaN
    valid = ~np.isnan(windows).any(axis=(1, 2))
    return windows[valid] if valid.sum() > 0 else None


def main():
    print("=" * 80)
    print("PER-DATASET INFERENCE ANALYSIS")
    print("=" * 80)

    model = load_model()
    scaler_mean, scaler_scale = load_scaler()

    raw_dir = "data/raw"
    accel_files = sorted(glob.glob(os.path.join(raw_dir, "*accelerometer*")))
    print(f"Found {len(accel_files)} accelerometer files\n")

    results = []

    for accel_f in accel_files:
        base = os.path.basename(accel_f)
        pair_id = base.replace("_accelerometer.csv", "").replace("-accelerometer_data.xlsx", "")
        gyro_f = accel_f.replace("accelerometer", "gyroscope")
        if not os.path.exists(gyro_f):
            continue

        try:
            data = fuse_and_normalize(accel_f, gyro_f, scaler_mean, scaler_scale)
            if data is None:
                continue

            windows = create_windows(data)
            if windows is None:
                continue

            # Run inference
            probs = model.predict(windows, batch_size=64, verbose=0)
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            n_windows = len(windows)
            n_uncertain = int((confs < CONFIDENCE_THRESHOLD).sum())
            uncertain_pct = n_uncertain / n_windows * 100

            # Activity distribution
            unique, counts = np.unique(preds, return_counts=True)
            activity_dist = {ACTIVITY_LABELS[u]: int(c) for u, c in zip(unique, counts)}
            dominant_activity = max(activity_dist, key=activity_dist.get)
            dominant_pct = max(counts) / n_windows * 100
            n_activities = len(unique)

            results.append({
                "file": pair_id[:35],
                "n_windows": n_windows,
                "mean_conf": float(confs.mean()),
                "std_conf": float(confs.std()),
                "min_conf": float(confs.min()),
                "median_conf": float(np.median(confs)),
                "uncertain_pct": uncertain_pct,
                "n_uncertain": n_uncertain,
                "n_activities": n_activities,
                "dominant_activity": dominant_activity,
                "dominant_pct": dominant_pct,
                "activity_dist": activity_dist,
            })

        except Exception as e:
            print(f"  Skip {base}: {e}")

    # ── Print results table ───────────────────────────────────────────
    results.sort(key=lambda r: r["mean_conf"])

    print(f"\nAnalyzed {len(results)} datasets\n")
    header = (
        f"{'File':<37} {'Win':>5} {'MeanConf':>9} {'StdConf':>8} "
        f"{'Unc%':>6} {'#Act':>4} {'DominantAct':<22} {'Dom%':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        flag = " ⚠️" if r["uncertain_pct"] > 10 or r["mean_conf"] < 0.65 else ""
        print(
            f"{r['file']:<37} {r['n_windows']:>5} {r['mean_conf']:>9.4f} "
            f"{r['std_conf']:>8.4f} {r['uncertain_pct']:>5.1f}% {r['n_activities']:>4} "
            f"{r['dominant_activity']:<22} {r['dominant_pct']:>4.1f}%{flag}"
        )

    # ── Summary statistics ────────────────────────────────────────────
    confs = np.array([r["mean_conf"] for r in results])
    unc_pcts = np.array([r["uncertain_pct"] for r in results])

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS ACROSS ALL DATASETS")
    print(f"{'=' * 80}")
    print(f"  Mean confidence:      {confs.mean():.4f} ± {confs.std():.4f}")
    print(f"  Range:                [{confs.min():.4f}, {confs.max():.4f}]")
    print(f"  Datasets with conf < 0.65: {(confs < 0.65).sum()}/{len(confs)}")
    print(f"  Datasets with conf < 0.50: {(confs < 0.50).sum()}/{len(confs)}")
    print()
    print(f"  Mean uncertain %:     {unc_pcts.mean():.2f}% ± {unc_pcts.std():.2f}%")
    print(f"  Datasets with unc > 10%: {(unc_pcts > 10).sum()}/{len(unc_pcts)}")
    print(f"  Datasets with unc > 20%: {(unc_pcts > 20).sum()}/{len(unc_pcts)}")

    # ── Struggling datasets ───────────────────────────────────────────
    struggling = [r for r in results if r["mean_conf"] < 0.65 or r["uncertain_pct"] > 10]
    if struggling:
        print(f"\n{'=' * 80}")
        print(f"STRUGGLING DATASETS ({len(struggling)}/{len(results)})")
        print(f"{'=' * 80}")
        for r in struggling:
            reasons = []
            if r["mean_conf"] < 0.65:
                reasons.append(f"low conf={r['mean_conf']:.3f}")
            if r["uncertain_pct"] > 10:
                reasons.append(f"high unc={r['uncertain_pct']:.1f}%")
            print(f"  {r['file']:<35}  Reason: {', '.join(reasons)}")
            # Show its activity distribution
            for act, cnt in sorted(r["activity_dist"].items(), key=lambda x: -x[1]):
                pct = cnt / r["n_windows"] * 100
                print(f"    {act:<25} {cnt:>5} ({pct:>5.1f}%)")
    else:
        print(f"\n✅ All {len(results)} datasets have mean confidence ≥ 0.65 and uncertain % ≤ 10%")

    # ── Save results as JSON ──────────────────────────────────────────
    output_path = Path("reports/per_dataset_inference_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
