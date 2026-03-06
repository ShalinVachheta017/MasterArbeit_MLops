"""
scripts/threshold_sweep.py
===========================
Data-driven calibration of monitoring and trigger thresholds using real
Stage 4 softmax outputs.

Methodology:
- Clean reference windows sampled from models/normalized_baseline.json stats.
- Perturbed windows generated with controlled synthetic drift/noise.
- Stage 4 pretrained HAR model predicts softmax probabilities.
- Metrics are computed per session; temporal flip-rate is computed strictly
  on adjacent windows in timestamp order.

Outputs:
  reports/THRESHOLD_CALIBRATION.csv
  reports/THRESHOLD_CALIBRATION.png
  reports/THRESHOLD_CALIBRATION_SUMMARY.md

Usage:
  python scripts/threshold_sweep.py
  python scripts/threshold_sweep.py --n-sessions 30 --windows-per-session 80
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.temporal_metrics import flip_rate_per_session, summarize_rates

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
BASELINE_JSON = ROOT / "models" / "normalized_baseline.json"
MODEL_PATH = ROOT / "models" / "pretrained" / "fine_tuned_model_1dcnnbilstm.keras"
TEMPERATURE_JSON = ROOT / "outputs" / "calibration" / "temperature.json"

N_SENSOR_CHANNELS = 6


# --------------------------------------------------------------------------- #
# Synthetic data generators
# --------------------------------------------------------------------------- #


def _load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(model_path: Path):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"TensorFlow import failed: {exc}") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    return model


def _load_temperature(path: Path) -> float:
    if not path.exists():
        return 1.0
    data = json.loads(path.read_text(encoding="utf-8"))
    t = float(data.get("temperature", 1.0))
    if t <= 0:
        return 1.0
    return t


def _apply_temperature_scaling(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling from softmax probabilities."""
    if abs(temperature - 1.0) < 1e-9:
        return probabilities

    log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled = log_probs / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def make_clean_windows(
    baseline: dict, n: int, window_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw windows from baseline Gaussian (no perturbation)."""
    means = np.array(baseline["mean"])
    stds = np.array(baseline["std"])
    noise = rng.standard_normal((n, window_size, N_SENSOR_CHANNELS))
    return noise * stds[None, None, :] + means[None, None, :]


def make_perturbed_windows(
    baseline: dict,
    n: int,
    window_size: int,
    perturbation: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw windows then apply a selected perturbation."""
    clean = make_clean_windows(baseline, n, window_size, rng)
    stds = np.array(baseline["std"])

    if perturbation == "noise":
        extra = rng.standard_normal(clean.shape) * stds[None, None, :] * 1.5
        return clean + extra
    if perturbation == "axis_flip":
        out = clean.copy()
        out[:, :, 0] = -out[:, :, 0]
        return out
    if perturbation == "time_shift":
        return np.roll(clean, window_size // 2, axis=1)
    if perturbation == "scale":
        return clean * 2.0

    raise ValueError(f"Unknown perturbation: {perturbation}")


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #


def _build_session_index(
    n_sessions: int, windows_per_session: int
) -> tuple[np.ndarray, np.ndarray]:
    session_ids = np.repeat(np.arange(n_sessions), windows_per_session)
    timestamps = np.tile(np.arange(windows_per_session), n_sessions)
    return session_ids.astype(int), timestamps.astype(float)


def _ordered_unique(vals: np.ndarray) -> list:
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _session_reduce(values: np.ndarray, session_ids: np.ndarray, reducer) -> np.ndarray:
    reduced = []
    for sid in _ordered_unique(session_ids):
        sess_vals = values[session_ids == sid]
        reduced.append(reducer(sess_vals))
    return np.array(reduced, dtype=float)


def compute_session_drift_zscore(
    windows: np.ndarray, baseline: dict, session_ids: np.ndarray
) -> np.ndarray:
    means = np.array(baseline["mean"])
    stds = np.array(baseline["std"])

    out = []
    for sid in _ordered_unique(session_ids):
        sess = windows[session_ids == sid]
        sess_mean = sess.mean(axis=(0, 1))
        z = np.abs((sess_mean - means) / (stds + 1e-8))
        out.append(float(z.max()))
    return np.array(out)


def compute_session_mean_confidence(
    probabilities: np.ndarray, session_ids: np.ndarray
) -> np.ndarray:
    max_probs = probabilities.max(axis=1)
    return _session_reduce(max_probs, session_ids, np.mean)


def compute_session_uncertain_pct(
    probabilities: np.ndarray,
    session_ids: np.ndarray,
    uncertain_window_threshold: float,
) -> np.ndarray:
    max_probs = probabilities.max(axis=1)
    uncertain_mask = (max_probs < uncertain_window_threshold).astype(float)
    return _session_reduce(uncertain_mask, session_ids, lambda x: np.mean(x) * 100.0)


def compute_session_flip_rates(
    predicted_labels: np.ndarray,
    session_ids: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    rates_map = flip_rate_per_session(predicted_labels, session_ids, timestamps)
    summary = summarize_rates(rates_map)
    ordered = np.array([rates_map[sid] for sid in _ordered_unique(session_ids)], dtype=float)
    return ordered, summary


# --------------------------------------------------------------------------- #
# Sweep helpers
# --------------------------------------------------------------------------- #


def sweep_far_tpr(
    metric_clean: np.ndarray,
    metric_perturbed: np.ndarray,
    thresholds: np.ndarray,
    *,
    metric_name: str,
    direction: str,
) -> pd.DataFrame:
    """Compute false-alarm and trigger rates across thresholds."""
    rows = []
    for t in thresholds:
        if direction == "above":
            clean_rate = float((metric_clean > t).mean())
            pert_rate = float((metric_perturbed > t).mean())
        elif direction == "below":
            clean_rate = float((metric_clean < t).mean())
            pert_rate = float((metric_perturbed < t).mean())
        else:
            raise ValueError(f"Invalid direction: {direction}")

        rows.append(
            {
                "metric": metric_name,
                "threshold": round(float(t), 4),
                "clean_rate": round(clean_rate, 4),
                "perturbed_rate": round(pert_rate, 4),
                "clean_rate_label": "false_alarm_rate",
                "perturbed_rate_label": "trigger_rate",
                "rate_kind": "far_tpr",
                "note": "",
                "requires_labels": False,
            }
        )
    return pd.DataFrame(rows)


def sweep_pseudo_label_acceptance(
    conf_clean: np.ndarray,
    conf_perturbed: np.ndarray,
    thresholds: np.ndarray,
    *,
    labels_subset: np.ndarray | None,
    pred_clean: np.ndarray,
) -> pd.DataFrame:
    """Pseudo-label threshold sweep uses accept-rate semantics unless labels exist."""
    rows = []
    has_labels = labels_subset is not None and len(labels_subset) == len(conf_clean)

    for t in thresholds:
        clean_accept = float((conf_clean >= t).mean())
        pert_accept = float((conf_perturbed >= t).mean())

        row = {
            "metric": "pseudo_label_threshold",
            "threshold": round(float(t), 4),
            "clean_rate": round(clean_accept, 4),
            "perturbed_rate": round(pert_accept, 4),
            "clean_rate_label": "accept_rate_clean",
            "perturbed_rate_label": "accept_rate_perturbed",
            "rate_kind": "accept_rate",
            "note": "",
            "requires_labels": not has_labels,
            "pseudo_label_error_rate": np.nan,
        }

        if has_labels:
            mask = conf_clean >= t
            if mask.any():
                err = float((pred_clean[mask] != labels_subset[mask]).mean())
                row["pseudo_label_error_rate"] = round(err, 4)
                row["note"] = "error_rate computed from labeled subset"
            else:
                row["note"] = "no accepted samples at this threshold"
        else:
            row["note"] = "requires labels"

        rows.append(row)

    return pd.DataFrame(rows)


def find_sweetspot(
    df: pd.DataFrame,
    *,
    clean_max: float = 0.05,
    perturbed_min: float = 0.70,
) -> float | None:
    mask = (df["clean_rate"] <= clean_max) & (df["perturbed_rate"] >= perturbed_min)
    valid = df[mask]
    if len(valid) == 0:
        return None
    return float(valid.iloc[0]["threshold"])


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _plot_metric(ax, df_metric: pd.DataFrame, title: str) -> None:
    clean_label = df_metric["clean_rate_label"].iloc[0]
    pert_label = df_metric["perturbed_rate_label"].iloc[0]
    rate_kind = df_metric["rate_kind"].iloc[0]

    ax.plot(df_metric["threshold"], df_metric["clean_rate"], "r-o", label=clean_label)
    ax.plot(df_metric["threshold"], df_metric["perturbed_rate"], "g-s", label=pert_label)

    if rate_kind == "far_tpr":
        ax.axhline(0.05, color="r", ls="--", alpha=0.5, label="FAR target 5%")
        ax.axhline(0.70, color="g", ls="--", alpha=0.5, label="Trigger target 70%")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_results(
    df_drift: pd.DataFrame,
    df_conf: pd.DataFrame,
    df_flip: pd.DataFrame,
    df_pl: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    _plot_metric(axes[0], df_drift, "drift_zscore")
    _plot_metric(axes[1], df_conf, "confidence_warn")
    _plot_metric(axes[2], df_flip, "flip_rate")
    _plot_metric(axes[3], df_pl, "pseudo_label_accept_rate")

    fig.suptitle(
        "Threshold Calibration using Stage 4 softmax outputs on synthetic sessions",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _fmt(value: float | None) -> str:
    if value is None:
        return "see THRESHOLD_CALIBRATION.csv"
    return str(value)


def write_summary(
    *,
    out_path: Path,
    n_sessions: int,
    windows_per_session: int,
    temperature: float,
    sweetspot_drift: float | None,
    sweetspot_conf: float | None,
    sweetspot_pct: float | None,
    sweetspot_flip: float | None,
    df_pl: pd.DataFrame,
    clean_flip_summary: dict[str, float],
    pert_flip_summary: dict[str, float],
) -> None:
    requires_labels = bool(df_pl["requires_labels"].any())

    lines = [
        "# Threshold Calibration Summary",
        "",
        "## Methodology",
        f"- **Sessions**: {n_sessions} synthetic sessions x {windows_per_session} windows/session.",
        "- **Model outputs**: real Stage 4 model softmax probabilities from `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`.",
        "- **Clean reference**: windows sampled from `models/normalized_baseline.json` statistics (Gaussian).",
        "- **Perturbed**: synthetic scale x2 / Gaussian noise x1.5 / axis flip.",
        f"- **Temperature scaling**: T={temperature:.4f} applied to probabilities (from Stage 11 output if available).",
        "- **Sweet-spot rule (monitoring thresholds)**: false_alarm_rate <= 5% on clean and trigger_rate >= 70% on perturbed sessions.",
        "- **Flip-rate definition**: fraction of adjacent windows in timestamp order (per session) where predicted label changes; aggregated as median/p95 across sessions.",
        "",
        "## Results",
        "",
        "| Threshold | Default | Data-driven sweetspot | Finding |",
        "|---|---|---|---|",
        f"| `drift_zscore_threshold` | 2.0 | {_fmt(sweetspot_drift)} | Session-level FAR/trigger sweep from real softmax outputs. |",
        f"| `confidence_warn_threshold` | 0.60 | {_fmt(sweetspot_conf)} | Based on session mean confidence from Stage 4 outputs. |",
        f"| `uncertain_pct_threshold` | 30% | {_fmt(sweetspot_pct)} | Based on percent of low-confidence windows (session-level). |",
        f"| `flip_rate_threshold` | 0.25 | {_fmt(sweetspot_flip)} | Clean flip median/p95={clean_flip_summary['median']:.3f}/{clean_flip_summary['p95']:.3f}; perturbed median/p95={pert_flip_summary['median']:.3f}/{pert_flip_summary['p95']:.3f}. |",
        "| `initial_pseudo_label_threshold` | 0.95 | see THRESHOLD_CALIBRATION.csv | Reported as accept_rate_clean/accept_rate_perturbed (not FAR). |",
        "",
        "## Notes",
    ]

    if requires_labels:
        lines.append(
            "- Pseudo-label FAR/error-rate **requires labels**; no labeled subset was provided in this sweep."
        )
    else:
        lines.append(
            "- Pseudo-label error_rate was computed because a labeled subset was provided."
        )

    lines.extend(
        [
            "- The pseudo-label section reports acceptance behavior only unless labels are explicitly supplied.",
            "",
            "![Calibration plot](THRESHOLD_CALIBRATION.png)",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold calibration sweep")
    parser.add_argument("--baseline-json", type=Path, default=BASELINE_JSON)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--temperature-json", type=Path, default=TEMPERATURE_JSON)
    parser.add_argument("--disable-temperature-scaling", action="store_true")
    parser.add_argument(
        "--labels-subset",
        type=Path,
        default=None,
        help="Optional .npy labels aligned to clean windows for pseudo-label error-rate",
    )
    parser.add_argument("--n-sessions", type=int, default=20)
    parser.add_argument("--windows-per-session", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=200)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.baseline_json.exists():
        print(f"ERROR: baseline JSON not found: {args.baseline_json}")
        return 1

    baseline = _load_baseline(args.baseline_json)

    n_total = int(args.n_sessions * args.windows_per_session)
    session_ids, timestamps = _build_session_index(args.n_sessions, args.windows_per_session)

    print(
        f"Generating {args.n_sessions} sessions x {args.windows_per_session} windows "
        f"({n_total} total) per condition ..."
    )

    clean = make_clean_windows(baseline, n_total, args.window_size, rng=np.random.default_rng(0))
    perturbed_noise = make_perturbed_windows(
        baseline, n_total, args.window_size, perturbation="noise", rng=np.random.default_rng(42)
    )
    perturbed_scale = make_perturbed_windows(
        baseline, n_total, args.window_size, perturbation="scale", rng=np.random.default_rng(99)
    )

    print(f"Loading model: {args.model_path}")
    model = _load_model(args.model_path)

    probs_clean = model.predict(clean, verbose=0, batch_size=128)
    probs_noise = model.predict(perturbed_noise, verbose=0, batch_size=128)
    probs_scale = model.predict(perturbed_scale, verbose=0, batch_size=128)

    temperature = 1.0
    if not args.disable_temperature_scaling:
        temperature = _load_temperature(args.temperature_json)
        probs_clean = _apply_temperature_scaling(probs_clean, temperature)
        probs_noise = _apply_temperature_scaling(probs_noise, temperature)
        probs_scale = _apply_temperature_scaling(probs_scale, temperature)

    pred_clean = np.argmax(probs_clean, axis=1)
    pred_noise = np.argmax(probs_noise, axis=1)

    # ---- Drift z-score threshold ----
    z_clean = compute_session_drift_zscore(clean, baseline, session_ids)
    z_pert = compute_session_drift_zscore(perturbed_scale, baseline, session_ids)
    drift_thresholds = np.arange(1.0, 3.75, 0.25)
    df_drift = sweep_far_tpr(
        z_clean,
        z_pert,
        drift_thresholds,
        metric_name="drift_zscore_threshold",
        direction="above",
    )
    sweetspot_drift = find_sweetspot(df_drift)

    # ---- Confidence warning threshold ----
    conf_clean = compute_session_mean_confidence(probs_clean, session_ids)
    conf_pert = compute_session_mean_confidence(probs_noise, session_ids)
    conf_thresholds = np.arange(0.40, 0.86, 0.05)
    df_conf = sweep_far_tpr(
        conf_clean,
        conf_pert,
        conf_thresholds,
        metric_name="confidence_warn_threshold",
        direction="below",
    )
    sweetspot_conf = find_sweetspot(df_conf)

    # ---- Uncertain percentage threshold ----
    pct_clean = compute_session_uncertain_pct(
        probs_clean, session_ids, uncertain_window_threshold=0.50
    )
    pct_pert = compute_session_uncertain_pct(
        probs_noise, session_ids, uncertain_window_threshold=0.50
    )
    pct_thresholds = np.arange(5, 55, 5)
    df_pct = sweep_far_tpr(
        pct_clean,
        pct_pert,
        pct_thresholds,
        metric_name="uncertain_pct_threshold",
        direction="above",
    )
    sweetspot_pct = find_sweetspot(df_pct)

    # ---- Flip-rate threshold (strict per-session adjacency in order) ----
    flip_clean, clean_flip_summary = compute_session_flip_rates(pred_clean, session_ids, timestamps)
    flip_pert, pert_flip_summary = compute_session_flip_rates(pred_noise, session_ids, timestamps)
    flip_thresholds = np.arange(0.10, 0.65, 0.05)
    df_flip = sweep_far_tpr(
        flip_clean,
        flip_pert,
        flip_thresholds,
        metric_name="flip_rate_threshold",
        direction="above",
    )
    sweetspot_flip = find_sweetspot(df_flip)

    # ---- Pseudo-label threshold (accept-rate semantics unless labels exist) ----
    labels_subset = None
    if args.labels_subset is not None and args.labels_subset.exists():
        labels_subset = np.load(args.labels_subset)
        if len(labels_subset) != n_total:
            print(
                "WARNING: labels-subset length does not match generated clean windows; "
                "ignoring labels and marking pseudo-label error-rate as requires labels."
            )
            labels_subset = None

    window_conf_clean = probs_clean.max(axis=1)
    window_conf_pert = probs_noise.max(axis=1)
    pl_thresholds = np.arange(0.50, 0.97, 0.05)
    df_pl = sweep_pseudo_label_acceptance(
        window_conf_clean,
        window_conf_pert,
        pl_thresholds,
        labels_subset=labels_subset,
        pred_clean=pred_clean,
    )

    # ---- Combine and save ----
    all_df = pd.concat([df_drift, df_conf, df_pct, df_flip, df_pl], ignore_index=True)
    csv_path = REPORTS_DIR / "THRESHOLD_CALIBRATION.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ---- Plot ----
    try:
        plot_results(df_drift, df_conf, df_flip, df_pl, REPORTS_DIR / "THRESHOLD_CALIBRATION.png")
    except Exception as exc:
        print(f"WARNING: plot failed ({exc})")

    # ---- Summary ----
    write_summary(
        out_path=REPORTS_DIR / "THRESHOLD_CALIBRATION_SUMMARY.md",
        n_sessions=args.n_sessions,
        windows_per_session=args.windows_per_session,
        temperature=temperature,
        sweetspot_drift=sweetspot_drift,
        sweetspot_conf=sweetspot_conf,
        sweetspot_pct=sweetspot_pct,
        sweetspot_flip=sweetspot_flip,
        df_pl=df_pl,
        clean_flip_summary=clean_flip_summary,
        pert_flip_summary=pert_flip_summary,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
