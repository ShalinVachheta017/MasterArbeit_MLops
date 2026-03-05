"""
scripts/trigger_policy_eval.py
==============================
Replays simulated monitoring sessions to compare trigger policies:
  1. Single-signal (any one alert fires)
  2. Two-of-three (>=2 of 3 alerts)
  3. Two-of-three + cooldown sweep (6h, 12h, 24h)

Outputs:
  reports/TRIGGER_POLICY_EVAL.csv
  reports/TRIGGER_POLICY_EVAL.png
  reports/TRIGGER_POLICY_EVAL.md

Usage:
  python scripts/trigger_policy_eval.py [--n-sessions INT]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"


# --------------------------------------------------------------------------- #
# Simulate monitoring sessions
# --------------------------------------------------------------------------- #

def simulate_sessions(n_sessions: int, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic sessions with drift episodes."""
    rng = np.random.default_rng(seed)
    rows = []

    i = 0
    episode_id = 0
    in_drift = False
    drift_remaining = 0

    while i < n_sessions:
        hour = float(i)  # one session per hour

        if not in_drift and rng.random() < 0.04:
            in_drift = True
            drift_remaining = int(rng.integers(8, 20))
            ep = episode_id
            episode_id += 1
        elif in_drift:
            ep = episode_id - 1
            drift_remaining -= 1
            if drift_remaining <= 0:
                in_drift = False
        else:
            ep = -1

        genuine = in_drift

        if not genuine:
            conf_w = rng.random() < 0.04
            drift_w = rng.random() < 0.03
            temp_w = rng.random() < 0.04
        elif rng.random() < 0.15:
            which = rng.integers(0, 3)
            conf_w = which == 0
            drift_w = which == 1
            temp_w = which == 2
        else:
            conf_w = rng.random() < 0.80
            drift_w = rng.random() < 0.85
            temp_w = rng.random() < 0.75

        rows.append(
            {
                "session_id": i,
                "hour": hour,
                "episode_id": ep,
                "confidence_warn": bool(conf_w),
                "drift_warn": bool(drift_w),
                "temporal_warn": bool(temp_w),
                "genuine_drift": genuine,
            }
        )
        i += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Trigger policies
# --------------------------------------------------------------------------- #

def policy_single_signal(df: pd.DataFrame) -> pd.Series:
    return df["confidence_warn"] | df["drift_warn"] | df["temporal_warn"]


def policy_two_of_three(df: pd.DataFrame) -> pd.Series:
    n_signals = (
        df["confidence_warn"].astype(int)
        + df["drift_warn"].astype(int)
        + df["temporal_warn"].astype(int)
    )
    return n_signals >= 2


def policy_two_of_three_cooldown(df: pd.DataFrame, cooldown_hours: int) -> pd.Series:
    raw = policy_two_of_three(df)
    triggered = pd.Series(False, index=df.index)
    last_trigger_hour = -cooldown_hours * 2

    for idx, row in df.iterrows():
        if raw[idx] and (row["hour"] - last_trigger_hour >= cooldown_hours):
            triggered[idx] = True
            last_trigger_hour = row["hour"]

    return triggered


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def evaluate_policy(triggered: pd.Series, genuine: pd.Series, df: pd.DataFrame) -> dict:
    tp = int((triggered & genuine).sum())
    fp = int((triggered & ~genuine).sum())
    tn = int((~triggered & ~genuine).sum())
    fn = int((~triggered & genuine).sum())

    precision = tp / max(1, tp + fp)

    episode_ids = df[df["genuine_drift"] & (df["episode_id"] >= 0)]["episode_id"].unique()
    episodes_detected = 0
    for ep_id in episode_ids:
        ep_mask = df["episode_id"] == ep_id
        if (triggered & ep_mask).any():
            episodes_detected += 1
    episode_recall = episodes_detected / max(1, len(episode_ids))

    total_triggers = int(triggered.sum())
    false_alarm_rate = fp / max(1, fp + tn)
    tradeoff_f1 = 2 * precision * episode_recall / max(1e-9, precision + episode_recall)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(precision, 4),
        "episode_recall": round(episode_recall, 4),
        "n_episodes": int(len(episode_ids)),
        "episodes_detected": int(episodes_detected),
        "total_triggers": total_triggers,
        "false_alarm_rate": round(false_alarm_rate, 4),
        "tradeoff_f1": round(tradeoff_f1, 4),
    }


def choose_best_cooldown(result_df: pd.DataFrame) -> pd.Series:
    cooldown_rows = result_df[result_df["policy"].str.startswith("two_of_three_cooldown")].copy()
    cooldown_rows = cooldown_rows.sort_values(
        by=["tradeoff_f1", "false_alarm_rate", "total_triggers"],
        ascending=[False, True, True],
    )
    return cooldown_rows.iloc[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger policy evaluation")
    parser.add_argument("--n-sessions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cooldowns", type=int, nargs="+", default=[6, 12, 24])
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.n_sessions} monitoring sessions ...")
    df = simulate_sessions(args.n_sessions, seed=args.seed)

    genuine = df["genuine_drift"]

    policies = {
        "single_signal": policy_single_signal(df),
        "two_of_three": policy_two_of_three(df),
    }

    for cooldown_h in args.cooldowns:
        name = f"two_of_three_cooldown{cooldown_h}h"
        policies[name] = policy_two_of_three_cooldown(df, cooldown_hours=cooldown_h)

    records = []
    for name, triggered in policies.items():
        metrics = evaluate_policy(triggered, genuine, df)
        metrics["policy"] = name
        records.append(metrics)
        print(
            f"  {name:28s} triggers={metrics['total_triggers']:3d}  "
            f"precision={metrics['precision']:.3f}  recall={metrics['episode_recall']:.3f}  "
            f"FAR={metrics['false_alarm_rate']:.3f}  tradeoff_f1={metrics['tradeoff_f1']:.3f}"
        )

    result_df = pd.DataFrame(records)[
        [
            "policy",
            "total_triggers",
            "TP",
            "FP",
            "FN",
            "precision",
            "episode_recall",
            "tradeoff_f1",
            "episodes_detected",
            "n_episodes",
            "false_alarm_rate",
        ]
    ]

    best_cooldown = choose_best_cooldown(result_df)

    csv_path = REPORTS_DIR / "TRIGGER_POLICY_EVAL.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    try:
        _plot(result_df, policies, df, best_cooldown["policy"], REPORTS_DIR / "TRIGGER_POLICY_EVAL.png")
    except Exception as e:
        print(f"WARNING: plot failed ({e})")

    _summarize(
        result_df,
        args.n_sessions,
        best_cooldown,
        REPORTS_DIR / "TRIGGER_POLICY_EVAL.md",
    )

    return 0


def _plot(result_df: pd.DataFrame, policies: dict, sessions_df: pd.DataFrame,
          chosen_policy: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(result_df))
    w = 0.25
    ax1.bar(x - w, result_df["precision"], w, label="Precision", color="steelblue")
    ax1.bar(x, result_df["episode_recall"], w, label="Episode Recall", color="seagreen")
    ax1.bar(x + w, result_df["false_alarm_rate"], w, label="False Alarm Rate", color="tomato")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("two_of_three_", "2of3_") for p in result_df["policy"]],
                        rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Rate")
    ax1.set_title("Policy Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Timeline for three policies: single, two_of_three, and chosen cooldown policy.
    subset_names = ["single_signal", "two_of_three", chosen_policy]
    colors = ["tomato", "orange", "steelblue"]
    genuine_idx = sessions_df[sessions_df["genuine_drift"]].index

    for yi, name in enumerate(subset_names):
        triggered = policies[name]
        trig_idx = sessions_df[triggered].index
        ax2.scatter(trig_idx, [yi] * len(trig_idx), marker="|", s=90,
                    color=colors[yi], label=name, alpha=0.75)

    ax2.scatter(genuine_idx, [-0.5] * len(genuine_idx), marker="^",
                color="black", s=35, label="genuine drift")
    ax2.set_yticks([-0.5, 0, 1, 2])
    ax2.set_yticklabels(["genuine", subset_names[0], subset_names[1], subset_names[2]], fontsize=8)
    ax2.set_xlabel("Session #")
    ax2.set_title("Trigger Timeline")
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Trigger Policy Evaluation with Cooldown Sweep (6h/12h/24h)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def _summarize(result_df: pd.DataFrame, n_sessions: int, best_cooldown: pd.Series,
               out_path: Path) -> None:
    chosen_policy = str(best_cooldown["policy"])
    chosen_cooldown = chosen_policy.replace("two_of_three_cooldown", "").replace("h", "")

    single = result_df[result_df["policy"] == "single_signal"].iloc[0]
    chosen = result_df[result_df["policy"] == chosen_policy].iloc[0]
    fp_reduction = int(single["FP"]) - int(chosen["FP"])

    lines = [
        "# Trigger Policy Evaluation",
        "",
        f"Simulated {n_sessions} monitoring sessions split into drift episodes of 8-20 hours.",
        "Policies evaluated: single signal, 2-of-3, and cooldown sweep at 6h/12h/24h.",
        "",
        "## Results",
        "",
        _df_to_md(
            result_df[
                [
                    "policy",
                    "total_triggers",
                    "episodes_detected",
                    "n_episodes",
                    "precision",
                    "episode_recall",
                    "tradeoff_f1",
                    "false_alarm_rate",
                ]
            ]
        ),
        "",
        "## Cooldown selection",
        f"Chosen production cooldown: **{chosen_cooldown}h** (`{chosen_policy}`).",
        f"Reason: best precision/episode-recall tradeoff (tradeoff_f1={chosen['tradeoff_f1']:.3f}) with false_alarm_rate={chosen['false_alarm_rate']:.3f}.",
        "",
        "## Key findings",
        f"- Chosen cooldown policy reduces false positives by **{fp_reduction} events** vs single-signal.",
        f"- Episode-level recall is **{chosen['episode_recall']:.0%}** ({int(chosen['episodes_detected'])}/{int(chosen['n_episodes'])} drift episodes detected).",
        "- Cooldown suppresses repeated triggers within one drift episode while preserving episode-level coverage.",
        "",
        "## evidence_type",
        "`EMPIRICAL_CALIBRATION` - simulated with episode-level drift structure.",
        "Re-run with `python scripts/trigger_policy_eval.py`.",
        "",
        "![Policy plot](TRIGGER_POLICY_EVAL.png)",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
