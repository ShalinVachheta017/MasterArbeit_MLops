"""
scripts/verify_prometheus_metrics.py
===================================
Verify that required HAR Prometheus metrics are available on a live /metrics endpoint.
Writes reports/PROMETHEUS_METRICS_CHECK.txt.

Usage:
  python scripts/verify_prometheus_metrics.py [--url URL] [--timeout SEC] [--offline]
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"

REQUIRED_METRICS = [
    "har_api_requests_total",
    "har_inference_latency_ms_bucket",
    "har_confidence_mean",
    "har_entropy_mean",
    "har_flip_rate",
    "har_drift_detected",
    "har_baseline_age_days",
]

DEFAULT_URL = "http://localhost:8000/metrics"


def fetch_metrics_text(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def check_metrics(text: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    lines = text.splitlines()
    for metric in REQUIRED_METRICS:
        present = any(
            line.startswith(metric)
            or line.startswith(f"# HELP {metric}")
            or line.startswith(f"# TYPE {metric}")
            for line in lines
        )
        results[metric] = present
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Prometheus metric exports")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Print expected checks without hitting the endpoint",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "PROMETHEUS_METRICS_CHECK.txt"

    lines = [
        "Prometheus Metrics Verification",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Endpoint:  {args.url}",
        f"Mode:      {'OFFLINE (dry-run)' if args.offline else 'LIVE'}",
        "",
    ]

    if args.offline:
        lines.append("OFFLINE MODE - listing expected metrics only:\n")
        for metric in REQUIRED_METRICS:
            lines.append(f"  [ ] {metric}")
        lines.append("\nRun without --offline when the inference service is running.")
        lines.append("\nRESULT: SKIPPED (offline)")
        result_code = 0
    else:
        try:
            text = fetch_metrics_text(args.url, timeout=args.timeout)
            results = check_metrics(text)
            all_pass = all(results.values())

            lines.append("Metric check results:\n")
            for metric, ok in results.items():
                icon = "[PASS]" if ok else "[FAIL]"
                lines.append(f"  {icon}  {metric}")

            lines.append("")
            if all_pass:
                lines.append("RESULT: PASS - all required metrics present")
                result_code = 0
            else:
                missing = [m for m, ok in results.items() if not ok]
                lines.append(f"RESULT: FAIL - {len(missing)} metric(s) missing: {missing}")
                result_code = 1

        except urllib.error.URLError as exc:
            lines.append(f"CONNECTION ERROR: {exc}")
            lines.append("\nRun the live stack first, then retry:")
            lines.append("  docker compose up -d inference prometheus grafana")
            lines.append("  OR: python -m src.api.app")
            lines.append("\nRESULT: FAIL - service unavailable")
            result_code = 1

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nReport written to {out_path}")
    return result_code


if __name__ == "__main__":
    sys.exit(main())
