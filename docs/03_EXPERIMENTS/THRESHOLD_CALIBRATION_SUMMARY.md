# Threshold Calibration Summary

## Methodology
- **Sessions**: 20 synthetic sessions x 100 windows/session.
- **Model outputs**: real Stage 4 model softmax probabilities from `models/pretrained/fine_tuned_model_1dcnnbilstm.keras`.
- **Clean reference**: windows sampled from `models/normalized_baseline.json` statistics (Gaussian).
- **Perturbed**: synthetic scale x2 / Gaussian noise x1.5 / axis flip.
- **Temperature scaling**: T=1.5000 applied to probabilities (from Stage 11 output if available).
- **Sweet-spot rule (monitoring thresholds)**: false_alarm_rate <= 5% on clean and trigger_rate >= 70% on perturbed sessions.
- **Flip-rate definition**: fraction of adjacent windows in timestamp order (per session) where predicted label changes; aggregated as median/p95 across sessions.

## Results

| Threshold | Default | Data-driven sweetspot | Finding |
|---|---|---|---|
| `drift_zscore_threshold` | 2.0 | 1.0 | Session-level FAR/trigger sweep from real softmax outputs. |
| `confidence_warn_threshold` | 0.60 | see THRESHOLD_CALIBRATION.csv | Based on session mean confidence from Stage 4 outputs. |
| `uncertain_pct_threshold` | 30% | see THRESHOLD_CALIBRATION.csv | Based on percent of low-confidence windows (session-level). |
| `flip_rate_threshold` | 0.25 | see THRESHOLD_CALIBRATION.csv | Clean flip median/p95=0.480/0.536; perturbed median/p95=0.444/0.507. |
| `initial_pseudo_label_threshold` | 0.95 | see THRESHOLD_CALIBRATION.csv | Reported as accept_rate_clean/accept_rate_perturbed (not FAR). |

## Notes
- Pseudo-label FAR/error-rate **requires labels**; no labeled subset was provided in this sweep.
- The pseudo-label section reports acceptance behavior only unless labels are explicitly supplied.

![Calibration plot](THRESHOLD_CALIBRATION.png)