# Windowing Justification

## Mentor decision
The production pipeline uses **window_size=200 samples (4 s @ 50 Hz) with 50% overlap** as specified by the project supervisor.

## Ablation study
A lightweight Logistic Regression classifier was trained on statistical features extracted from all window combos of the full labeled dataset (385,326 rows, 11 activity classes).

Flip-rate definition used in this report:
`flip_rate(session) = (# label changes between adjacent windows in timestamp order) / (n_session_windows - 1)`.
Aggregate summary is reported as `flip_rate_median` and `flip_rate_p95` across sessions.

### Results table

| window_size | overlap | step_samples | duration_s | n_windows | n_sessions | accuracy | f1_macro | mean_confidence | mean_entropy | flip_rate_median | flip_rate_p95 | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128.0 | 0.25 | 96.0 | 2.56 | 4010.0 | 6.0 | 0.6733 | 0.6705 | 0.6157 | 1.0454 | 0.2815 | 0.3583 | 0.79 |
| 128.0 | 0.5 | 64.0 | 2.56 | 6011.0 | 6.0 | 0.6658 | 0.6607 | 0.6264 | 1.0167 | 0.2227 | 0.2895 | 1.18 |
| 200.0 | 0.25 | 150.0 | 4.0 | 2564.0 | 6.0 | 0.6764 | 0.6762 | 0.6256 | 1.0159 | 0.2916 | 0.3438 | 0.53 |
| 200.0 | 0.5 | 100.0 | 4.0 | 3845.0 | 6.0 | 0.6879 | 0.6852 | 0.6351 | 0.9853 | 0.2395 | 0.3101 | 0.8 |
| 256.0 | 0.25 | 192.0 | 5.12 | 2000.0 | 6.0 | 0.6775 | 0.67 | 0.6306 | 1.0015 | 0.304 | 0.3643 | 0.46 |
| 256.0 | 0.5 | 128.0 | 5.12 | 3002.0 | 6.0 | 0.7354 | 0.7305 | 0.6369 | 0.9809 | 0.2333 | 0.2881 | 0.75 |

### Best combo by F1-macro: window_size=256.0, overlap=0.50 -> F1=0.7305

### Chosen production config (ws=200, ov=0.50): F1=0.6852, confidence=0.635, flip_rate_median=0.239, flip_rate_p95=0.310

## Conclusion
- Best F1 in this ablation is **0.7305** at **window_size=256, overlap=0.50**.
- Production keeps **200/0.50** by mentor decision to prioritize faster update cadence (new decision every 2 seconds at 50% overlap) and lower per-decision latency.
- This is an operational tradeoff; we do not claim 200/0.50 is the F1-optimal setting.

![Ablation plot](ABLATION_WINDOWING.png)