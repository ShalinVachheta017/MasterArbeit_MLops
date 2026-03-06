# Trigger Policy Evaluation

Simulated 500 monitoring sessions split into drift episodes of 8-20 hours.
Policies evaluated: single signal, 2-of-3, and cooldown sweep at 6h/12h/24h.

## Results

| policy | total_triggers | episodes_detected | n_episodes | precision | episode_recall | tradeoff_f1 | false_alarm_rate |
|---|---|---|---|---|---|---|---|
| single_signal | 251 | 16 | 16 | 0.8406 | 1.0 | 0.9134 | 0.1389 |
| two_of_three | 164 | 16 | 16 | 0.9878 | 1.0 | 0.9939 | 0.0069 |
| two_of_three_cooldown6h | 42 | 16 | 16 | 0.9524 | 1.0 | 0.9756 | 0.0069 |
| two_of_three_cooldown12h | 24 | 16 | 16 | 0.9167 | 1.0 | 0.9565 | 0.0069 |
| two_of_three_cooldown24h | 14 | 14 | 16 | 1.0 | 0.875 | 0.9333 | 0.0 |

## Cooldown selection
Chosen production cooldown: **6h** (`two_of_three_cooldown6h`).
Reason: best precision/episode-recall tradeoff (tradeoff_f1=0.976) with false_alarm_rate=0.007.

## Key findings
- Chosen cooldown policy reduces false positives by **38 events** vs single-signal.
- Episode-level recall is **100%** (16/16 drift episodes detected).
- Cooldown suppresses repeated triggers within one drift episode while preserving episode-level coverage.

## evidence_type
`EMPIRICAL_CALIBRATION` - simulated with episode-level drift structure.
Re-run with `python scripts/trigger_policy_eval.py`.

![Policy plot](TRIGGER_POLICY_EVAL.png)