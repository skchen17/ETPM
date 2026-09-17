# Interleaved Endogenous Time

All schedule pairs use identical external events, transition counts and compute budgets. Only ordering changes.

## E — Think before versus after B

| model | schedule | matched_schedule_state_distance | slow_retention | accuracy | confidence |
|---|---|---|---|---|---|
| B0_no_memory_mlp | think_after_interruption | 9.9737 | 0.0000 | 0.0174 | 0.0191 |
| B0_no_memory_mlp | think_before_interruption | 9.9737 | 0.0000 | 0.0174 | 0.0191 |
| B1_gru | think_after_interruption | 0.2459 | 0.0000 | 0.0035 | 0.0251 |
| B1_gru | think_before_interruption | 0.2459 | 0.0000 | 0.0035 | 0.0251 |
| B2_single_persistent | think_after_interruption | 8.7951 | 1.0000 | 1.0000 | 0.9534 |
| B2_single_persistent | think_before_interruption | 8.7951 | 1.0000 | 1.0000 | 0.9534 |
| B3_uniform | think_after_interruption | 4.9990 | 1.0000 | 1.0000 | 0.9180 |
| B3_uniform | think_before_interruption | 4.9990 | 1.0000 | 1.0000 | 0.9180 |
| B5_no_idle | think_after_interruption | 0.0000 | 1.0000 | 1.0000 | 0.9755 |
| B5_no_idle | think_before_interruption | 0.0000 | 1.0000 | 1.0000 | 0.9755 |
| B6_full | think_after_interruption | 6.2574 | 1.0000 | 1.0000 | 0.9559 |
| B6_full | think_before_interruption | 6.2574 | 1.0000 | 1.0000 | 0.9575 |

## F — Consolidate before versus after interference

| model | schedule | slow_retention | accuracy | confidence |
|---|---|---|---|---|
| B0_no_memory_mlp | after_interference | 0.0000 | 0.0104 | 0.0192 |
| B0_no_memory_mlp | before_interference | 0.0000 | 0.0104 | 0.0192 |
| B1_gru | after_interference | 0.0000 | 0.0139 | 0.0251 |
| B1_gru | before_interference | 0.0000 | 0.0139 | 0.0251 |
| B2_single_persistent | after_interference | 0.0574 | 0.0174 | 0.4455 |
| B2_single_persistent | before_interference | 0.0572 | 0.0174 | 0.4460 |
| B3_uniform | after_interference | 0.3574 | 0.0278 | 0.3701 |
| B3_uniform | before_interference | 0.3901 | 0.0243 | 0.4022 |
| B5_no_idle | after_interference | 0.5006 | 0.0312 | 0.4335 |
| B5_no_idle | before_interference | 0.8084 | 0.0556 | 0.4398 |
| B6_full | after_interference | 0.4813 | 0.0660 | 0.4247 |
| B6_full | before_interference | 0.9650 | 0.0972 | 0.4576 |

## G — Reason before versus after interruption

| model | schedule | accuracy | confidence | entropy |
|---|---|---|---|---|
| B1_gru | reason_after_interruption | 0.5039 | 0.5130 | 0.6927 |
| B1_gru | reason_before_interruption | 0.5052 | 0.5137 | 0.6926 |
| B5_no_idle | reason_after_interruption | 0.5026 | 0.5039 | 0.6931 |
| B5_no_idle | reason_before_interruption | 0.5026 | 0.5039 | 0.6931 |
| B6_full | reason_after_interruption | 0.5885 | 0.5557 | 0.6777 |
| B6_full | reason_before_interruption | 0.5807 | 0.5570 | 0.6771 |

## Finding

G11 is **FAIL**; the B6 behavioral schedule margin was 0.0000. Nonzero state distance is recorded but is not treated as usefulness. N8 is therefore triggered.
