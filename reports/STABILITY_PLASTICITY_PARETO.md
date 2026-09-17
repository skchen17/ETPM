# Stability / Plasticity Pareto

Every model was evaluated over the complete 5x5x5 grid of old exposures, new exposures and new reuse. The full episode-level grid is in `results/stage1_1/processed/stage1_1-formal-v1a1/records.parquet`; no cell is discarded. The tables below summarize every reuse level for every model, then the complete B6 old/new grid averaged across the five reuse settings.

## All models by new-reuse count

| model | new_reuses | old_probability | new_probability | unrelated_retention | accuracy |
|---|---|---|---|---|---|
| B0_no_memory_mlp | 0 | 0.0156 | 0.0155 | 0.0000 | 0.0417 |
| B0_no_memory_mlp | 1 | 0.0156 | 0.0155 | 0.0000 | 0.0417 |
| B0_no_memory_mlp | 2 | 0.0156 | 0.0155 | 0.0000 | 0.0417 |
| B0_no_memory_mlp | 4 | 0.0156 | 0.0155 | 0.0000 | 0.0417 |
| B0_no_memory_mlp | 8 | 0.0156 | 0.0155 | 0.0000 | 0.0417 |
| B1_gru | 0 | 0.0158 | 0.0148 | 0.0000 | 0.0104 |
| B1_gru | 1 | 0.0158 | 0.0148 | 0.0000 | 0.0104 |
| B1_gru | 2 | 0.0158 | 0.0148 | 0.0000 | 0.0104 |
| B1_gru | 4 | 0.0158 | 0.0148 | 0.0000 | 0.0104 |
| B1_gru | 8 | 0.0158 | 0.0148 | 0.0000 | 0.0104 |
| B2_single_persistent | 0 | 0.0206 | 0.9397 | 0.9574 | 0.9929 |
| B2_single_persistent | 1 | 0.0206 | 0.9397 | 0.9574 | 0.9929 |
| B2_single_persistent | 2 | 0.0206 | 0.9396 | 0.9574 | 0.9929 |
| B2_single_persistent | 4 | 0.0206 | 0.9395 | 0.9574 | 0.9929 |
| B2_single_persistent | 8 | 0.0206 | 0.9393 | 0.9574 | 0.9929 |
| B3_uniform | 0 | 0.0074 | 0.9386 | 0.8626 | 1.0000 |
| B3_uniform | 1 | 0.0078 | 0.9358 | 0.9103 | 1.0000 |
| B3_uniform | 2 | 0.0083 | 0.9327 | 0.9322 | 1.0000 |
| B3_uniform | 4 | 0.0095 | 0.9255 | 0.9509 | 1.0000 |
| B3_uniform | 8 | 0.0126 | 0.9055 | 0.9616 | 0.9996 |
| B5_no_idle | 0 | 0.0148 | 0.9610 | 0.3636 | 0.9983 |
| B5_no_idle | 1 | 0.0172 | 0.9573 | 0.3703 | 0.9979 |
| B5_no_idle | 2 | 0.0195 | 0.9536 | 0.3712 | 0.9979 |
| B5_no_idle | 4 | 0.0240 | 0.9471 | 0.3606 | 0.9954 |
| B5_no_idle | 8 | 0.0310 | 0.9370 | 0.3381 | 0.9900 |
| B6_full | 0 | 0.0242 | 0.9389 | 0.2963 | 0.9904 |
| B6_full | 1 | 0.0318 | 0.9275 | 0.3140 | 0.9829 |
| B6_full | 2 | 0.0387 | 0.9176 | 0.3174 | 0.9746 |
| B6_full | 4 | 0.0492 | 0.9031 | 0.3143 | 0.9629 |
| B6_full | 8 | 0.0593 | 0.8895 | 0.3090 | 0.9492 |

## Full ET-RCM old/new exposure grid (all reuse settings retained in the average)

| old_exposures | new_exposures | old_probability | new_probability | unrelated_retention | accuracy |
|---|---|---|---|---|---|
| 1 | 1 | 0.0309 | 0.8985 | 0.4743 | 1.0000 |
| 1 | 2 | 0.0026 | 0.9704 | 0.4428 | 1.0000 |
| 1 | 4 | 0.0011 | 0.9781 | 0.4075 | 1.0000 |
| 1 | 8 | 0.0009 | 0.9792 | 0.3765 | 1.0000 |
| 1 | 16 | 0.0009 | 0.9798 | 0.3490 | 1.0000 |
| 2 | 1 | 0.1020 | 0.7932 | 0.3735 | 0.9708 |
| 2 | 2 | 0.0056 | 0.9633 | 0.3603 | 1.0000 |
| 2 | 4 | 0.0017 | 0.9763 | 0.3379 | 1.0000 |
| 2 | 8 | 0.0014 | 0.9777 | 0.3148 | 1.0000 |
| 2 | 16 | 0.0015 | 0.9781 | 0.2924 | 1.0000 |
| 4 | 1 | 0.1937 | 0.6777 | 0.3141 | 0.8729 |
| 4 | 2 | 0.0097 | 0.9546 | 0.3100 | 1.0000 |
| 4 | 4 | 0.0024 | 0.9743 | 0.2962 | 1.0000 |
| 4 | 8 | 0.0020 | 0.9760 | 0.2790 | 1.0000 |
| 4 | 16 | 0.0022 | 0.9761 | 0.2596 | 1.0000 |
| 8 | 1 | 0.2612 | 0.6002 | 0.2734 | 0.7875 |
| 8 | 2 | 0.0139 | 0.9463 | 0.2770 | 1.0000 |
| 8 | 4 | 0.0033 | 0.9722 | 0.2718 | 1.0000 |
| 8 | 8 | 0.0027 | 0.9743 | 0.2609 | 1.0000 |
| 8 | 16 | 0.0029 | 0.9743 | 0.2448 | 1.0000 |
| 16 | 1 | 0.3397 | 0.5161 | 0.2483 | 0.6687 |
| 16 | 2 | 0.0208 | 0.9339 | 0.2537 | 1.0000 |
| 16 | 4 | 0.0046 | 0.9688 | 0.2536 | 1.0000 |
| 16 | 8 | 0.0037 | 0.9717 | 0.2482 | 1.0000 |
| 16 | 16 | 0.0039 | 0.9719 | 0.2353 | 1.0000 |

## Finding

G13 is **PASS**. The best preregistered balanced cell satisfying the unrelated-retention constraint reached mean P(new)=0.9793. This is a toy stability/plasticity result, not evidence of general continual learning.
