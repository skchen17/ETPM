# Capacity and Distractor Scaling — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Five preregistered sparse model sizes, 8 fresh seeds each; all capacity models have equal train budgets and development LR search. The trained prediction arm clones one external stream at distractor checkpoints 0/32/128/512/2048/8192, inserts the same genuine bridge, then 4 NULL ticks. An independent FP32 continuous-vector associative-cell microbenchmark varies 1–128 stored items; it does not use learned routing.

## Results

Selected points (all intermediate N/item cells remain machine-readable):

| Variant | Distractors | Future CE | Accuracy | M-only cosine | 128-item top1 |
| --- | --- | --- | --- | --- | --- |
| B5_separate | 0 | 0.599889 | 0.921875 | 0.942901 | 0.104492 |
| B5_separate | 128 | 0.982862 | 0.765625 | 0.352671 | 0.017578 |
| B5_separate | 2048 | 1.845231 | 0.453125 | 0.147326 | 0.007812 |
| B5_separate | 8192 | 2.150381 | 0.281250 | 0.049501 | 0.009766 |
| B5_separate_h128_m32 | 0 | 0.048526 | 1.000000 | 0.960483 | 0.231445 |
| B5_separate_h128_m32 | 128 | 0.226084 | 0.937500 | 0.442053 | 0.037109 |
| B5_separate_h128_m32 | 2048 | 0.974102 | 0.765625 | 0.171046 | 0.012695 |
| B5_separate_h128_m32 | 8192 | 1.529807 | 0.578125 | 0.029600 | 0.003906 |
| B5_separate_h256_m128 | 0 | 0.055087 | 0.968750 | 0.979393 | 0.725586 |
| B5_separate_h256_m128 | 128 | 0.346323 | 0.875000 | 0.605960 | 0.683594 |
| B5_separate_h256_m128 | 2048 | 1.150443 | 0.734375 | 0.271327 | 0.089844 |
| B5_separate_h256_m128 | 8192 | 1.632436 | 0.687500 | 0.087863 | 0.005859 |
| B5_separate_h256_m64 | 0 | 0.065321 | 1.000000 | 0.931254 | 0.434570 |
| B5_separate_h256_m64 | 128 | 0.164938 | 0.968750 | 0.398988 | 0.166992 |
| B5_separate_h256_m64 | 2048 | 0.836983 | 0.750000 | 0.219044 | 0.035156 |
| B5_separate_h256_m64 | 8192 | 1.215257 | 0.687500 | 0.038237 | 0.008789 |
| B5_separate_h32_m8 | 0 | 1.691729 | 0.578125 | 0.905974 | 0.039062 |
| B5_separate_h32_m8 | 128 | 2.162691 | 0.187500 | 0.213559 | 0.007812 |
| B5_separate_h32_m8 | 2048 | 2.527513 | 0.109375 | 0.153324 | 0.002930 |
| B5_separate_h32_m8 | 8192 | 2.682962 | 0.046875 | -0.031541 | 0.008789 |

Persistent bytes and trainable parameters:

| Variant | d_H | d_M | State bytes | Parameters |
| --- | --- | --- | --- | --- |
| B5_separate | 64 | 16 | 2560 | 151877 |
| B5_separate_h128_m32 | 128 | 32 | 9216 | 584165 |
| B5_separate_h256_m128 | 256 | 128 | 133120 | 2570661 |
| B5_separate_h256_m64 | 256 | 64 | 34816 | 2290469 |
| B5_separate_h32_m8 | 32 | 8 | 768 | 40949 |

Approximate cell operations and event counts are recorded for every capacity condition; these are not wall-clock benchmarks.

## Scope and limitations

The sparse grid couples H and memory dimensions except at H=256, so their separate causal scaling effects are not identified. Synthetic cell accuracy is a mechanistic storage benchmark, not world prediction; finite dimension and interference preclude infinite-capacity claims.
