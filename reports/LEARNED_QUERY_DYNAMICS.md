# Learned Query Dynamics

## Protocol

Each target fact appeared once. Reuse consisted of a current query cue plus shared learned recurrent transitions; the answer value was never restated. After 64 distractors, H was reset and the answer had to be reconstructed through state. The table averages episodes and three formal seeds.

## Results

| model | reuse_count | query_target_alignment | cumulative_access | cumulative_transfer | slow_retention | accuracy |
|---|---|---|---|---|---|---|
| A1_gamma_zero | 0 | -0.6809 | 0.0000 | 0.0000 | 0.0000 | 0.0208 |
| A1_gamma_zero | 1 | -0.8062 | 1.9936 | 0.0000 | 0.0000 | 0.0174 |
| A1_gamma_zero | 2 | -0.8066 | 3.9879 | 0.0000 | 0.0000 | 0.0174 |
| A1_gamma_zero | 4 | -0.8073 | 7.9785 | 0.0000 | 0.0000 | 0.0174 |
| A1_gamma_zero | 8 | -0.8084 | 15.9643 | 0.0000 | 0.0000 | 0.0174 |
| A2_uniform_transfer | 0 | -0.6794 | 0.0000 | 0.0000 | 0.7408 | 0.0243 |
| A2_uniform_transfer | 1 | -0.8061 | 1.9935 | 0.0104 | 0.8562 | 0.0243 |
| A2_uniform_transfer | 2 | -0.8065 | 3.9878 | 0.0202 | 0.8597 | 0.0243 |
| A2_uniform_transfer | 4 | -0.8072 | 7.9781 | 0.0378 | 0.8645 | 0.0243 |
| A2_uniform_transfer | 8 | -0.8083 | 15.9635 | 0.0666 | 0.8702 | 0.0243 |
| A3_equal_timescales | 0 | -0.5983 | 0.0000 | 0.0000 | 0.6940 | 0.0486 |
| A3_equal_timescales | 1 | -0.8023 | 1.9891 | 0.0886 | 0.8655 | 0.0486 |
| A3_equal_timescales | 2 | -0.8023 | 3.9782 | 0.1579 | 0.8851 | 0.0486 |
| A3_equal_timescales | 4 | -0.8023 | 7.9566 | 0.2551 | 0.8947 | 0.0486 |
| A3_equal_timescales | 8 | -0.8024 | 15.9136 | 0.3561 | 0.8974 | 0.0451 |
| A4_no_null_dynamics | 0 | -0.7752 | 0.0000 | 0.0000 | 0.7265 | 0.0451 |
| A4_no_null_dynamics | 1 | -0.8013 | 0.9979 | 0.0396 | 0.8070 | 0.0868 |
| A4_no_null_dynamics | 2 | -0.8014 | 1.9957 | 0.0734 | 0.8319 | 0.1111 |
| A4_no_null_dynamics | 4 | -0.8017 | 3.9915 | 0.1269 | 0.8548 | 0.1528 |
| A4_no_null_dynamics | 8 | -0.8022 | 7.9831 | 0.1943 | 0.8692 | 0.2292 |
| A5_random_query | 0 | -0.0193 | 0.0000 | 0.0000 | 0.0829 | 0.0417 |
| A5_random_query | 1 | -0.0193 | 1.9961 | 0.0376 | 0.1603 | 0.0347 |
| A5_random_query | 2 | -0.0193 | 3.9923 | 0.0649 | 0.1586 | 0.0347 |
| A5_random_query | 4 | -0.0193 | 7.9851 | 0.0995 | 0.1565 | 0.0382 |
| A5_random_query | 8 | -0.0193 | 15.9712 | 0.1276 | 0.1551 | 0.0382 |
| A6_frozen_H | 0 | -0.6434 | 0.0000 | 0.0000 | 0.7683 | 0.0139 |
| A6_frozen_H | 1 | -0.7938 | 0.4308 | 0.0198 | 0.8941 | 0.0139 |
| A6_frozen_H | 2 | -0.7938 | 0.8616 | 0.0373 | 0.9006 | 0.0139 |
| A6_frozen_H | 4 | -0.7938 | 1.7233 | 0.0665 | 0.9021 | 0.0139 |
| A6_frozen_H | 8 | -0.7938 | 3.4466 | 0.1077 | 0.9015 | 0.0139 |
| A7_nonconserving | 0 | -0.6664 | 0.0000 | 0.0000 | 0.6780 | 0.0694 |
| A7_nonconserving | 1 | -0.8018 | 1.9872 | 0.0913 | 0.8470 | 0.2153 |
| A7_nonconserving | 2 | -0.8004 | 3.9705 | 0.1769 | 0.8738 | 0.3090 |
| A7_nonconserving | 4 | -0.7974 | 7.9255 | 0.3322 | 0.8912 | 0.5000 |
| A7_nonconserving | 8 | -0.7914 | 15.7944 | 0.5884 | 0.8973 | 0.7326 |
| B6_full | 0 | -0.6765 | 0.0000 | 0.0000 | 0.7265 | 0.0486 |
| B6_full | 1 | -0.8056 | 1.9929 | 0.0713 | 0.8623 | 0.1528 |
| B6_full | 2 | -0.8059 | 3.9863 | 0.1238 | 0.8782 | 0.2257 |
| B6_full | 4 | -0.8063 | 7.9742 | 0.1911 | 0.8855 | 0.2882 |
| B6_full | 8 | -0.8068 | 15.9520 | 0.2496 | 0.8877 | 0.3507 |

## Finding

G7 is **FAIL**. Query vectors varied with state, but their frozen cosine alignment to the target key was worse than the random-direction control by -0.7875. Accuracy and reuse-dependent retention improved, so the failure is specifically semantic target alignment, not a fixed-query collapse. The non-conserving arm's high accuracy is not valid evidence for ET-RCM because it violates the defining no-amplification law.
