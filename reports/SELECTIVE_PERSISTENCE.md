# Selective Persistence

## Capacity-pressure protocol

Eight identically encoded useful facts received downstream use, followed by 32/128/512 distractor events. B2 has two persistent matrices and therefore the same matrix-state float count as F+M. B0/B1 capacity mismatches are reported but do not adjudicate G8.

| model | distractor_count | state_bytes | parameter_count | slow_retention | accuracy | selective_persistence_efficiency |
|---|---|---|---|---|---|---|
| B0_no_memory_mlp | 32 | 2048.0000 | 256802.0000 | 0.0000 | 0.0104 | 0.0000 |
| B0_no_memory_mlp | 128 | 2048.0000 | 256802.0000 | 0.0000 | 0.0104 | 0.0000 |
| B0_no_memory_mlp | 512 | 2048.0000 | 256802.0000 | 0.0000 | 0.0104 | 0.0000 |
| B1_gru | 32 | 2048.0000 | 1070497.0000 | 0.0000 | 0.0104 | 0.0000 |
| B1_gru | 128 | 2048.0000 | 1070497.0000 | 0.0000 | 0.0104 | 0.0000 |
| B1_gru | 512 | 2048.0000 | 1070497.0000 | 0.0000 | 0.0104 | 0.0000 |
| B2_single_persistent | 32 | 10240.0000 | 261922.0000 | 0.3739 | 0.2951 | 0.0000 |
| B2_single_persistent | 128 | 10240.0000 | 261922.0000 | 0.0710 | 0.0174 | 0.0000 |
| B2_single_persistent | 512 | 10240.0000 | 261922.0000 | 0.0253 | 0.0174 | 0.0000 |
| B3_uniform | 32 | 10240.0000 | 256802.0000 | 0.8002 | 0.0729 | 0.0000 |
| B3_uniform | 128 | 10240.0000 | 256802.0000 | 0.4318 | 0.0208 | 0.0000 |
| B3_uniform | 512 | 10240.0000 | 256802.0000 | 0.1676 | 0.0243 | 0.0000 |
| B5_no_idle | 32 | 10240.0000 | 256802.0000 | 0.8182 | 0.1840 | 0.0000 |
| B5_no_idle | 128 | 10240.0000 | 256802.0000 | 0.7315 | 0.0625 | 0.0000 |
| B5_no_idle | 512 | 10240.0000 | 256802.0000 | 0.6534 | 0.0590 | 0.0000 |
| B6_full | 32 | 10240.0000 | 256802.0000 | 0.8546 | 0.2292 | 0.0000 |
| B6_full | 128 | 10240.0000 | 256802.0000 | 0.8228 | 0.0938 | 0.0000 |
| B6_full | 512 | 10240.0000 | 256802.0000 | 0.7906 | 0.0764 | 0.0000 |

## Memory-dependence lesion

| model | distractor_count | slow_retention | accuracy | confidence |
|---|---|---|---|---|
| B6_full | 32 | 0.8546 | 0.2292 | 0.4774 |
| B6_full | 128 | 0.8228 | 0.0938 | 0.4494 |
| B6_full | 512 | 0.7906 | 0.0764 | 0.4731 |
| B6_full_memory_lesion | 32 | 0.8546 | 0.0104 | 0.1482 |
| B6_full_memory_lesion | 128 | 0.8228 | 0.0104 | 0.1482 |
| B6_full_memory_lesion | 512 | 0.7906 | 0.0104 | 0.1482 |

## Frequency-versus-utility protocol and results

Useful facts received 1/2 external exposures and eight real retrieval uses; unused competitors received 8/16/32 exposures without a utility marker.

| model | useful_frequency | useless_frequency | useful_retention | useless_retention | cumulative_access | accuracy |
|---|---|---|---|---|---|---|
| B2_single_persistent | 1 | 8 | 0.0748 | 0.0930 | 16.0000 | 0.0243 |
| B2_single_persistent | 1 | 16 | 0.0704 | 0.1006 | 16.0000 | 0.0486 |
| B2_single_persistent | 1 | 32 | 0.0605 | 0.0920 | 16.0000 | 0.0139 |
| B2_single_persistent | 2 | 8 | 0.0833 | 0.0811 | 16.0000 | 0.0243 |
| B2_single_persistent | 2 | 16 | 0.0966 | 0.0939 | 16.0000 | 0.0174 |
| B2_single_persistent | 2 | 32 | 0.1026 | 0.1021 | 16.0000 | 0.0382 |
| B3_uniform | 1 | 8 | 0.4594 | 0.7089 | 14.3785 | 0.0208 |
| B3_uniform | 1 | 16 | 0.4817 | 0.7809 | 14.5179 | 0.0278 |
| B3_uniform | 1 | 32 | 0.4869 | 0.8655 | 14.7207 | 0.0243 |
| B3_uniform | 2 | 8 | 0.5798 | 0.6935 | 14.1559 | 0.0208 |
| B3_uniform | 2 | 16 | 0.6103 | 0.7778 | 14.3028 | 0.0174 |
| B3_uniform | 2 | 32 | 0.6396 | 0.8693 | 14.5556 | 0.0382 |
| B6_full | 1 | 8 | 0.9240 | 0.9526 | 15.9492 | 0.3403 |
| B6_full | 1 | 16 | 0.8944 | 0.9757 | 15.9616 | 0.2500 |
| B6_full | 1 | 32 | 0.8003 | 0.9847 | 15.9712 | 0.1597 |
| B6_full | 2 | 8 | 0.9640 | 0.8730 | 15.8860 | 0.6111 |
| B6_full | 2 | 16 | 0.9439 | 0.9323 | 15.9169 | 0.5486 |
| B6_full | 2 | 32 | 0.9004 | 0.9648 | 15.9474 | 0.3611 |

## Finding

G8 is **PASS**, but G9 is **FAIL**. Thus ET-RCM beat the state-byte-matched B2 at the frozen high-pressure endpoint, yet did not prefer low-frequency useful content over the high-frequency unused competitor in the preregistered competition.
