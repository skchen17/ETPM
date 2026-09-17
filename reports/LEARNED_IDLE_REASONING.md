# Learned Idle Reasoning

The learned shared recurrent operator was trained on graph distances 1–6. Formal distances 4–6 are in distribution; 7–8 are explicitly marked unseen-longer. No hand-written reachability transition is used.

## In-distribution

| model | internal_tick | accuracy | confidence | entropy |
|---|---|---|---|---|
| B1_gru | 0 | 0.4903 | 0.5428 | 0.6869 |
| B1_gru | 1 | 0.4667 | 0.5214 | 0.6917 |
| B1_gru | 2 | 0.4710 | 0.5179 | 0.6923 |
| B1_gru | 4 | 0.4720 | 0.5152 | 0.6926 |
| B1_gru | 8 | 0.4720 | 0.5131 | 0.6927 |
| B1_gru | 16 | 0.4720 | 0.5125 | 0.6927 |
| B5_no_idle | 0 | 0.4753 | 0.5047 | 0.6931 |
| B5_no_idle | 1 | 0.4753 | 0.5047 | 0.6931 |
| B5_no_idle | 2 | 0.4753 | 0.5047 | 0.6931 |
| B5_no_idle | 4 | 0.4753 | 0.5047 | 0.6931 |
| B5_no_idle | 8 | 0.4753 | 0.5047 | 0.6931 |
| B5_no_idle | 16 | 0.4753 | 0.5047 | 0.6931 |
| B6_full | 0 | 0.5247 | 0.5554 | 0.6789 |
| B6_full | 1 | 0.5204 | 0.5559 | 0.6789 |
| B6_full | 2 | 0.5247 | 0.5570 | 0.6780 |
| B6_full | 4 | 0.5237 | 0.5568 | 0.6776 |
| B6_full | 8 | 0.5269 | 0.5542 | 0.6788 |
| B6_full | 16 | 0.5269 | 0.5493 | 0.6813 |

## Unseen longer paths

| model | internal_tick | accuracy | confidence | entropy |
|---|---|---|---|---|
| B1_gru | 0 | 0.4967 | 0.5428 | 0.6869 |
| B1_gru | 1 | 0.5165 | 0.5212 | 0.6917 |
| B1_gru | 2 | 0.5215 | 0.5177 | 0.6923 |
| B1_gru | 4 | 0.5182 | 0.5150 | 0.6926 |
| B1_gru | 8 | 0.5182 | 0.5130 | 0.6927 |
| B1_gru | 16 | 0.5182 | 0.5125 | 0.6927 |
| B5_no_idle | 0 | 0.5017 | 0.5048 | 0.6931 |
| B5_no_idle | 1 | 0.5017 | 0.5048 | 0.6931 |
| B5_no_idle | 2 | 0.5017 | 0.5048 | 0.6931 |
| B5_no_idle | 4 | 0.5017 | 0.5048 | 0.6931 |
| B5_no_idle | 8 | 0.5017 | 0.5048 | 0.6931 |
| B5_no_idle | 16 | 0.5017 | 0.5048 | 0.6931 |
| B6_full | 0 | 0.5776 | 0.5556 | 0.6797 |
| B6_full | 1 | 0.5743 | 0.5621 | 0.6758 |
| B6_full | 2 | 0.5957 | 0.5660 | 0.6733 |
| B6_full | 4 | 0.5891 | 0.5677 | 0.6719 |
| B6_full | 8 | 0.5908 | 0.5663 | 0.6728 |
| B6_full | 16 | 0.5809 | 0.5608 | 0.6760 |

## Finding

G10 is **FAIL**. B6 K=16 minus K=0 accuracy was 0.0026, far below the frozen 0.10 margin.
