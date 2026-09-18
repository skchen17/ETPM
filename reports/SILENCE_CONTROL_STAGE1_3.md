# Silence Control — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## Complete B6 development threshold sweep

|   threshold |   precision |   recall |     f1 |   noise_false_emission_rate |
|------------:|------------:|---------:|-------:|----------------------------:|
|        0.2  |      0.4678 |   0.8651 | 0.6072 |                           0 |
|        0.25 |      0.5047 |   0.8571 | 0.6353 |                           0 |
|        0.3  |      0.5309 |   0.8175 | 0.6437 |                           0 |
|        0.35 |      0.5587 |   0.7812 | 0.6515 |                           0 |
|        0.4  |      0.5655 |   0.7422 | 0.6419 |                           0 |
|        0.45 |      0.5912 |   0.7344 | 0.6551 |                           0 |
|        0.5  |      0.617  |   0.6797 | 0.6468 |                           0 |
|        0.55 |      0.621  |   0.6016 | 0.6111 |                           0 |
|        0.6  |      0.625  |   0.5469 | 0.5833 |                           0 |
|        0.65 |      0.6471 |   0.5156 | 0.5739 |                           0 |
|        0.7  |      0.6593 |   0.4688 | 0.5479 |                           0 |
|        0.75 |      0.6842 |   0.4062 | 0.5098 |                           0 |
|        0.8  |      0.6719 |   0.3359 | 0.4479 |                           0 |
|        0.85 |      0.7174 |   0.2578 | 0.3793 |                           0 |
|        0.9  |      0.6818 |   0.1172 | 0.2    |                           0 |
|        0.95 |      0.75   |   0.0469 | 0.0882 |                           0 |

The complete all-model sweep is preserved as machine-readable `threshold_sweep.parquet`.

## Formal noise streams

| model                        | condition   |   false_emission_rate |   mean_expression_score |   max_expression_score |
|:-----------------------------|:------------|----------------------:|------------------------:|-----------------------:|
| B0_no_persistent             | noise_1000  |                     0 |                  0.0002 |                 0.0006 |
| B1_gru                       | noise_1000  |                     0 |                  0.0003 |                 0.0005 |
| B2_single_persistent         | noise_1000  |                     0 |                  0.0002 |                 0.0017 |
| B3_joint                     | noise_1000  |                     0 |                  0.0001 |                 0.001  |
| B4_m_only                    | noise_1000  |                     0 |                  0.0002 |                 0.0014 |
| B5_f_only                    | noise_1000  |                     0 |                  0.0003 |                 0.0012 |
| B6_arbitration               | noise_1000  |                     0 |                  0.0002 |                 0.0011 |
| B6_arbitration               | noise_10000 |                     0 |                  0.0002 |                 0.001  |
| B7_nonconserving_self_replay | noise_1000  |                     0 |                  0.0002 |                 0.0011 |
| B7_nonconserving_self_replay | noise_10000 |                     0 |                  0.0002 |                 0.001  |

G19: **FAIL**. Development feasibility is part of the frozen decision; a diagnostic fallback cannot convert this gate to PASS.
