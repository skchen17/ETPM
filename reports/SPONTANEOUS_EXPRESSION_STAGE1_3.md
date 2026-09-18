# Spontaneous Expression — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## Frozen development choices

| model                        |   learning_rate |   threshold | primary_threshold_feasible   | selection_status                                  |
|:-----------------------------|----------------:|------------:|:-----------------------------|:--------------------------------------------------|
| B0_no_persistent             |           0.001 |        0.3  | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B1_gru                       |           0.001 |        0.3  | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B2_single_persistent         |           0.001 |        0.55 | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B3_joint                     |           0.001 |        0.4  | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B4_m_only                    |           0.001 |        0.4  | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B5_f_only                    |           0.001 |        0.65 | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B6_arbitration               |           0.001 |        0.45 | False                        | diagnostic_fallback_no_primary_feasible_threshold |
| B7_nonconserving_self_replay |           0.001 |        0.45 | False                        | diagnostic_fallback_no_primary_feasible_threshold |

## Evidence accumulation

| model                        |   correct_emission_rate |   insufficient_emission_rate |   sufficient_minus_insufficient |
|:-----------------------------|------------------------:|-----------------------------:|--------------------------------:|
| B0_no_persistent             |                  0.5796 |                       0.0796 |                          0.5    |
| B1_gru                       |                  0.3887 |                       0.0674 |                          0.3213 |
| B2_single_persistent         |                  0.5791 |                       0.0688 |                          0.5103 |
| B3_joint                     |                  0.5869 |                       0.084  |                          0.5029 |
| B4_m_only                    |                  0.3857 |                       0.0815 |                          0.3042 |
| B5_f_only                    |                  0.4668 |                       0.0293 |                          0.4375 |
| B6_arbitration               |                  0.5044 |                       0.0503 |                          0.4541 |
| B7_nonconserving_self_replay |                  0.5044 |                       0.0503 |                          0.4541 |

## Formal expression PR

| model                        |   precision |   recall |     f1 |
|:-----------------------------|------------:|---------:|-------:|
| B0_no_persistent             |      0.5693 |   0.5771 | 0.5051 |
| B1_gru                       |      0.4553 |   0.3887 | 0.4008 |
| B2_single_persistent         |      0.5736 |   0.5708 | 0.5274 |
| B3_joint                     |      0.5553 |   0.5742 | 0.5065 |
| B4_m_only                    |      0.5291 |   0.3799 | 0.3447 |
| B5_f_only                    |      0.6392 |   0.4648 | 0.4679 |
| B6_arbitration               |      0.5773 |   0.5    | 0.4707 |
| B7_nonconserving_self_replay |      0.5773 |   0.5    | 0.4707 |

G18: **FAIL**. Expression score is an action/value score, not calibrated truth probability.

## Pattern discovery controls

| model            | condition          |   correct_emission |   false_emission |   accuracy |   expression_score |
|:-----------------|:-------------------|-------------------:|-----------------:|-----------:|-------------------:|
| B0_no_persistent | accidental         |             0      |           0.0524 |     0.1953 |             0.047  |
| B0_no_persistent | disappears         |             0      |           0.1183 |     0.2278 |             0.1104 |
| B0_no_persistent | distribution_shift |             0.0608 |           0.116  |     0.4157 |             0.1523 |
| B0_no_persistent | random_frequency   |             0      |           0.3523 |     0.0311 |             0.2995 |
| B0_no_persistent | reverses           |             0.0596 |           0.1158 |     0.4084 |             0.152  |
| B0_no_persistent | stable             |             0.1092 |           0.0769 |     0.4483 |             0.1593 |
| B3_joint         | accidental         |             0      |           0.0519 |     0.2088 |             0.0567 |
| B3_joint         | disappears         |             0      |           0.1414 |     0.2481 |             0.1437 |
| B3_joint         | distribution_shift |             0.0753 |           0.1213 |     0.4064 |             0.1881 |
| B3_joint         | random_frequency   |             0      |           0.383  |     0.0314 |             0.3502 |
| B3_joint         | reverses           |             0.074  |           0.1215 |     0.4022 |             0.1886 |
| B3_joint         | stable             |             0.1273 |           0.0795 |     0.4305 |             0.1969 |
| B6_arbitration   | accidental         |             0      |           0.0416 |     0.1943 |             0.0472 |
| B6_arbitration   | disappears         |             0      |           0.092  |     0.2526 |             0.1087 |
| B6_arbitration   | distribution_shift |             0.0549 |           0.085  |     0.433  |             0.1481 |
| B6_arbitration   | random_frequency   |             0      |           0.3093 |     0.0307 |             0.3048 |
| B6_arbitration   | reverses           |             0.0549 |           0.0831 |     0.4291 |             0.1469 |
| B6_arbitration   | stable             |             0.0935 |           0.0495 |     0.4922 |             0.1539 |
