# Memory Arbitration — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## Distractor sweep

| model          | condition      |   distractor_count |   accuracy |   useful_emission |   false_emission |   retention |   fast_gate |
|:---------------|:---------------|-------------------:|-----------:|------------------:|-----------------:|------------:|------------:|
| B3_joint       | absent_context |                  0 |   nan      |            0      |           1      |      1      |      0.5    |
| B3_joint       | absent_context |                 32 |   nan      |            0      |           1      |      0.6783 |      0.5    |
| B3_joint       | absent_context |                128 |   nan      |            0      |           1      |      0.6357 |      0.5    |
| B3_joint       | absent_context |                512 |   nan      |            0      |           1      |      0.5721 |      0.5    |
| B3_joint       | absent_context |               2048 |   nan      |            0      |           1      |      0.4757 |      0.5    |
| B3_joint       | absent_context |               8192 |   nan      |            0      |           1      |      0.1912 |      0.5    |
| B3_joint       | target_context |                  0 |     0.5938 |            0.5938 |           0.4062 |      1      |      0.5    |
| B3_joint       | target_context |                 32 |     0.0742 |            0.0742 |           0.9258 |      0.6783 |      0.5    |
| B3_joint       | target_context |                128 |     0.0469 |            0.0469 |           0.9531 |      0.6357 |      0.5    |
| B3_joint       | target_context |                512 |     0.0234 |            0.0234 |           0.9766 |      0.5721 |      0.5    |
| B3_joint       | target_context |               2048 |     0.0352 |            0.0352 |           0.9648 |      0.4757 |      0.5    |
| B3_joint       | target_context |               8192 |     0.0273 |            0.0273 |           0.9727 |      0.1912 |      0.5    |
| B4_m_only      | absent_context |                  0 |   nan      |            0      |           0.957  |      1      |      0      |
| B4_m_only      | absent_context |                 32 |   nan      |            0      |           0.9531 |      0.9782 |      0      |
| B4_m_only      | absent_context |                128 |   nan      |            0      |           0.9531 |      0.9489 |      0      |
| B4_m_only      | absent_context |                512 |   nan      |            0      |           0.9531 |      0.9113 |      0      |
| B4_m_only      | absent_context |               2048 |   nan      |            0      |           0.9414 |      0.7783 |      0      |
| B4_m_only      | absent_context |               8192 |   nan      |            0      |           0.9414 |      0.4004 |      0      |
| B4_m_only      | target_context |                  0 |     0.8867 |            0.8594 |           0.0859 |      1      |      0      |
| B4_m_only      | target_context |                 32 |     0.8281 |            0.8008 |           0.1445 |      0.9782 |      0      |
| B4_m_only      | target_context |                128 |     0.7891 |            0.7695 |           0.1719 |      0.9489 |      0      |
| B4_m_only      | target_context |                512 |     0.7305 |            0.7148 |           0.2227 |      0.9113 |      0      |
| B4_m_only      | target_context |               2048 |     0.3984 |            0.3984 |           0.5352 |      0.7783 |      0      |
| B4_m_only      | target_context |               8192 |     0.0234 |            0.0234 |           0.9102 |      0.4004 |      0      |
| B5_f_only      | absent_context |                  0 |   nan      |            0      |           1      |      1      |      1      |
| B5_f_only      | absent_context |                 32 |   nan      |            0      |           1      |      0.6965 |      1      |
| B5_f_only      | absent_context |                128 |   nan      |            0      |           1      |      0.4614 |      1      |
| B5_f_only      | absent_context |                512 |   nan      |            0      |           1      |      0.3311 |      1      |
| B5_f_only      | absent_context |               2048 |   nan      |            0      |           1      |      0.1851 |      1      |
| B5_f_only      | absent_context |               8192 |   nan      |            0      |           1      |      0.049  |      1      |
| B5_f_only      | target_context |                  0 |     0.4062 |            0.4062 |           0.5938 |      1      |      1      |
| B5_f_only      | target_context |                 32 |     0.0469 |            0.0469 |           0.9531 |      0.6965 |      1      |
| B5_f_only      | target_context |                128 |     0.0586 |            0.0586 |           0.9414 |      0.4614 |      1      |
| B5_f_only      | target_context |                512 |     0.0352 |            0.0352 |           0.9648 |      0.3311 |      1      |
| B5_f_only      | target_context |               2048 |     0.0391 |            0.0391 |           0.9609 |      0.1851 |      1      |
| B5_f_only      | target_context |               8192 |     0.043  |            0.043  |           0.957  |      0.049  |      1      |
| B6_arbitration | absent_context |                  0 |   nan      |            0      |           1      |      1      |      0.7462 |
| B6_arbitration | absent_context |                 32 |   nan      |            0      |           1      |      0.6644 |      0.7462 |
| B6_arbitration | absent_context |                128 |   nan      |            0      |           1      |      0.4618 |      0.7462 |
| B6_arbitration | absent_context |                512 |   nan      |            0      |           1      |      0.3259 |      0.7462 |
| B6_arbitration | absent_context |               2048 |   nan      |            0      |           1      |      0.1978 |      0.7462 |
| B6_arbitration | absent_context |               8192 |   nan      |            0      |           1      |      0.078  |      0.7462 |
| B6_arbitration | target_context |                  0 |     0.3945 |            0.3945 |           0.6055 |      1      |      0.7463 |
| B6_arbitration | target_context |                 32 |     0.0742 |            0.0742 |           0.9258 |      0.6644 |      0.7463 |
| B6_arbitration | target_context |                128 |     0.0586 |            0.0586 |           0.9414 |      0.4618 |      0.7463 |
| B6_arbitration | target_context |                512 |     0.0469 |            0.0469 |           0.9531 |      0.3259 |      0.7463 |
| B6_arbitration | target_context |               2048 |     0.0273 |            0.0273 |           0.9727 |      0.1978 |      0.7463 |
| B6_arbitration | target_context |               8192 |     0.0547 |            0.0547 |           0.9453 |      0.078  |      0.7463 |

G20: **FAIL**; accuracy margin versus R0=-0.0078, useful-expression margin=-0.0078.
