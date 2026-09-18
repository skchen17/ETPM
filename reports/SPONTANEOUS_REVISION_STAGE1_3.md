# Spontaneous Revision — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## New-evidence sweep

| model          | condition     |   new_evidence_count |   accuracy |   revision_correct |   expression_score |   confidence |
|:---------------|:--------------|---------------------:|-----------:|-------------------:|-------------------:|-------------:|
| B3_joint       | new_evidence  |                    1 |     0.6807 |             0.2627 |             0.3536 |       0.4525 |
| B3_joint       | new_evidence  |                    2 |     0.9551 |             0.6035 |             0.5651 |       0.7084 |
| B3_joint       | new_evidence  |                    3 |     0.9092 |             0.8984 |             0.944  |       0.643  |
| B3_joint       | new_evidence  |                    4 |     0.7783 |             0.7686 |             0.9582 |       0.4767 |
| B3_joint       | new_evidence  |                    5 |     0.6924 |             0.6602 |             0.9041 |       0.3404 |
| B3_joint       | new_evidence  |                    6 |     0.6748 |             0.6631 |             0.9439 |       0.3    |
| B3_joint       | new_evidence  |                    7 |     0.6504 |             0.6475 |             0.9643 |       0.2611 |
| B3_joint       | new_evidence  |                    8 |     0.6572 |             0.6543 |             0.9587 |       0.2397 |
| B3_joint       | old_supported |                    0 |     1      |             0      |             0.9814 |       0.9216 |
| B6_arbitration | new_evidence  |                    1 |     0.6221 |             0.1855 |             0.3295 |       0.4347 |
| B6_arbitration | new_evidence  |                    2 |     0.9463 |             0.5146 |             0.5229 |       0.723  |
| B6_arbitration | new_evidence  |                    3 |     0.9141 |             0.8994 |             0.9471 |       0.6747 |
| B6_arbitration | new_evidence  |                    4 |     0.7715 |             0.7588 |             0.9415 |       0.5006 |
| B6_arbitration | new_evidence  |                    5 |     0.6699 |             0.6514 |             0.9021 |       0.3393 |
| B6_arbitration | new_evidence  |                    6 |     0.6299 |             0.625  |             0.9483 |       0.2848 |
| B6_arbitration | new_evidence  |                    7 |     0.6162 |             0.6152 |             0.9625 |       0.2482 |
| B6_arbitration | new_evidence  |                    8 |     0.6348 |             0.6338 |             0.9587 |       0.2243 |
| B6_arbitration | old_supported |                    0 |     1      |             0      |             0.986  |       0.9267 |

Authorization audit `revision_healthy`: **False**. Expression is not an irreversible commitment; only genuine new external evidence uses the delta-write path.
