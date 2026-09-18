# Thought-Driven Consolidation — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## Matched-exposure usage/retention association

| model          |   usage_retention_spearman |
|:---------------|---------------------------:|
| B3_joint       |                     0.1847 |
| B6_arbitration |                     0.1445 |

G21: **FAIL**; mean per-seed Spearman=0.1445. Attribution is query-alignment weighted fast read, not an importance label.
