# Peripheral Read Mediation — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Paired F-only and M-only swaps; at every tick the corresponding *effective* read contribution is replaced by the intact branch's contribution, while the swapped stored tensor remains untouched. Absolute JS/CE are measured at 1/2/4/8; ratio only if swap JS>1e-5.

## Results

G30: **PASS**. Read mediation meets registered criterion.

| Channel | Relevant | Swap JS | Restored JS | Mediation fraction | Identifiable |
| --- | --- | --- | --- | --- | --- |
| F | PASS | 0.000361 | 0.000024 | 0.891525 | PASS |
| M | PASS | 0.000910 | 0.000000 | 1.000000 | PASS |

## Scope and limitations

Mediation fraction is descriptive, not a nonparametric causal mediation identification theorem. Other state-dependent paths and post-intervention interactions remain possible.
