# Architectural Observability — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Frozen B5 states from four families; 64 train and 32 held-out examples per family, per seed. Train-only dual ridge fits predict future H, future logits and future events from H/HF/HM/HFM. Four future NULL ticks define the state target. No intervention label trains the probe.

## Results

Held-out future-event CE:

| Features | Mean CE | 95% seed CI | ΔCE vs H-only |
| --- | --- | --- | --- |
| H | 0.982740 | [0.909325346983791, 1.0629530311946755] | 0.000000 |
| HF | 3.309658 | [3.0689819728537078, 3.5475928010875872] | -2.326918 |
| HFM | 4.385550 | [4.077798352181178, 4.704527325030371] | -3.402810 |
| HM | 1.993896 | [1.7377783211902613, 2.225816946962108] | -1.011157 |

Future-H and future-logit MSE for every family/seed are in the Parquet records.

## Scope and limitations

Probe improvement is observational predictive sufficiency only. It cannot establish a peripheral causal pathway without A5/B2 finite swaps.
