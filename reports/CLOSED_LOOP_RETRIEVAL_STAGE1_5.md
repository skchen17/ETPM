# Static versus Closed-Loop Oracle — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

On the same long-gap bank, static oracle reuses the last matching historical contribution. Closed-loop oracle recomputes an H-dependent mixture of eligible historical reads after every transition; neither sees a future label. Compare with learned and zero read.

## Results

Static CE minus closed-loop CE (positive favors closed-loop):

| Ticks | Mean | 95% seed CI | Positive seeds |
| --- | --- | --- | --- |
| 1 | -0.000019 | [-4.722519467278974e-05, 6.0530360012182846e-06] | 3 |
| 2 | -0.000038 | [-9.452600109694995e-05, 1.0744369744091963e-05] | 3 |
| 4 | -0.000076 | [-0.0001880372389374642, 2.0486521922728062e-05] | 3 |
| 8 | -0.000151 | [-0.0003715559646176793, 3.860394450997599e-05] | 3 |

## Scope and limitations

This diagnostic H-dependent oracle-bank rule is not learned routing. Eligible sources may be identical, so closed-loop improvement is not guaranteed by construction.
