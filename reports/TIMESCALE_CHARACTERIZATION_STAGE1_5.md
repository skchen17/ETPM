# H/F/M Effective Timescales — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Per seed: 512 train and 256 untouched test histories, 513 real external events per history, random visible key/value writes. Train-only ridge probes decode historical visible values at lags 1–512 from H/F/M separately. Baseline is train-frequency constant CE. Ablations: primary, equal fast decay, equal slow decay, no transfer. Decoding gain is not MI.

## Results

G28: **FAIL**. Registered profile areas and ablations:

| Metric | Mean | 95% seed CI | Seeds > threshold |
| --- | --- | --- | --- |
| F−H lag area | 0.000000 | [0.0, 0.0] | 0 |
| M−F lag area | 0.000000 | [0.0, 0.0] | 0 |
| minimum equal-decay difference | 0.025013 | [0.015620044944244054, 0.03511509842848143] | 7 |

Primary held-out lag curve:

| State | Lag | CE gain (nats) | Accuracy |
| --- | --- | --- | --- |
| F | 1 | -0.466717 | 0.097168 |
| F | 2 | -0.651746 | 0.079102 |
| F | 4 | -0.792470 | 0.068848 |
| F | 8 | -0.943909 | 0.053711 |
| F | 16 | -1.003368 | 0.042480 |
| F | 32 | -1.027242 | 0.046875 |
| F | 64 | -1.070751 | 0.038574 |
| F | 128 | -1.017018 | 0.038574 |
| F | 256 | -1.075119 | 0.041992 |
| F | 512 | -1.047614 | 0.041504 |
| H | 1 | -0.242508 | 0.042480 |
| H | 2 | -0.228994 | 0.048828 |
| H | 4 | -0.263945 | 0.041992 |
| H | 8 | -0.239360 | 0.047852 |
| H | 16 | -0.197823 | 0.046875 |
| H | 32 | -0.270315 | 0.044922 |
| H | 64 | -0.255720 | 0.038086 |
| H | 128 | -0.234562 | 0.041992 |
| H | 256 | -0.255080 | 0.035156 |
| H | 512 | -0.159719 | 0.067871 |
| M | 1 | -1.084473 | 0.039062 |
| M | 2 | -1.003738 | 0.040039 |
| M | 4 | -1.071370 | 0.035156 |
| M | 8 | -1.091841 | 0.037598 |
| M | 16 | -0.978449 | 0.047852 |
| M | 32 | -1.045420 | 0.036621 |
| M | 64 | -1.075815 | 0.042969 |
| M | 128 | -1.039570 | 0.041992 |
| M | 256 | -1.047497 | 0.038086 |
| M | 512 | -0.927981 | 0.046387 |

Full per-seed/decay curves are in the Parquet shards.

## Scope and limitations

Every primary held-out CE gain in the table is negative; clipping these gains for the registered area yields zero for H, F and M. Thus this probe did not establish the proposed effective-time hierarchy. The raw ablation difference does not rescue that failure. The probe distribution is much longer and more IID than training histories, feature dimension differs by state, and the fixed ridge/score calibration may be inadequate. A null or negative probe result is not proof that history is absent; conversely, a decodable trace would not by itself imply future use or causal long-term memory.
