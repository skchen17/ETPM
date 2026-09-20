# Architecture Stability — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Four genuinely external-history worlds × 8 initial states × 8 fresh checkpoints. Every NULL tick through 1024 is recorded; no external writes occur. Finite H/F/M perturbations of 0.001 and 0.01 are tracked to 128; no JVP substitutes for these rollouts. G27 uses maximum state norm, prediction JS drift, and median finite-perturbation growth.

## Results

G27: **FAIL**. Per-seed maximums:

| Seed | Max norm | Max JS | Max growth@128 | Tail Δ | Peak period | Peak power | Descriptive class | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8501 | 2067.494385 | 0.504883 | 4702.291260 | 1.454199 | 256.000000 | 0.423941 | unstable_or_divergent_on_sample | FAIL |
| 8502 | 1744.601562 | 0.539797 | 1107.058350 | 1.334614 | 256.000000 | 0.684267 | unstable_or_divergent_on_sample | FAIL |
| 8503 | 1822.044189 | 0.597392 | 1174.324341 | 1.263304 | 256.000000 | 0.602003 | unstable_or_divergent_on_sample | FAIL |
| 8504 | 1864.501709 | 0.485146 | 896.011688 | 1.341386 | 256.000000 | 0.652299 | unstable_or_divergent_on_sample | FAIL |
| 8505 | 1750.943115 | 0.620335 | 1439.803345 | 1.193677 | 256.000000 | 0.391484 | unstable_or_divergent_on_sample | FAIL |
| 8506 | 1941.235352 | 0.416281 | 2248.625488 | 1.396873 | 256.000000 | 0.575300 | unstable_or_divergent_on_sample | FAIL |
| 8507 | 1941.082520 | 0.542308 | 2177.211731 | 1.278979 | 256.000000 | 0.338434 | unstable_or_divergent_on_sample | FAIL |
| 8508 | 1820.722778 | 0.593880 | 3479.373901 | 1.279471 | 256.000000 | 0.526681 | unstable_or_divergent_on_sample | FAIL |

Mean trajectories:

| K | ||H|| | ||F|| | ||M|| | ΔH | JS from K0 | Entropy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 20.415419 | 1.184782 | 0.134886 | 0.000000 | 0.000000 | 1.439862 |
| 1 | 22.109428 | 1.146202 | 0.145895 | 1.833720 | 0.000208 | 1.429656 |
| 8 | 33.911503 | 0.916363 | 0.199783 | 1.738019 | 0.003031 | 1.421168 |
| 64 | 113.111891 | 0.164454 | 0.252366 | 1.367645 | 0.027274 | 1.572647 |
| 256 | 358.754430 | 0.000473 | 0.230619 | 1.325779 | 0.084600 | 1.757707 |
| 1024 | 1346.941569 | 0.000000 | 0.157064 | 1.316955 | 0.163874 | 1.815457 |

Perturbation growth by component/epsilon/tick:

| Component | ε | K | H response | Full response | Prediction JS |
| --- | --- | --- | --- | --- | --- |
| F | 0.001000 | 1 | 0.663834 | 1.320015 | -0.000000 |
| F | 0.001000 | 8 | 6.131926 | 6.317955 | 0.000000 |
| F | 0.001000 | 32 | 23.860660 | 23.875233 | 0.000000 |
| F | 0.001000 | 128 | 88.308093 | 88.308453 | 0.000000 |
| F | 0.010000 | 1 | 0.635227 | 1.291365 | 0.000000 |
| F | 0.010000 | 8 | 5.890523 | 6.079579 | 0.000002 |
| F | 0.010000 | 32 | 23.155892 | 23.170410 | 0.000007 |
| F | 0.010000 | 128 | 92.755377 | 92.755732 | 0.000031 |
| H | 0.001000 | 1 | 1.002670 | 1.002671 | 0.000000 |
| H | 0.001000 | 8 | 1.077080 | 1.077093 | -0.000000 |
| H | 0.001000 | 32 | 1.503383 | 1.503394 | -0.000000 |
| H | 0.001000 | 128 | 3.565770 | 3.565774 | 0.000000 |
| H | 0.010000 | 1 | 1.002049 | 1.002050 | 0.000000 |
| H | 0.010000 | 8 | 1.072344 | 1.072358 | 0.000000 |
| H | 0.010000 | 32 | 1.512118 | 1.512130 | 0.000000 |
| H | 0.010000 | 128 | 3.847740 | 3.847744 | 0.000000 |
| M | 0.001000 | 1 | 4.228661 | 4.520504 | 0.000000 |
| M | 0.001000 | 8 | 36.325266 | 36.412056 | 0.000000 |
| M | 0.001000 | 32 | 176.321993 | 176.347899 | 0.000003 |
| M | 0.001000 | 128 | 991.744683 | 991.749481 | 0.000023 |
| M | 0.010000 | 1 | 4.256487 | 4.546996 | 0.000001 |
| M | 0.010000 | 8 | 36.575673 | 36.661200 | 0.000021 |
| M | 0.010000 | 32 | 173.112491 | 173.137783 | 0.000288 |
| M | 0.010000 | 128 | 907.127293 | 907.131932 | 0.001584 |

## Scope and limitations

These are finite empirical trajectories on trained toy states. Boundedness over 1024 ticks is not a global stability theorem; perturbation growth is not called mathematical chaos.
