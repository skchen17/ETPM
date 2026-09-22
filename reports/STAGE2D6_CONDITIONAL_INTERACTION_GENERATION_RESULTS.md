# ET-RCM Stage 2D.6 — Conditional Interaction Generation in the Fusion Nonlinearity

> **How does the fusion nonlinearity transform history-dependent state effects and candidate-action effects into the causal state×action interaction required for behavioral memory, and why does this computation form only in some training runs?**
> **fusion 非线性如何把 history-derived state main effect 与 candidate-action main effect 转换成行为相关的 state×action interaction？为何只在部分训练 run 中形成？**
## Executive result

**Outcome D — no replicated nonlinear-generation explanation.** The new 24-run cohort contained only **7 healthy** endpoints, below the prespecified 8-run healthy confirmation minimum; gates marked `INCONCLUSIVE_N_LT_8` are not mechanistic failures. The frozen SiLU often enlarges factorial interaction in healthy checkpoints, but descriptive generation is not sufficient to establish a causal operating-point mechanism. Development froze the global common offset at **+0.25 native SD**, with selected top-**8** finite-contribution units before confirmatory labels were available. The diagnostic stopping rule did not authorize training or memory-law changes. Fusion architecture redesign and formal F→M handoff remain unjustified.
| Gate | Decision |
|---|---|
| G110 | INCONCLUSIVE_N_LT_8 |
| G111 | INCONCLUSIVE_N_LT_8 |
| G112 | FAIL |
| G113 | INCONCLUSIVE_N_LT_8 |
| G114 | FAIL |
| G115 | FAIL |
| G116 | FAIL |
| G117 | FAIL |
| G118 | NOT_RUN_BY_PROTOCOL |
| G119 | NOT_RUN_BY_PROTOCOL |
| G120 | NOT_RUN_BY_PROTOCOL |
| G121 | NOT_RUN_BY_PROTOCOL |
| G122 | NOT_RUN_BY_PROTOCOL |

## Frozen design and exact computation

The real candidate path is `context/event → candidate temporary H → pooled H → W_H H + W_A e_a + b → SiLU → protected linear logits → softmax`. The H branch is already candidate-conditioned; consequently preactivation interaction is not theoretically forced to zero. The action projection itself is action-only and its 2×2 interaction is numerically zero. All 2×2 histories/actions and 16 within-run replicates were aggregated to independent-run statistics. Analytic SiLU slope/curvature, exact hook equality, finite output response and persistent-state/parameter purity were unit-tested.
Development used all **24** pre-existing C0 legacy runs (primary sorted 8 healthy/8 shortcut). The independent confirmatory baseline trained **24** new C0 runs from initialization seeds 21101–21108 × streams 22101–22103, 1500 updates each, batch 16, episodes 4/6/8, observed-only consequence cross-entropy, γ=.50, ρF=.97, ρM=.9995, no paired scaffold, 1000-step protected evaluator pretraining. Ten checkpoint times were retained. Their training summaries, checkpoint paths, final hashes and immutable-head checks are machine-readable; binary checkpoints remain on the research server and are git-ignored. A1 original/confirmatory checkpoints (24 total) were analyzed only as supplementary F1/F2 context, never pooled into C0 replication gates. Healthy/partial/shortcut labels use the frozen TV/IHA/BS/CFA thresholds, not fusion metrics.
The finite 4×32 probability-interaction response to ±0.02/4 per-unit post-SiLU perturbation was SVD-decomposed; Q reports projection onto ranks 1/2/4. `C_j` is the finite IHA loss from removing one unit's factorial interaction. The development offset grid was {-1,-.5,-.25,0,.25,.5,1} native preactivation SD; the *group-mean* best scale, not each run's best, was frozen. Common offsets were identical across four cells and verified to preserve preactivation state/action interaction contrasts. Controls included same-support random signed offsets, curvature-lowering offsets, state/action projection scaling, same-k random unit removal and a known post-SiLU injection ceiling. This ceiling did not enter any rescue gate.

## Factorial fusion anatomy and behavioral alignment

Values are means over independent runs. Each cell is healthy/shortcut (not neuron-level pseudo-replicates).

| Cohort | Runs | Basin counts | pre-I H/S | post-I H/S | ΔNL H/S | Q1 H/S | O H/S |
|---|---:|---|---:|---:|---:|---:|---:|
| legacy development | 24 | {'healthy': 12, 'shortcut': 12} | 0.469/0.029 | 1.947/0.185 | 1.478/0.156 | 0.627/0.113 | 5.343/0.520 |
| new confirmatory | 24 | {'healthy': 7, 'shortcut': 17} | 0.311/0.017 | 1.713/0.305 | 1.402/0.288 | 0.641/0.124 | 4.124/0.786 |
| A1 supplementary | 24 | {'healthy': 4, 'partial': 10, 'shortcut': 10} | 0.005/0.008 | 0.358/0.106 | 0.353/0.098 | 0.880/0.340 | 0.938/0.285 |

**historical:** healthy state/action main norms 2.264/15.807; shortcut 0.223/15.341. Mean curvature healthy/shortcut 0.255/0.254; rank-2 response energy 0.997/0.997. Q ranks 1/2/4 healthy = 0.627/0.636/0.658, shortcut = 0.113/0.134/0.295. G110 positive healthy **8/8**, Q1 paired direction **8/8**, O paired direction **8/8**. AUC(O) 1.000; state/action/curvature/fusion norm comparators 1.000/0.719/0.750/0.562.

Run-level healthy−shortcut post_I difference 1.822, 95% bootstrap CI [1.478, 2.199]. Run-level healthy−shortcut Q1 difference 0.561, 95% bootstrap CI [0.469, 0.654]. Run-level healthy−shortcut overlap difference 5.029, 95% bootstrap CI [3.988, 6.260]. These intervals resample independent trained runs, not neurons or histories.

**confirmatory:** healthy state/action main norms 1.912/15.477; shortcut 0.362/16.077. Mean curvature healthy/shortcut 0.263/0.240; rank-2 response energy 0.997/0.997. Q ranks 1/2/4 healthy = 0.641/0.646/0.664, shortcut = 0.124/0.179/0.291. G110 positive healthy **7/7**, Q1 paired direction **7/7**, O paired direction **7/7**. AUC(O) 1.000; state/action/curvature/fusion norm comparators 1.000/0.286/0.857/0.321.

Run-level healthy−shortcut post_I difference 1.416, 95% bootstrap CI [1.100, 1.776]. Run-level healthy−shortcut Q1 difference 0.524, 95% bootstrap CI [0.416, 0.621]. Run-level healthy−shortcut overlap difference 3.417, 95% bootstrap CI [2.867, 3.981]. These intervals resample independent trained runs, not neurons or histories.

The repeated phenotype is a much stronger **fusion state-main** contrast in healthy runs, while the action main effect remains large in both basins and mean SiLU curvature changes little. O separates classes descriptively, but its AUC ties the state-main norm (both 1.000), so O adds no independent basin discrimination under G112. Because `fusion_pre` already has history×action interaction from candidate-conditioned H, the post-SiLU increase is an amplification/nonlinear transformation of a mixed input, not proof that SiLU created the full causal pattern from two pure main effects. F2's small fusion state-main despite high upstream probes strengthens the distinction between stored information and useful fusion input.

## Finite unit causality, offsets and training timing

**historical:** frozen k=8 removal met the finite-vs-random criterion in **8/8** healthy runs; same-k random means and all k=1/2/4/8/16 outcomes are in `unit_causality/summary.json`. Fixed common-offset full rescue **0/8** failed; curvature-lowering healthy destruction meeting its *matched common-shift* control **1/8**; O half-rise before/at IHA onset **4/8**.

**confirmatory:** frozen k=8 removal met the finite-vs-random criterion in **7/7** healthy runs; same-k random means and all k=1/2/4/8/16 outcomes are in `unit_causality/summary.json`. Fixed common-offset full rescue **0/8** failed; curvature-lowering healthy destruction meeting its *matched common-shift* control **5/7**; O half-rise before/at IHA onset **3/7**.

Development offset grid mean IHA changes: -1.00:-0.0088, -0.50:-0.0064, -0.25:-0.0026, +0.00:+0.0000, +0.25:+0.0004, +0.50:-0.0017, +1.00:-0.0073. The selected scale had 0/8 development primary failed runs with >.025 IHA improvement. Output-aligned post-SiLU positive ceiling achieved threshold behavior in **49/49** failed audits; this confirms the assay's downstream reach but cannot rescue the preactivation hypothesis.

A common preactivation shift preserves the *preactivation* factorial S/A/I contrasts exactly, but may change their post-SiLU images. An extreme ±3 SD low-curvature shift can also destroy generic information; therefore matched random-sign common shifts, not native-only comparisons, decide G115. Curvature overlap O is descriptive and cannot be called causal merely because it covaries with healthy basin entry. The timing test does not infer mediation from order alone.

## F1/F2 and conditional stopping

Across explicitly labeled cohorts, F1-like nonhealthy n=36 had mean fusion state-main/post-I/Q1/O 0.246/0.209/0.218/0.542. Strict F2 stored-but-unused (high H/M probe, |BS|<.10) n=10 had 0.274/0.227/0.294/0.615; healthy n=23 had 1.828/1.599/0.675/4.206. Another 3 high-probe partial cases with |BS|≥.10 are kept separately. An earlier broad probe-only F2-like grouping had n=13; it is retained machine-readably but excluded from G117 because it omitted the frozen |BS| criterion. Fixed common-offset full rescue occurred in **0/10** strict F2 audits. Probe-based F2 status alone does not establish that state information reached the right fusion direction.

G114/G115/G116 did not produce the two passes required to authorize any training intervention. Hence G118–G122 and reduced formation (N=0/1/4/16/64), persistence (D=0/100/500/1000), revision (R=0/8/32/128), selectivity, F/M clamps and 1000/5000/10000-tick stabilized-model diagnostics are **NOT_RUN_BY_PROTOCOL**. This is not a measured failure of a training rescue or a claimed lack of long-run safety; no stabilized model exists. F/M laws and the F→M handoff stay frozen.

## Direct answers to the 35 required questions

1. **fusion_pre interaction:** legacy healthy/shortcut 0.469/0.029; new 0.311/0.017. Nonzero because candidate H is action-conditioned.
2. **fusion_post interaction:** legacy 1.947/0.185; new 1.713/0.305.
3. **Healthy nonlinear generation:** G110 INCONCLUSIVE_N_LT_8; mean ΔNL legacy/new healthy 1.478/1.402.
4. **Shortcut generation:** mean ΔNL legacy/new 0.156/0.288; compare absolute post-I, not just ratio.
5. **State main effect:** healthy/shortcut legacy 2.264/0.223; see F1/F2 caveat.
6. **Action main effect:** healthy/shortcut legacy 15.807/15.341.
7. **Interaction norm sufficiency:** No causal sufficiency follows from its basin separation; Stage 2D.5 upstream counterexample remains.
8. **Behavioral alignment:** G111 INCONCLUSIVE_N_LT_8; finite response/SVD Q1, not only gradient.
9. **Low-dimensional output response:** rank-2 energy healthy legacy/new 0.997/0.997; this is local output response, not proof that the full internal interaction is globally low-rank.
10. **Sparse support:** finite per-unit contributions and k=1/2/4/8/16 curves retained; only the frozen G113 criterion licenses a causal set.
11. **Top-k removal:** G113 INCONCLUSIVE_N_LT_8; frozen k=8 passes legacy 8/8, new 7/7.
12. **Random-k:** 16 same-k deterministic controls per run; their mean losses are in `unit_causality/summary.json` and explicitly enter G113.
13. **Operating points:** mean preactivation legacy healthy/shortcut -0.426/-0.403; full per-unit four-cell distributions retained in each run JSON.
14. **Curvature exposure:** mean |SiLU''| healthy/shortcut legacy 0.255/0.254; new 0.263/0.240.
15. **Curvature-weighted overlap:** G112 FAIL; O legacy/new healthy 5.343/4.124.
16. **Failed common-offset rescue:** G114 FAIL; legacy 0/8, new 0/8.
17. **Offset/scale controls:** random, low-curvature, state-scale and action-scale were norm/support-matched where applicable; all recorded per run and required to lose to the primary offset.
18. **Healthy destruction:** G115 FAIL; matched-control-qualified legacy 1/8, new 5/7.
19. **Temporal order:** G116 FAIL; O-before/at-IHA legacy 4/8, new 3/7.
20. **F1 phenotype:** fusion S/post-I/Q1/O = 0.246/0.209/0.218/0.542; weak state formation is a probe-qualified interpretation, not shortcut label alone.
21. **F2 phenotype:** fusion S/post-I/Q1/O = 0.274/0.227/0.294/0.615; high H/M probe can coexist with weak or misaligned fusion interaction.
22. **F2 rescue:** fixed preactivation offset 0/10; G117 FAIL.
23. **Training authorization:** No; fewer than two of G114/G115/G116 passed.
24. **Fusion architecture redesign:** Not justified by this diagnostic.
25. **Initialization/stabilization:** Not tested; causal authorization failed, so it cannot be claimed sufficient.
26. **Training healthy rate ≥6/8:** Not run by protocol; the new 24-run baseline's healthy count is reported separately.
27. **Mechanism recovery after training:** Not run by protocol.
28. **Gradual accumulation:** Not rechecked in a stabilized model.
29. **Persistence:** Not rechecked in a stabilized model.
30. **Revision:** Not rechecked in a stabilized model.
31. **Predictive selectivity:** Not rechecked in a stabilized model.
32. **F/M mediation:** Not rechecked; no stabilized model was authorized.
33. **Continuous regression:** Not assessed; no new model was trained.
34. **F/M-law redesign:** Not justified.
35. **Formal F→M handoff:** Remains closed.

## Integrity, limitations and reproducibility

Stage 2D.6 dedicated tests passed **15/15**. Dedicated historical tracked-asset manifest: **2611/2611**, zero changes. The full repository suite passed **248/255**; seven older historical snapshot tests now fail because their selection logic includes later-stage tracked files or cumulative README edits (`test_stage1_5`, `test_stage2d`, `test_stage2d1`–`test_stage2d5`). The Stage 2D.5 snapshot test becomes the seventh failure when Stage 2D.6 files are tracked. These tests were not modified or waived. The frozen evaluator head was unchanged in all 24 new training runs; across all audited endpoints, the exact candidate hook's maximum logit error was **9.54e-07** and protected-head, parameter and persistent-state purity checks all passed. No complete confirmatory endpoint summary existed at development-rule freeze: **True**; this is an endpoint-availability check, not a cryptographic seal. The 8×3 design shares initialization within each triplet, so 24 runs are not 24 independent initialization draws. Per-run JSON includes seeds, checkpoints, 2×2 fusion cell means, derivatives, rank-1/2/4 Q, finite unit contributions, 10-step trajectories, full offset grid and controls. The top-k units were selected and evaluated on the same run's factorial sample; a held-out-history validation would be needed before making a strong sparse-circuit claim. Legacy/A1 architecture differences are not hidden. The intervention on post-SiLU units is a ceiling, not evidence that their successful pattern was caused by memory. No result implies human-like cognition or causal F→M handoff.
