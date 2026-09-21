# ET-RCM Stage 2C.2 — Frozen-L3 Pathway Audit and Endogenous Memory-Formation Curriculum

> **When the jointly trained endogenous ET-RCM fails to develop behavioral memory, which link in the chain from persistent state to action-conditioned prediction is actually broken: downstream action evaluation, read integration, memory addressing, history encoding, endogenous state formation, or their joint optimization?**

> **当联合训练的 endogenous ET-RCM 无法形成行为记忆时，真正断裂的是哪一级：行为评估、read integration、M 寻址、历史编码、内生记忆形成，还是这些模块之间的联合优化？**

Formal decision: **G51 FAIL (2/8)**. G52–G56: **NOT_RUN_BY_GATE**. This is a gated negative diagnostic, not a completed curriculum trial. Historical Stage 2C/2C.1 claims remain unchanged.

## 1. Scope, prior evidence and frozen discipline

Stage 2C.1 showed G48–G50 PASS 8/8, but its joint endogenous L3 rerun had action-TV 0.011601 ± 0.008913 and BS −0.000206 ± 0.000660. Stage 2C.2 audits those *same failed L3 checkpoints* (7201–7208). Two new development L3 checkpoints (7301–7302) were used only before freezing thresholds. Frozen L3 core, action embedding, consequence head, queries, read gate and F/M law were never updated during P0/P1 or native probes. Every seed records checkpoint SHA-256 and before/after parameter SHA-256. No L1/L2 M tensor was copied into L3.

## 2. Formal protocol and data split

P0 replays four matched N=16 paired histories per formal seed, with no parameter updates. P1 fits only two local H vectors per frozen checkpoint, each projected to that checkpoint's native N=16 median H norm. Four 1,000-step restarts use exactly balanced z×action consequence-CE batches (128; no correct-action label); restart selection uses independent observed-outcome samples, and final CE uses another independent 4,096-sample split. P1 injects H immediately before the frozen action head, so it tests the downstream ceiling, not recurrent integration. All gates were frozen in `configs/stage2c2_formal.yaml` after development 7301–7302 and before formal 7201–7208.

## 3. P0 — native failed-L3 reproduction

Native action-TV 0.011601 ± 0.008913; native entropy BS -0.000206 ± 0.000660. The replay reproduces Stage 2C.1 L3 per seed to numerical precision. Native H/F/M/r_M medians, q_F/q_M vectors, paired states, read vectors and direct forecast metrics are retained in `results/stage2c2/oracle_h/formal/*/native.json`. No nonfinite values were observed in this short-horizon audit.

## 4. P1 — frozen L3-local oracle H ceiling and G51

Oracle-H mean action-TV 0.090644 ± 0.038351 (per-seed minimum latent branch TV 0.076628 ± 0.031549), interaction 0.180747 ± 0.077410, entropy BS 0.135618 ± 0.061464. This is above native sensitivity but below robust recovery. Pre-registered G51 requires both TV branches≥0.10, interaction≥0.15 and entropy BS≥0.10 in at least 6/8 seeds; only 2/8 pass. The frozen downstream head retains limited state-contingent capacity in some checkpoints but fails replicated native-norm control.

| Frozen L3 seed | native TV | oracle-H TV A/B | interaction | entropy BS | G51 | H norm target | held-out CE |
|---:|---:|---:|---:|---:|---|---:|---:|
| 7201 | 0.01205 | 0.11769/0.17766 | 0.29535 | 0.22517 | True | 20.643 | 1.1075 |
| 7202 | 0.00819 | 0.07303/0.06168 | 0.13471 | 0.09900 | False | 11.474 | 1.1788 |
| 7203 | 0.00530 | 0.04088/0.05405 | 0.09060 | 0.06585 | False | 9.846 | 1.1993 |
| 7204 | 0.03146 | 0.16391/0.11572 | 0.27963 | 0.21605 | True | 23.295 | 1.1136 |
| 7205 | 0.00518 | 0.03118/0.05923 | 0.09041 | 0.06284 | False | 8.646 | 1.2004 |
| 7206 | 0.01427 | 0.07541/0.10319 | 0.17860 | 0.13554 | False | 16.620 | 1.1576 |
| 7207 | 0.01254 | 0.11474/0.08954 | 0.20428 | 0.15452 | False | 16.379 | 1.1463 |
| 7208 | 0.00381 | 0.08092/0.09147 | 0.17240 | 0.12597 | False | 15.985 | 1.1605 |

## 5. P2–P4 and curriculum gate discipline

G51 failed. Therefore P2 oracle-read fitting, P3 local oracle-M fitting and swap/read-clamp, P4 history→L3-M trained adapter, routing rescue, and curriculum C1–C3 were **not run**. C0 is the already frozen Stage 2C.1 L3 scratch baseline, not a new curriculum run. G52/G53/G54/G55/G56 are `NOT_RUN_BY_GATE`, not FAIL. No negative result is assigned to read→H, M addressing, history→M mapping, endogenous F/M formation, or curriculum survival from these absent interventions. The read/M tests in the new unit suite check intervention mechanics only, not behavioral gates.

## 6. Native latent information audit (non-causal)

For each frozen seed, 128 independent train and 128 held-out test paired lifetimes were generated. Test history surfaces use parity-odd held-out combinations; train surfaces use parity-even. Balanced z labels were provided only to linear/small-MLP probes, never to L3. Per-feature standardization used train data only. Chance is 0.5.

| State | linear accuracy | small-MLP accuracy |
|---|---:|---:|
| H | 0.543457 ± 0.058999 | 0.530762 ± 0.039531 |
| F | 0.531250 ± 0.045122 | 0.507324 ± 0.017147 |
| M | 0.720703 ± 0.089150 | 0.701172 ± 0.094882 |
| FM | 0.714844 ± 0.105850 | 0.629883 ± 0.096810 |

The highest mean linear accuracy is in M. M linear accuracy exceeds 0.70 in 4/8 seeds, below the preregistered 6/8 condition for a replicated native-M rescue diagnostic. Some seeds nevertheless contain decodable z in M. This is not causal evidence that the native read uses it; with G51 failed, a routing-only explanation is not identifiable.

## 7. Per-layer survival metrics

D_M=||M_A−M_B||, D_r=||r_M,A−r_M,B||, D_H=||H_A−H_B||, D_P is mean fixed-action forecast JS, and D_B is absolute entropy-policy BS. P0 values are observational paired-state differences. P1 is a finite H-state intervention; its M/read entries are not applicable. P1 matches each state's native norm but permits a larger inter-state D_H than native histories, so it is a bounded ceiling, not evidence that endogenous dynamics create those directions. These metrics locate *attenuation*, but only intervention rows support causal claims.

| Link metric | native P0 | injected oracle H P1 |
|---|---:|---:|
| D_M | 0.202596 ± 0.083294 | NOT_RUN |
| D_r | 0.098105 ± 0.041807 | NOT_RUN |
| D_H | 2.290248 ± 1.009302 | 27.986202 ± 10.422746 |
| D_P_JS_fixed_action | 0.000020 ± 0.000016 | 0.005387 ± 0.004174 |
| D_B | 0.000824 ± 0.000507 | 0.135618 ± 0.061464 |

## 8. Independent joint-training learning-curve audit

A separate scratch arm reproduced Stage 2C.1 L3 training without changing F/M law: 500 AdamW steps, batch 16, 4/8/16 experience lengths, consequence CE plus 0.001 H-square penalty. Development seeds 7351–7352 and disjoint formal seeds 7401–7408; parameters were hashed before/after every no-update snapshot at 0/25/50/100/200/300/500. Four paired N=16 lifetimes, a held-out M ridge probe, read/H differences and group gradients were recorded per snapshot. This arm is a **parallel diagnostic**, not a curriculum or G55 result.

Development endpoints diverged sharply: 7351 TV=0.0036, BS=-0.0003; 7352 TV=0.9409, BS=+0.8774. This motivated independent formal replication, not threshold revision.

| Step | action-TV | history-TV | interaction y0 | BS | M-probe accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.04544 ± 0.01568 | 0.00228 ± 0.00129 | 0.00006 ± 0.00050 | 0.00007 ± 0.00021 | 0.5723 |
| 25 | 0.02639 ± 0.01061 | 0.00731 ± 0.00520 | 0.00010 ± 0.00058 | 0.00002 ± 0.00065 | 0.5811 |
| 50 | 0.01419 ± 0.00399 | 0.00558 ± 0.00260 | -0.00006 ± 0.00042 | -0.00006 ± 0.00024 | 0.6025 |
| 100 | 0.01502 ± 0.00766 | 0.00580 ± 0.00328 | 0.00017 ± 0.00197 | -0.00023 ± 0.00078 | 0.5830 |
| 200 | 0.01197 ± 0.00769 | 0.00561 ± 0.00363 | -0.00004 ± 0.00111 | 0.00002 ± 0.00078 | 0.6113 |
| 300 | 0.01097 ± 0.00335 | 0.00354 ± 0.00211 | 0.00065 ± 0.00103 | 0.00050 ± 0.00073 | 0.6133 |
| 500 | 0.01046 ± 0.00798 | 0.00445 ± 0.00296 | 0.00032 ± 0.00132 | 0.00025 ± 0.00097 | 0.6152 |

Across formal seeds, mean action-TV moves from 0.04544 at initialization to 0.01502 by step 100 and 0.01046 at step 500. Thus the dominant formal pattern is early loss of weak random action sensitivity, not a replicated high-TV learned phase followed by collapse.

Mean N=16 H norm changed from 78.215 at step 0 to 15.057 at step 500; this alone is not evidence that H instability caused the behavioral null.

| Curve seed | TV step 0 | TV step 100 | TV step 500 | BS step 500 | M-probe step 500 |
|---:|---:|---:|---:|---:|---:|
| 7401 | 0.04046 | 0.00809 | 0.00441 | -0.00097 | 0.8438 |
| 7402 | 0.02503 | 0.01044 | 0.00941 | -0.00018 | 0.5859 |
| 7403 | 0.06902 | 0.01233 | 0.01229 | +0.00102 | 0.5156 |
| 7404 | 0.05674 | 0.01083 | 0.02237 | +0.00016 | 0.6562 |
| 7405 | 0.05564 | 0.03220 | 0.02235 | +0.00222 | 0.5156 |
| 7406 | 0.04054 | 0.01563 | 0.00318 | -0.00000 | 0.6016 |
| 7407 | 0.05143 | 0.01862 | 0.00629 | +0.00005 | 0.6406 |
| 7408 | 0.02465 | 0.01201 | 0.00334 | -0.00032 | 0.5625 |

At step 500, 0/8 scratch seeds meet the descriptive TV≥0.10 and BS≥0.05 criterion; seed identities: []. The separate Stage 2C.1 failed-checkpoint cohort also showed near-zero behavior in all eight seeds. These are different seeded cohorts, and the descriptive cutoff is not a G55 gate or controlled curriculum effect. Per-seed trajectories and checkpoints are preserved under `results/stage2c2/learning_curves/formal/`.
The striking development-seed 7352 endpoint did not replicate in the eight formal curve seeds; it must be treated as a non-replicated development observation.

At step 500, high M-probe accuracy (≥0.70) coexists with |BS|<0.05 in scratch seeds [7401]; this is a non-causal stored-but-unused *possibility*, not proof of a routing defect.

## 9. Gradient audit

Mean pre-clipping gradient norms at the update immediately before each snapshot (not causal evidence):

| Step | action | consequence | q_M | read gate | event encoder | H core |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.00571 | 0.25021 | 0.00058 | 0.00567 | 0.18777 | 0.26458 |
| 50 | 0.00601 | 0.32057 | 0.00383 | 0.00494 | 0.16939 | 0.24077 |
| 100 | 0.00362 | 0.17889 | 0.00956 | 0.00704 | 0.13267 | 0.16480 |
| 200 | 0.00385 | 0.17893 | 0.00841 | 0.00461 | 0.12561 | 0.16025 |
| 300 | 0.00477 | 0.24004 | 0.00053 | 0.01023 | 0.12268 | 0.16342 |
| 500 | 0.00476 | 0.27307 | 0.00090 | 0.00903 | 0.14133 | 0.14576 |

At step 500, mean action/head/q_M/read-gate gradient norms were 0.00476/0.27307/0.00090/0.00903. They are not identically zero, but small action-branch gradients relative to the head warrant an optimization audit. A small gradient cannot by itself prove starvation; compare its trajectory with action-TV, CE and state differences. H norms are recorded separately; recurrence/stability parameters were not altered.

## 10. Decision localization and next step

By the mandated decision ladder, G51 FAIL localizes the first *replicated demonstrated* blocker to the downstream action-evaluation/head interface in the failed L3 checkpoints, under the preregistered native-norm oracle-H fit. It does not establish that the head has zero capacity at every seed, nor that no other upstream defect coexists. A targeted next stage may pre-register a stronger joint action-conditioning objective or training schedule, while retaining F/M law and explicitly preserving action-TV through training. The unreplicated development-seed 7352 anomaly should be understood before a curriculum claim. Do not redesign F/M write/consolidation or H recurrence from these data alone, and do not rerun the full Stage 2C habit suite yet.

## Direct answers to the 25 required questions

1. Only inconsistently: native-norm oracle H passes G51 in 2/8 frozen checkpoints.
2. The first replicated failure is downstream action-head/interface capacity under frozen joint-trained weights; state quality may also contribute but is not isolated.
3. NOT_RUN_BY_GATE: oracle-read integration behavior was not fitted or evaluated.
4. NOT_RUN_BY_GATE: frozen-L3 oracle-M behavioral control was not evaluated.
5. NOT_RUN_BY_GATE: no fitted oracle-M preference-swap test.
6. NOT_RUN_BY_GATE: no oracle-M read-clamp behavioral test.
7. NOT_RUN_BY_GATE: no trained legal-history→L3-M adapter.
8. Yes, variably: held-out z is most consistently probe-decodable from native M/FM, not reliably from H/F.
9. Highest mean linear-probe accuracy: M; this is non-causal.
10. Not determined: M decodability alone cannot identify why the learned read did not control behavior, especially with G51 failed.
11. NOT_RUN_BY_GATE: routing rescue was not trained; preregistered M decodability criterion also missed.
12. Native observational differences and P1 H intervention are quantified; causal M→read→H survival was not adjudicated downstream of G51.
13. Formal mean action-TV drops mostly by step 100; there is no replicated high-TV learned phase followed by collapse.
14. History-TV and interaction are reported at every trajectory checkpoint; no single pattern is imposed across seeds.
15. Action gradients are relatively small versus the consequence head, while q_M/read gradients vary and are not uniformly zero; this is diagnostic, not proof of starvation.
16. Curriculum comparison NOT_RUN_BY_GATE; scratch training alone was observed.
17. Oracle-M curriculum NOT_RUN_BY_GATE; final zero-oracle survival unknown.
18. History-M curriculum NOT_RUN_BY_GATE; final zero-teacher survival unknown.
19. Not established by the failed frozen cohort or eight formal scratch-curve seeds; one positive development scratch seed was non-replicated and is not G55.
20. New G56 interventions NOT_RUN_BY_GATE; frozen Stage 2C.1 peripheral swaps were near-null.
21. Primary localized blocker: downstream action-evaluation/head interface in frozen failed L3; joint optimization and upstream defects remain possible.
22. No evidence yet requiring an F/M external-write-law change.
23. No evidence yet requiring readout-conserving consolidation change.
24. No: no demonstrated nonfinite or extreme-norm causal relation warrants H redesign now.
25. No: G51 failed and G55/G56 were not run; full Stage 2C rerun is premature.

## Integrity, tests and limitations

All formal fitted states are L3-local and norm matched. The 906-file prior Stage 2C/2C.1 tracked-artifact SHA-256 baseline was captured before Stage 2C.2 and verified unchanged at reporting. New Stage 2C.2 tests pass. The repository-wide suite retains one pre-existing Stage 1.5 README immutability assertion failure inherited from the Stage 2C commit, not changed here. No oracle-M/read/history adapter or curriculum data were fabricated when gated off. P1's 1,000-step/four-restart optimizer is a bounded ceiling test, not proof of absolute mathematical incapacity; formal outcome targets share the same world rule but independent sampled observations. The separate scratch-curve arm is not a curriculum and cannot revise frozen Stage 2C or Stage 2C.1 gates.
