# ET-RCM Stage 2D.1 — Success-vs-Failure Training Basin Audit and Routing Robustness

> **Why do only a minority of identically specified ET-RCM training runs enter the noisy behavioral-memory regime, and is the successful basin caused by slow-memory routing, by earlier conditional-binding dynamics, or by their interaction?**
>
> **为什么在完全相同的 ET-RCM 架构与训练协议下，只有少数训练运行能够进入 noisy behavioral-memory regime？成功 basin 是由 slow-memory routing 因果驱动、由更早的 conditional-binding dynamics 驱动，还是二者共同形成？**

## Executive result

**Outcome D — Basin Not Explained.** The expanded baseline reproduced a clear success/failure split, but not the earlier exact 2/8 rate: the complete 8-initialization × 4-stream crossing produced 16/32 healthy, 1/32 partial, and 15/32 shortcut runs; one fixed stream produced 4/8 healthy. Initialization and initialization×stream interaction dominated the variance, while the main stream effect was small.

High M routing was associated with success and small healthy-model M-read effects were measurable, but it was neither necessary in every healthy run nor sufficient to rescue failures. G68 failed because the best *single* frozen intervention improved all three registered interfaces in only 4/8 failed checkpoints (0/8 if a +.01 margin per interface is required). The development-selected 50-step temporary M-floor curriculum achieved only 4/8 healthy formal runs. Consequently G68–G72 all failed and Stage 2D.1 does not authorize a new formal F→M handoff experiment or an F/M-law redesign.

## Frozen experimental design

No external write, F→M conservation, decay, memory dimension, H recurrence, action evaluator, NULL, or SELF_OUTPUT rule was changed. The baseline was the Stage 2D A2 protected-evaluator model with:

| Field | Frozen value |
|---|---:|
| gamma | .50 |
| rho_fast / rho_slow | .97 / .9995 |
| baseline training evidence p | .70 |
| training steps / batch | 1,500 / 16 |
| lifetime lengths | 16, 32, 64 |
| evaluator pretraining | 1,000 steps, then frozen |
| high-resolution checkpoints | 0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500 |

Initialization seeds 9101–9108, stream seeds 12101–12104, and evaluation seed 15101 were strictly separate. The 32-cell crossing uses every initialization with every stream, so stream effects are estimable within an initialization and initialization effects are estimable on a fixed stream. The residual term is explicitly named `interaction_residual`: with one run per cell, interaction and residual stochasticity cannot be separated.

Final basin labels were assigned only from action-TV, absolute history×action interaction, and absolute BS. M gate, probes, gradients, and final forecast quality were not label inputs. A run was healthy only when all three values were at least .10; partial meant one or two passed; shortcut meant none passed.

### Checkpoint measurements

Every checkpoint received the same held-out 32-experience evaluation and recorded:

- action-TV, history-TV, interaction, BS, observed/conditional/marginal CE, and CFA;
- centroid probes `Acc(z|H/F/M/FM)` on held-out paired histories;
- full F/M gate distributions (mean, SD, median, p10/p25/p75/p90 and per-event values), split by early/late, predictive/matched-noise, and NULL events;
- q/r norms, q/r cosine, latent-conditioned differences, external-write, F→M transfer, cumulative transfer, decay contributions, H increments, and H/F/M norms;
- pre-update gradients for H core, event encoder, q_F, q_M, read gate, learned access/consolidation, action branch, and frozen evaluator.

Probes and gradients are diagnostic only. The evaluator gradient was zero as required. Cross-seed representation comparisons use CKA and principal angles, not raw coordinates.

## A–C. Basin, variance, and earliest divergence

### Expanded baseline outcome

Healthy counts by initialization across four streams were:

| Init | 9101 | 9102 | 9103 | 9104 | 9105 | 9106 | 9107 | 9108 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Healthy / 4 | 2 | 4 | 2 | 0 | 4 | 0 | 1 | 3 |

Healthy counts by stream across eight initializations were 4, 5, 4, and 3. Thus no stream deterministically produced success, whereas two initializations succeeded on every stream and two failed on every stream.

Method-of-moments variance fractions were:

| Endpoint | Initialization | Stream | Interaction/residual |
|---|---:|---:|---:|
| action-TV | .480 | .025 | .495 |
| interaction | .484 | .010 | .506 |
| BS | .462 | .009 | .529 |
| CFA | .301 | .000 | .699 |

The appropriate conclusion is **mixed initialization and initialization×stream basin**, not a pure data-order effect.

### Earliest divergence

The descriptive bootstrap/effect-size screen found final-label group separation in action-TV/CFA at checkpoint 0, H norm at 10, M gate at 25, H probe at 50, behavioral separation and M probe at 500, r_M latent difference at 500, and transfer amount at 750. The step-0 result must not be read as memory formation: the evaluator is already pretrained, the lifetime model is not, action-TV is high for unrelated random latent representations, and labels are assigned retrospectively. At step 0 mean interaction and BS were both near zero in both groups.

The first scientifically meaningful joint separation was step 500:

| Final group | action-TV | interaction | BS | M probe | mean g_M |
|---|---:|---:|---:|---:|---:|
| healthy | .151 | .216 | .142 | .652 | .305 |
| failed | .063 | .014 | -.004 | .516 | .315 |

At step 750, healthy/failed M probe was .758/.523 and cumulative early-outcome transfer was 32.28/25.89. At step 1500, healthy/failed M probe was .930/.566, BS was .504/-.003, and mean g_M was .480/.386. Thus useful conditional binding and decodable M emerge together around step 500; a high gate is not the earliest or sufficient explanation.

Two failed runs were `stored-but-unused` by the declared diagnostic (`M probe ≥ .70`, `|BS| < .10`): `i9106_d12104` (.75, .005) and `i9107_d12102` (.75, .004). This directly separates information in M from behavioral use of M.

Alignment-invariant geometry did not reveal a simple shared memory subspace signature. Mean trajectory CKA within healthy versus healthy-to-failed was .957/.945 for H, .507/.517 for F, and .500/.534 for M; the corresponding three-dimensional maximum principal angles at the endpoint were 85.6°, 88.5°, and 84.3°. These high angles and overlapping CKA values argue against a single coordinate-free F/M geometry that cleanly explains the basin at this sample size.

## D–E. Frozen routing causality and healthy necessity

Parameters and input F/M tensors were unchanged by all interventions. One-factor raw-read scaling used alpha in `{0,.5,1,1.5,2}`; normalized gate overrides used `g_M in {0,.2,.4,.6,.75,.9,1}`; joint scaling was limited to `(0,0),(.5,1.5),(1.5,.5),(2,2)`.

No common rescue generalized to six failed models. The largest counts were 4/8 for alpha_F=1.5, alpha_F=2, gate_M=0, gate_M=.9, and two joint settings. Normalized `g_M≈.7–.9` sometimes changed a failed model, but no fixed normalized gate rescued more than 4/8 and none achieved a +.01 improvement in action-TV, interaction, and |BS| simultaneously. **G68 FAIL.**

In eight healthy checkpoints, zeroing M reads gave a positive signed BS reduction in 4/8 early-formation, 6/8 late-formation, 6/8 mid-delay, 6/8 late-delay, and 6/8 probe runs. Mean reductions were -.0012, .0037, .0003, .0074, and .0112 respectively. These small effects support limited M-read participation late in a lifetime, not strong necessity. Together with failed models whose g_M was high, the evidence rejects “M-heavy routing alone causes the basin.”

## F. Curriculum development and formal replication

Development used two matched new init/stream pairs. C1 (`.90→.80→.75→.70→.65`) and C3 (`p=.90` for 300 steps, then .70) were each 2/2 healthy. C2 screened the registered 3×3 grid; outcomes ranged from 0/2 to 2/2. C4 was not run because G68 failed and independent C1+C2 causal benefit was not established.

The frozen selector first maximized healthy count, then mean action-TV + interaction + |BS| + positive CFA. It selected **C2, g_M floor .65 for only the first 50 steps** (2/2 development; quality 1.843). The constraint was absent for the final 1,450 updates and for all evaluation. C1 is the least routing-specific scaffold, but it was not the frozen winner.

Eight new formal init/data pairs (9301/12301 through 9308/12308; evaluation seed 15501) produced:

| Seed | G69 healthy | early/mid/late BS (N=4/16/64) | D500 ratio | revision | selectivity | G71 |
|---|---|---|---:|---|---|---|
| 9301 | yes | .338 / .303 / .273 | .099 | yes | no | no |
| 9302 | no | .005 / .027 / .069 | .021 | yes | no | no |
| 9303 | no | .031 / .061 / .001 | .021 | yes | no | no |
| 9304 | no | .007 / .002 / .003 | .295 | yes | no | no |
| 9305 | yes | .080 / .359 / .529 | .042 | yes | no | no |
| 9306 | no | .008 / .005 / .003 | .701 | yes | no | no |
| 9307 | yes | .245 / .478 / .588 | .529 | yes | yes | yes |
| 9308 | yes | .213 / .288 / .385 | .290 | yes | yes | yes |

The formal learned g_M means were .953, .298, .659, .717, .928, .738, .884, and .619. Healthy mean was .846 and failed mean .603, but three failed runs retained g_M≥.65 after the warm-up was removed. This is another direct failure of routing sufficiency.

## Formal gates

| Gate | Result | Evidence |
|---|---|---|
| G68 frozen routing rescue | **FAIL** | best common intervention 4/8; +.01-margin result 0/8 |
| G69 robust conditional binding | **FAIL** | 4/8, required ≥6/8 |
| G70 gradual formation | **FAIL** | 3/8 early<mid<late, required ≥6/8 |
| G71 dynamics recovery | **FAIL** | 2/8 jointly passed persistence, revision, selectivity |
| G72 peripheral mediation | **FAIL** | best common finite read intervention was late-formation FM clamp, 5/8 |

Revision was finite in 8/8, but meaningful persistence passed 2/8 and predictive-over-matched-noise selectivity passed 2/8. These component successes cannot be promoted to the compound G71 claim.

## Continuous-running audit

All eight selected models completed 10,000 ticks without a nonfinite state. One run crossed H norm 100 and 1000; its 10k H norm was about 840. The other seven had 10k H norm from 1.5 to 9.5. Therefore the curriculum did not create majority instability, but it did not eliminate the inherited long-run H-growth failure mode.

At 10k ticks the eight M gates ranged from .309 to .918 and BS ranged from -.041 to .451. Gate magnitude again did not determine retention. The continuous check is diagnostic, not a new pass gate.

## Answers to the 26 required questions

1. **Was Stage 2D's 2/8 success reproduced?** A minority/heterogeneous basin was reproduced, but not the exact rate: fixed-stream baseline was 4/8 and the full crossing was 16/32.
2. **Initialization or stream variance?** Initialization (~46–48%) and interaction/residual (~49–53%) dominated; pure stream main effect was ~1–3%.
3. **Earliest split?** Descriptive label separation exists at step 0; meaningful BS/M-information separation first appears at step 500.
4. **Action-TV or M routing first?** Action-TV separates at step 0, gate_M at 25. The former is an evaluator/random-representation effect, not learned memory.
5. **Interaction or latent-M information first?** Statistical interaction separation is present at step 0 but both groups are near zero; meaningful interaction and M-probe separation co-emerge at about step 500.
6. **Earlier z-decodable M in successful seeds?** Yes descriptively: .652 vs .516 at step 500 and .758 vs .523 at 750.
7. **Is high M gate only correlation?** It is correlated and weakly involved in healthy readout, but not sufficient; the strongest defensible statement is association plus limited late read mediation.
8. **Can more M read rescue failed behavior?** Not reproducibly. No fixed scaling rescued ≥6/8.
9. **Can normalized override rescue?** Not reproducibly; best normalized settings reached only 4/8 and no setting passed the +.01 sensitivity analysis.
10. **Does lowering M routing damage healthy behavior?** Often slightly in late/probe windows (6/8), but mean effects are small and early effects are inconsistent.
11. **Causal necessity/sufficiency?** Strong necessity and sufficiency are not established.
12. **Do q_M/r_M dynamics distinguish basins?** q_M-related separation appears by step 25; r_M latent difference appears at 500. Because q is normalized and rescue fails, these are diagnostics, not a causal explanation.
13. **Does F→M transfer differ?** Yes later: 32.28 vs 25.89 at step 750, but divergence follows conditional behavior and does not prove transfer caused it.
14. **Stored-but-unused failures?** Yes, two explicit runs had M-probe ≥.75 with |BS|≈.005.
15. **Did evidence curriculum improve success?** C1 was 2/2 in development only; no formal robustness claim is allowed.
16. **Did temporary M warm-up improve success?** It looked promising in development, but formal success was only 4/8, below G69.
17. **Was learned routing maintained after removal?** Often: 7/8 final g_M values exceeded .60, but several such runs still failed.
18. **Least artificial curriculum?** C1. The selected C2 used the most limited direct scaffold (50/1500 updates), then became fully endogenous.
19. **Did selected curriculum reach ≥6/8?** No, 4/8.
20. **Was gradual formation replicated?** No, 3/8.
21. **Were persistence/revision/selectivity jointly recovered?** No, 2/8; revision alone was 8/8.
22. **Was peripheral mediation replicated?** No, best common intervention was 5/8.
23. **Did curriculum worsen continuous stability?** No majority worsening or nonfinite state; one 10k run nevertheless reached H≈840 and crossed 1000 earlier in the stream audit definition.
24. **Best bottleneck category?** Mixed initialization basin + initialization×stream/conditional-binding bootstrap. M routing is a participant/consequence, not a standalone bootstrap explanation.
25. **Reason to modify F/M law now?** No. Conditional binding is still not guaranteed, so the stopping rule for reconsidering the memory law is not met.
26. **Re-enter formal F→M handoff?** No. G69–G72 did not pass; G66 remains closed.

## Integrity, tests, and claim boundary

The new suite adds 14 tests for exact transition reproduction, seed separation, routing normalization, nonmutation, frozen parameters, schedule/warm-up removal, absence of privileged labels, matched noise, NULL write safety, and historical integrity. Stage 2C.x/Stage 2D tracked assets are checked against a 2,104-file pre-stage SHA256 manifest.

The Stage 2D.1 suite passed 14/14 and the 2,104-file manifest had zero changes. The repository-wide run collected 182 tests: 180 passed and two pre-existing frozen-test defects failed. `test_stage1_5` treats any later README addition as a forbidden Stage 1.4 edit; `test_stage2d` includes its own Stage2C-named manifest in the current snapshot although that file was not included when the manifest was created. Neither failure reflects a changed frozen experimental asset, and neither historical test was rewritten to obtain a green run.

These results concern a small synthetic noisy-evidence world. They do not demonstrate human-like memory, consciousness, autonomous human thought, infinite information capacity, or unrestricted causal memory. `healthy` is only the preregistered interface label for this experiment. Failures and null results were retained without changing gates.

## Machine-readable evidence

- `results/stage2d1/training_trajectories/`: 32 × 14 checkpoint audits
- `results/stage2d1/processed/factorial_analysis.json`: variance, divergence, CKA/principal-angle analysis
- `results/stage2d1/routing_interventions/routing_sweep.json`: frozen rescue and healthy necessity
- `results/stage2d1/curricula/selection.json`: development table and frozen choice
- `results/stage2d1/behavior/formal_selected/`: eight full Stage 2D dynamics evaluations
- `results/stage2d1/processed/formal_gates.json`: G69–G72 adjudication
- `results/stage2d1/continuous_runs/gate_audit.json`: 1k/5k/10k M-gate, norm, and BS audit
- `results/stage2d1/manifests/`: historical integrity manifests
