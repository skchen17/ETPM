# Stage 2D.7 — Frozen Evaluator Compatibility and State-Projection Bottleneck

> **Do failed ET-RCM runs contain useful history information in H that is poorly visible to the protected evaluator's frozen state projection, and can changing only this representation-to-readout compatibility recover conditional behavior without changing the memory law?**
>
> **失败 ET-RCM 中是否已有有用的 H 历史信息，只是 frozen state projection 读不到？仅修复这种兼容性、而不改记忆律，能否恢复条件行为？**

## Executive finding

**No replicated frozen-coordinate-bottleneck explanation.** A finite, healthy-scale, output-aligned *state-main activation injected after* the frozen `W_H` rescues conditional behavior in all 46 failed C0 models across three cohorts, and removing the projected state-main destroys behavior in all 26 available healthy models. This establishes that the state-projection interface is a powerful causal lever. It does **not** establish that failed H already contains a sufficiently strong usable history representation: failed/F2 H history separation is far smaller than healthy, projection gain and SVD coordinate signatures do not replicate, frozen-upstream refitting of `W_H, b_H` rescues 0/8 in either independent failed cohort, and the selected training adaptation is healthy in only 1/8 formal seeds—exactly the C0, parameter-matched generic and action-only control rates. The best current characterization is **Outcome C: state formation dominates, with a necessary and sufficient *injected* state-main interface signal**. This is not a verdict that F/M storage or consolidation law has failed.

All thresholds, negative outcomes and stopped branches are retained. Independent unit: **one trained model**, never one history pair or neuron.

## 1. Exact implementation and frozen provenance

The actual C0 candidate path is persistent `(H,F,M)` → legal candidate action event → temporary H → pooling (`latent_slots=1`) → `evaluator_H_input` → `s = W_H H` (`fusion_H_projection`) → addition of frozen action projection and bias → SiLU → frozen 4-logit output. `W_H` is 32×32; action embedding has width 8. The action candidate branch is ephemeral: no intervention changes persistent H/F/M or either clock. The Stage 2D.7 baseline reused the exact Stage 2D.3 C0 procedure (1500 observed-only updates, batch 16, episode lengths 4/6/8, 1000-step protected evaluator pretraining, gamma=.50, rho_F=.97, rho_M=.9995, checkpoints 0/25/50/100/200/300/500/750/1000/1500). A zero-step comparison of the new compatibility wrapper and original trainer was bit-identical; all 24 new C0 training summaries record unchanged evaluator hashes and 100% original objective.

Historical C0: 24 Stage 2D.3 checkpoints, 12 healthy / 12 nonhealthy (6 F1, 6 strict F2). Independent prior C0: 24 Stage 2D.6 checkpoints, 7 healthy / 17 nonhealthy (16 F1, 1 strict F2). New Stage 2D.7 C0: **8 initialization seeds 24101–24108 × 3 streams 25101–25103 = 24**, eval seed 26601, 7 healthy / 17 nonhealthy (14 F1, 3 strict F2). The eight frozen historical primary healthy and eight shortcut IDs were inherited from `results/stage2d6/processed/frozen_rule.json`. New cohort membership was never adjusted to obtain more healthy cases. Healthy = TV≥.10, |IHA|≥.10, |BS|≥.10, CFA>0. F2 is *per-run* nonhealthy with max(H,M probe)≥.75 and |BS|<.10; it does **not** mean the F2 group-average H probe is ≥.75.

See [Stage 2D.6 report](STAGE2D6_CONDITIONAL_INTERACTION_GENERATION_RESULTS.md), [frozen protocol](../docs/STAGE2D7_PROTOCOL.md), [training summaries](../results/stage2d7/confirmatory/training_summaries.json), and [model-level summary](../results/stage2d7/processed/summary.json). All Stage 2C–2D.6 assets stayed unchanged: 2,764 tracked-file SHA-256 matches, zero missing/changed/extra files in [verification](../results/stage2d7/manifests/historical_final_verification.json).

## 2. H→state visibility and frozen `W_H` geometry

For each model, the 2×2 matched history/action cell means were formed first. `ΔH` is the action-averaged history contrast at `evaluator_H_input`; `Δs` is the contrast at `fusion_H_projection`. We evaluated `D_H = ||ΔH||`, `D_S = ||Δs||`, `G_proj = D_S/(D_H + 1e-12)`, and `V_H = G_proj²`. Values below are **means of per-model values**, not pooled neurons/histories:

| Cohort | Phenotype (n) | D_H | D_S | G_proj | E_top8 | E_bottom8 |
|---|---:|---:|---:|---:|---:|---:|
| Historical | healthy (12) | 3.474 | 2.264 | .681 | .229 | .269 |
| Historical | F1 (6) | .369 | .223 | .583 | .249 | .215 |
| Historical | F2 (6) | .447 | .223 | .502 | .206 | .340 |
| Stage 2D.6 | healthy (7) | 2.792 | 1.912 | .715 | .251 | .292 |
| Stage 2D.6 | F1 (16) | .594 | .344 | .587 | .289 | .214 |
| Stage 2D.6 | F2 (1) | .925 | .645 | .698 | .330 | .246 |
| New Stage 2D.7 | healthy (7) | 3.828 | 2.462 | .658 | .238 | .240 |
| New Stage 2D.7 | F1 (14) | .524 | .310 | .565 | .223 | .265 |
| New Stage 2D.7 | F2 (3) | .770 | .465 | .588 | .216 | .197 |

The first-layer hook satisfied `Δs = W_H ΔH` with maximum absolute error 3.58×10⁻⁷ across all 72 models; per-action maximum was 4.17×10⁻⁷. SVD reconstruction error was ≤5.96×10⁻⁷. All 72 `W_H` matrices had numerical rank **32/32**: there is no literal nullspace. We used top/bottom singular-vector *gain* subspaces, k=1/2/4/8/16, and singular-weighted visibility, never called a small-gain direction a nullspace.

The large `D_S` difference largely follows the large `D_H` difference. Historical F2 gain (.502) is below healthy (.681), but the single Stage 2D.6 F2 gain (.698) nearly equals healthy (.715), and the new F2 gain (.588) is only modestly below healthy (.658). Frozen paired gain directions are 6/8 historical, 6/7 prior independent, and **4/7 new**; neither independent cohort supplies eight healthy models. In historical F2, bottom-8 energy is high (.340 vs healthy .269), but the direction reverses in both independent cohorts (.246 vs .292 and .197 vs .240). High-gain top-8 paired directions are 6/8, 2/7, 4/7. Thus G123 and G124 **FAIL**. In strict F2, the group-average H probes are .688/.625/.667 (historical/prior/new) and M probes .677/.813/.646; the per-run F2 threshold can be supplied by M rather than H. “Upstream probe high” is not evidence that the *particular matched history difference in candidate H* is strong.

## 3. Finite state-main sufficiency and necessity

**Restoration design.** Eight frozen historical healthy checkpoints define a healthy logit-interaction template and median projected state-main norm. On each failed model, the local target Jacobian adjoint maps that *output-space* template into that model's own 32-dimensional state-projection coordinates—no raw cross-model neuron swap. The preregistered λ=1 intervention adds `+δ_s/2` to both candidate-action cells of history A and `−δ_s/2` to both cells of history B, at `fusion_H_projection` *after* `W_H`. Its norm is 1.00× the healthy median, below the 1.25× cap. Action main and factorial interaction at the hook change by <10⁻⁵; the native frozen action addition, SiLU and output map then generate interaction. λ=.25/.5/1 were retained. Negative controls were a same-norm random state direction (fixed seed), same-norm lowest-output-sensitivity direction, action-main pattern, and common shift; a direct fusion-post aligned interaction injection was a positive ceiling.

| Failed cohort | Aligned state reaches TV/IHA/|BS| thresholds | Strict G125 (all control margins) | Random reaches thresholds | Low-sensitivity / action / common reaches thresholds |
|---|---:|---:|---:|---:|
| Historical (12) | 12/12 | 12/12; frozen primary 8/8 | 6/12 | 0/12 each |
| Stage 2D.6 (17) | 17/17 | 17/17; first 8/8 | 7/17 | 0/17 each |
| New Stage 2D.7 (17) | 17/17 | 17/17; first 8/8 | 7/17 | 0/17 each |

Historical failed native mean IHA=.014, |BS|=.011; aligned restoration IHA=1.023, |BS|=.665. New failed native IHA=.016, |BS|=.012; aligned IHA=1.045, |BS|=.664. The fusion-post positive ceiling reaches thresholds in 12/12 historical and 17/17 new. **Random directions sometimes rescue** (6/12 and 7/17); the stronger claim that random can *never* rescue is false. The aligned direction nevertheless exceeds every negative control by the frozen .025 IHA and |BS| margins on every failed model. G125 **PASS**, with the crucial caveat that the injection **supplies a history contrast not demonstrated to exist in failed H**.

**Healthy deletion.** Each healthy model's own projected history state-main was scaled around its midpoint by β∈{0,.25,.5,.75,1}; β=0 makes the two history midpoints equal while retaining the native action path. Equal-norm common and action-*pattern* perturbations were controls (the native action effect itself was not deleted). The strict deletion gate passes 12/12 historical, 7/7 prior and 7/7 new; each independent healthy cohort has only seven models, so G126 is **PASS with replication-size limitation**, not a claimed 8/8. New-cohort mean response is dose-ordered:

| β | post-fusion interaction norm | IHA | |BS| |
|---:|---:|---:|---:|
| 0 | .293 | .039 | .030 |
| .25 | .697 | .209 | .165 |
| .5 | 1.209 | .386 | .296 |
| .75 | 1.739 | .554 | .409 |
| 1 | 2.269 | .712 | .502 |

The historical cohort shows the same monotone pattern (IHA .022/.201/.378/.548/.708). These experiments establish state-main signal necessity *at this interface* and finite sufficiency of an injected signal, not pre-existing H compatibility. As G123/G124 jointly fail, H-space low→high/high→low norm-preserving rotation and its random controls were **not run** under the specified conditional gate; G127 is NOT_RUN, not a negative rotation outcome.

## 4. Frozen-upstream refit ceiling: R0–R4

Eight frozen historical shortcut checkpoints and eight independent new failed checkpoints underwent the same refit ceiling. Every R arm reused one frozen-upstream feature cache per checkpoint: 200 reset episodes, batch 16, lengths cycling 4/6/8, legal observed context/action/outcome only; 500 AdamW updates per arm, batch 256, lr=.001, zero weight decay. Training seed 37101 and train-parity color/shape pairs never overlap each checkpoint's independent held-out eval seed and novel-parity surface pairs. The test flips simulator latent/support metadata without changing observed records and verifies identical features/targets. R0: original; R1: only `W_H, b_H` (1056 trainable parameters); R2: only action weights `W_A` (256); R3: only 32 fusion biases; R4: 1056-parameter *generic downstream* dense residual adapter after SiLU. Protected slices, upstream, action embedding and final output weights remain exact. No z, correct-action, oracle-state, memory or BS labels enter fitting.

| Cohort, n=8 each | Arm | Held-out healthy | Mean TV | Mean IHA | Mean |BS| |
|---|---|---:|---:|---:|---:|
| Historical | R0 | 0/8 | .039 | .015 | .012 |
| Historical | R1 state-only | **0/8** | .063 | .007 | .005 |
| Historical | R2 action-only | 0/8 | .045 | .014 | .011 |
| Historical | R3 bias-only | 0/8 | .023 | .015 | .012 |
| Historical | R4 generic | 0/8 | .023 | .011 | .009 |
| New | R0 | 0/8 | .037 | .020 | .014 |
| New | R1 state-only | **0/8** | .045 | .017 | .014 |
| New | R2 action-only | 0/8 | .055 | .018 | .014 |
| New | R3 bias-only | 0/8 | .042 | .020 | .014 |
| New | R4 generic | 0/8 | .037 | .015 | .010 |

R1 neither restores held-out conditional behavior nor clearly outperforms R2/R3/R4. G128 **FAIL (0/8 + 0/8)**. The narrow inference is that this observed-only 500-update readout refit did not recover the required circuit under the tested synthetic-world split; it does not prove H contains zero information under every possible probe or optimizer.

## 5. Training trajectories and conditional rescue training

At every 0/25/50/100/200/300/500/750/1000/1500 checkpoint we recorded D_H, D_S, G_proj, top/bottom SVD energy, H/M probes, fusion-post interaction, IHA and BS. Historical healthy at step 500 already has mean H probe .745 but G_proj .558 versus failed .568; at 750, healthy D_H=1.310, D_S=.917, IHA=.306 and G_proj=.667 versus failed .793/.437/.026/.561. The prior independent cohort at step 500 has healthy IHA=.144 but only a modest gain contrast (.628 vs .579); by 750 its IHA=.282 while gain is nearly equal (.587 vs .574). Run-level `D_S` half-rise precedes/reaches IHA=.10 in 8/12, 3/7 and 4/7 healthy models; G_proj timing is 10/12, 5/7 and 6/7. No stable “H information first, readout compatibility later” sequence replicated. G129 **FAIL**; the supported pattern is joint growth of H separation and projected signal in the subset that becomes healthy.

Because G125 and limited G126 supported finite interface causality, we ran the authorized training branch without changing F/M law. Development held **two identical seed pairs per configuration**: C0; C1 (LR_{WH}/LR_{upstream}=.01,.1,.25); C2 progressive freeze (T_w=100,300,500); C3 H-coordinate low-rank residual (r=2,4); C4 matched generic post-SiLU low-rank (r=2,4) plus dense 1056-parameter C4 to match C1/C2; C5 action-projection-only. That is 13 configurations ×2 =26 runs. No configuration reached 2/2 healthy. Only C1 at .01× and C2 at (T_w=100) reached 1/2; all C3, C4, C5 controls reached 0/2. The rule frozen before complete development evaluation selected the higher-IHA 1/2 C1 at .01× (absolute state-projection LR (10^{-5})), **not** a post hoc per-seed optimum. C2 progressive freeze showed no stability gain; C3 was ineffective in this grid.

Formal used eight *new* matched seed pairs 29101–29108 × 30101–30108, eval seed 31601, for the selected C1 and C0/C4-dense/C5 controls. All use original observed consequence CE plus original .001 H penalty, no paired warmup or behavior/latent labels:

| Arm | Healthy / 8 | Mean TV | Mean IHA | Mean |BS| | Mean CFA |
|---|---:|---:|---:|---:|---:|
| C0 protected legacy | 1/8 | .121 | .091 | .069 | −.001 |
| C1 `W_H, b_H` adaptation | **1/8** | .070 | .087 | .070 | .008 |
| C4 1056-param generic downstream | 1/8 | .055 | .038 | .033 | ~0 |
| C5 action projection only | 1/8 | .058 | .064 | .048 | .005 |

G130 **FAIL**: the selected state-specific arm is 1/8, not ≥6/8, and does not exceed control healthy rates. In accordance with the preregistered gates, expanded 24-run rescue crossing (G131), mechanism preservation in a stabilized selected model (G132), N=0/1/4/16/64 formation and D/R/selectivity rerun (G133), formation-window F/M clamps (G134), and 1000/5000/10000 tick comparison (G135) were **not run**. This is not evidence those properties are absent; the required stabilized model did not exist. Status files are saved in each deferred result family.

## 6. Formal gate adjudication

| Gate | Status | Decisive model-level evidence |
|---|---|---|
| G123 Projection visibility deficit | **FAIL** | Gain directions 6/8 historical, 6/7 prior, 4/7 new; independent healthy counts 7 and 7; F2 gain deficit not stable. |
| G124 Coordinate visibility signature | **FAIL** | Top-8 directions 6/8, 2/7, 4/7; bottom-8 F2 tendency reverses independently. |
| G125 State-projection rescue | **PASS** | Strict aligned-vs-all-control gate 8/8 in each frozen primary set; all 46 failed across cohorts pass. |
| G126 State-projection necessity | **PASS, limited n** | 8/8 historical primary; 7/7 in each independent cohort; dose response, but no independent eight-healthy set. |
| G127 H-coordinate causality | **NOT_RUN** | Conditional G123/G124 prerequisite failed. |
| G128 Frozen-upstream `W_H, b_H` refit | **FAIL** | R1 0/8 historical and 0/8 new, no held-out superiority over controls. |
| G129 Compatibility emergence | **FAIL** | D_S onset precedes behavior in 8/12, 3/7, 4/7; gain onset not a stable separate phase. |
| G130 Training stabilization | **FAIL** | Selected C1 healthy 1/8; C0/C4/C5 each 1/8. |
| G131 Basin robustness | **NOT_RUN** | G130 failed; no expanded crossing. |
| G132 Mechanism preservation | **NOT_RUN** | No stabilized selected model. |
| G133 Behavioral dynamics recovery | **NOT_RUN** | G130/G132 prerequisites unmet. |
| G134 Peripheral memory mediation | **NOT_RUN** | G133 prerequisite unmet. |
| G135 No continuous regression | **NOT_RUN** | No selected stabilized model for 10k ticks. |

## 7. Required question-by-question answers

1. Healthy/F1/F2 `||ΔH||`: historical 3.474/.369/.447; prior 2.792/.594/.925; new 3.828/.524/.770.
2. Corresponding `||W_H ΔH||`: 2.264/.223/.223; 1.912/.344/.645; 2.462/.310/.465.
3. Gain distinction: descriptive historical difference, **not** stable independent separation (G123 fail).
4. F2 “high upstream/low projected”: some per-run H or M probes high by construction, but candidate H separation is also much smaller; not a clean frozen-projection-only signature.
5. Healthy high-gain concentration: no replicated top-8 enrichment.
6. F2 low-gain concentration: historical hint reverses in both independent cohorts.
7. State-main-only restoration: yes, 12/12 + 17/17 + 17/17 reach behavioral thresholds, all first-eight gates pass.
8. Native SiLU generation: yes, the hook receives no factorial interaction; post-fusion interaction and IHA rise through untouched action/SiLU/downstream path.
9. Random state direction: **not uniformly ineffective**—6/12 historical and 7/17 new reach thresholds, though aligned restoration beats it by the frozen margins in every failed run.
10. Action-main restoration: does not produce comparable rescue, 0/12 and 0/17 threshold passes in historical/new failed cohorts.
11. Healthy state-main removal: damages post-I/IHA/BS in 12/12, 7/7, 7/7.
12. Partial attenuation: clear monotone dose response (new IHA .039→.712 from β=0→1).
13. Failed H low→high rotation: not run after G123/G124 failed.
14. Healthy H high→low rotation: likewise not run.
15. Frozen-upstream state-only refit: no, R1 0/8 + 0/8.
16. Generalization to novel histories/surfaces: no restoration on independent seed and held-out novel-parity surfaces.
17. Action-only refit: also 0/8 + 0/8; R1 is not meaningfully superior.
18. Generic adapter refit: also 0/8 + 0/8; no state-specific ceiling advantage.
19. Earliest compatibility difference: no replicated gain onset; projected-signal/behavior separation emerges around steps 500–750 in successful subsets, alongside H growth.
20. H probe before visibility: historical H-probe hint exists, but the sequence does not replicate as a distinct `G_proj` stage.
21. F1: mainly weak formation of a history-contrasting candidate H (small `D_H`), not established F/M-law failure.
22. F2: a stored/probe signal can coexist with weak candidate-H separation; strict F2 is **not** proven a pure coordinate mismatch.
23. Protected evaluator bottleneck: this stage does not confirm frozen representational coordination as the dominant bottleneck.
24. State-only training healthy rate: selected C1 1/8, identical to C0.
25. Progressive freeze: C2 development only 1/2 at (T_w=100), 0/2 at 300/500; no stability claim.
26. Low-rank H adapter: C3 rank 2/4 each 0/2 healthy in development.
27. Selected rescue ≥6/8: no, 1/8.
28. Expanded 24-run rescue replication: not authorized by G130.
29. Initialization sensitivity: no observed reduction; formal healthy rate unchanged at 1/8.
30. Compatibility mechanism preserved after stabilization: not testable without stabilization.
31. Gradual formation recovery: not run by G133 prerequisites.
32. D500 persistence recovery: not run.
33. Revision recovery: not run.
34. Predictive-versus-noise selectivity recovery: not run.
35. F/M formation-window mediation: not reopened by G134 prerequisite.
36. 10k continuous behavior: not measured in this stage; no regression/safety claim.
37. Best bottleneck category: **state formation dominates**, with causal sensitivity at the later state-main interface; not a verified frozen-coordinate mismatch.
38. Fusion architecture redesign: **not justified by these data**; avoid adding bilinear interaction just to force positive results.
39. F/M memory-law redesign: **not justified**; the law remained frozen and this experiment did not isolate it as the cause.
40. Formal F→M causal handoff: **not reopened**; G130/G131/G132/G133/G134 prerequisites are unmet.

## 8. Scientific limits, controls and reproducibility

The injected state-main vector is deliberately output-aligned using a target local Jacobian and a healthy norm; it can be behaviorally sufficient even when failed H does **not** carry the same usable contrast. Random state directions occasionally cross the threshold, which further limits coordinate-specific interpretation. Independent C0 cohorts have only seven healthy runs each; no healthy runs were cherry-picked or training altered to fill eight. The refit negative result applies to the frozen observed-only 200-episode/500-update protocol and novel-parity evaluation—not every conceivable readout fit. Among the 40 follow-up questions, branches marked NOT_RUN remain unmeasured, never silently interpreted as failures. No neural state is called consciousness, causal memory, human-like cognition or unlimited capacity.

The development control was amended before complete evaluation to include a 1056-parameter C4 matched to C1/C2; an early healthy action control and positive-ceiling orientation were corrected before final gate adjudication, then all 48 older-cohort audits were rerun. These changes and the preliminary logs are documented in the protocol; no historical frozen result was modified. The dedicated Stage 2D.7 mathematical/integrity suite passes **9/9**; the full project suite still has the same seven older snapshot-manifest failures (manifests from earlier stages treat subsequently added Stage 2D.6 assets as changes), not new Stage 2D.7 mathematical failures. The Stage 2D.7 historical SHA-256 verification independently passes 2,764/2,764. The initial bare `pytest` invocation also lacked the established `PYTHONPATH=.:src:experiments` and caused four collection errors; the reported full-suite result uses the corrected invocation.

Machine-readable, model-level records: [gates](../results/stage2d7/processed/formal_gates.json), [projection/phenotype/trajectory summary](../results/stage2d7/processed/summary.json), [development selection](../results/stage2d7/processed/frozen_development_selection.json), [historical manifest](../results/stage2d7/manifests/historical_baseline_sha256.json). Separate families under `results/stage2d7/` hold projection visibility, state-projection SVD, F1/F2, state-main rescue, healthy destruction, R0–R4 refit, 10-step trajectories, controls, training rescue, and conditional-branch status. Reproduction entry points are `experiments/stage2d7_train_baseline.py`, `stage2d7_run_diagnostics.py`, `stage2d7_run_refits.py`, `stage2d7_run_development.py`, `stage2d7_run_formal.py`, `stage2d7_process.py`, and `stage2d7_adjudicate.py`.

**Next experiment:** investigate how legal experience is represented in the *candidate H history contrast* (including distributed/recurrent topology and context-conditioned state formation) before proposing evaluator unfreezing or F/M-law edits. A causal intervention that makes the pre-existing H contrast sufficient, not merely injects a new post-`W_H` contrast, would be required to revive the coordinate-mismatch hypothesis.
