# ET-RCM Stage 2C.1 — Action–Outcome Binding and Behavioral Interface Diagnostic

> **Before asking whether ET-RCM can form persistent behavioral memory, can the model first use a known latent state to predict different consequences for different candidate actions, and can that information control behavior through the existing persistent-state pathway?**

> **在判断 ET-RCM 能否形成持久行为记忆之前，首先验证：当正确的潜在状态已经已知时，模型能否根据不同候选行为预测不同未来结果，并通过现有 persistent-state pathway 让这些差异真正控制行为？**

Formal ladder: **G48 PASS (8/8), G49 PASS (8/8), G50 PASS (8/8)**. Historical Stage 2C remains Outcome C, G42–G47 FAIL; none of its gates or files were revised.

## 1. Stage 2C failure diagnosis

Frozen Stage 2C reported mean direct action-branch TV ≈ 2.59e-04, while history sensitivity was larger. Its consequence head consumed pooled H after an action event; it did not explicitly receive a candidate-action input. This motivated an interface diagnostic, not a retrospective memory-law failure claim. Source: `reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md`.

## 2. Level 0 oracle latent

The four z×action cells are exactly balanced per batch (64 items, 16/cell). Correct action emits outcome 0; incorrect action uniformly samples outcomes 1–3. The model sees oracle z as a learned context embedding and action as a separate candidate input; it is trained only on observed consequence CE. No correct-action target, memory state, or reward label enters L0. Two development seeds (7101–7102) preceded threshold freezing; eight disjoint formal training seeds (7201–7208) follow it.

## 3. Action-branch TV

Late-concat L0 action-TV: 0.999657 ± 0.000106; modulation: 0.999735 ± 0.000074. The late-concat mean is about 3860× the frozen Stage 2C TV. Values are per-seed means over latent A/B, not pooled episodes pretending to be seeds.

## 4. ΔQ and p(y=0) margins

Late-concat ΔQ_A: 0.999607 ± 0.000242; ΔQ_B: 0.999706 ± 0.000111. Both positive in 8/8. All four forecast probability vectors, TV, bidirectional KL, JS, entropy differences, p0 differences, and interaction vectors are retained in `results/stage2c1/action_interventions/direct_forecast_metrics.json`.

## 5. Entropy-policy behavior

The original Stage 2C policy softmax(−future entropy/0.35) now chooses the latent-matching action with mean probability 0.958592 ± 0.000165; entropy-policy behavioral separation 0.917183 ± 0.000330. This is a policy computed from forecasts, not supervised action selection.

## 6. Q-policy diagnostic

The diagnostic softmax(p(y=0)/0.35) chooses correctly with 0.945636 ± 0.000016; BS 0.891273 ± 0.000031. It is not substituted for the original entropy policy in formal gate comparisons. At this fixed temperature it has lower seed variance but also a lower mean correct-action probability.

## 7. Current vs action-conditioned consequence head

The historical Stage 2C linear head receives only pooled H; action must survive five recurrent context/action steps. The new late-concat head receives [H;candidate-action embedding] directly, whereas additive modulation feeds H+W_a a to a small nonlinear head. Both L0 forms bind successfully. Thus direct action injection is sufficient; these experiments do not isolate whether old failure arose from its topology, joint-training gradients, recurrent dynamics, or their combination. Late-concat was chosen before formal downstream tests for simplicity, not tuned on formal outcomes.

## 8. Training-step scaling

| Steps | late-concat TV / CE | modulation TV / CE |
|---:|---:|---:|
| 0 | 0.040682 / 1.3985 | 0.051739 / 1.3821 |
| 100 | 0.990890 / 0.5578 | 0.995562 / 0.5552 |
| 500 | 0.998796 / 0.5534 | 0.998657 / 0.5535 |
| 1000 | 0.999290 / 0.5519 | 0.999217 / 0.5520 |
| 3000 | 0.999657 / 0.5502 | 0.999735 / 0.5502 |

CE is exact expected consequence CE under the four balanced world cells, computed from fixed four-way forecasts. Training checkpoints are at 0/100/500/1000/3000; 500-step L0 success argues that lack of steps alone is not the oracle-interface blocker, but Stage 2C joint lifetime training is a different optimization problem.

## 9. Gradient diagnostics

Mean pre-clipping gradient norms (late-concat; H/source, action branch, consequence head):

| Step | H/source | action | head |
|---:|---:|---:|---:|
| 100 | 0.01558 | 0.01324 | 0.61712 |
| 500 | 0.00625 | 0.00394 | 0.36623 |
| 1000 | 0.00532 | 0.00383 | 0.42415 |
| 3000 | 0.00176 | 0.00145 | 0.21750 |

L0 action gradients are nonzero during learning; near-convergence gradients should be read alongside low loss. L3 joint-training action gradients were not instrumented, so starvation there remains untested. Gradient magnitude is diagnostic, not causal intervention evidence.

## 10. Level 1 oracle M

Fixed QR-derived M_A/M_B are 8×8, unit Frobenius norm and orthogonal; neither is a correct-action nor outcome one-hot. H0, F0=0 and present event=None are identical. Only M changes, through inherited q_M→r_M→normalized/gated read→H; M is not an output-head input. L1 action-TV 0.999171 ± 0.000176; entropy BS 0.916929 ± 0.000311; H1 difference 9.709334 ± 1.468679; mean forecast TV under M-read clamp 0.499586 ± 0.000088. One NULL tick is a short-horizon pathway diagnostic, not persistence.

## 11. Oracle M swap

For every saved L1 checkpoint, a finite intervention swaps M_A↔M_B with the same actions and parameters. Forecast permutation maximum absolute error: 0.00e+00; preference reversal 8/8. Read clamp collapses latent-dependent forecasts, further checking the required route.

## 12. Level 2 history encoder

A small GRU sees only eight past Stage 2C-observable records: color, shape, nuisance, actual action, observed outcome. Its output reshapes to M and enters the same read→H pathway. It receives neither z, correct-action labels, future outcomes, nor a direct output-head bypass. Evaluation uses 64 fresh histories per seed with held-out parity-combination surfaces; L2 action-TV 0.999253 ± 0.000272, entropy BS 0.917179 ± 0.000332. The current probe surface is not modeled in this L2 diagnostic, so novel generalization concerns held-out history surface combinations only.

## 13. History × action interaction

L0 interaction I_HA(y0): 1.999314 ± 0.000212; L1: 1.998342 ± 0.000352; L2 novel: 1.998507 ± 0.000543; L3 endogenous N16: -0.000162 ± 0.000732. At L3 N16, history-TV: 0.004718 ± 0.002293 and action-TV: 0.011601 ± 0.008913. Positive interaction requires the effect of switching action to reverse across A/B history-derived state. TV_H and TV_A are stored separately per seed; they should not be conflated.

## 14. Endogenous ET-RCM rerun

Only after G48–G50 passed, L3 newly trained eight 500-step Stage 2C-style lifetimes with the direct candidate-action head. Same B5 core, event stream, F write/consolidation/decay, AdamW 0.001, batch 16, 4/8/16 episode lengths, H penalty 0.001. Four independent paired evaluation lifetimes per seed; no parameter update during evaluation. This is a new diagnostic, not a re-score of frozen Stage 2C.

At N=16, L3 entropy BS: -0.000206 ± 0.000660; direct action-TV: 0.011601 ± 0.008913. State norms H/F/M: 15.378786 ± 5.175646 / 0.866628 ± 0.065046 / 0.689135 ± 0.172817. Evaluation parameters frozen in all seeds: True.

Formation (N actual experiences):

| Condition | mean BS ± seed SD |
|---|---:|
| 0 | +0.000000 ± 0.000000 |
| 1 | -0.000678 ± 0.000745 |
| 2 | +0.000291 ± 0.001028 |
| 4 | +0.000129 ± 0.000366 |
| 8 | +0.000168 ± 0.000562 |
| 16 | -0.000206 ± 0.000660 |
| 32 | +0.000090 ± 0.000298 |

Persistence (unrelated delay after N=16):

| Condition | mean BS ± seed SD |
|---|---:|
| 0 | -0.000206 ± 0.000660 |
| 10 | -0.000133 ± 0.000555 |
| 100 | -0.000043 ± 0.000287 |
| 500 | -0.000017 ± 0.000081 |

Generalization:

| Condition | mean BS ± seed SD |
|---|---:|
| seen | -0.000206 ± 0.000659 |
| novel | -0.000206 ± 0.000660 |
| hard_ood | -0.000206 ± 0.000661 |

Revision (opposing real experiences after N=16):

| Condition | mean BS ± seed SD |
|---|---:|
| 0 | -0.000206 ± 0.000660 |
| 1 | -0.000009 ± 0.000474 |
| 2 | -0.000007 ± 0.000706 |
| 4 | +0.000106 ± 0.000666 |
| 8 | +0.000135 ± 0.000643 |
| 16 | +0.000003 ± 0.000418 |
| 32 | +0.000079 ± 0.000372 |

Peripheral memory swaps at N=16:

| Condition | mean BS ± seed SD |
|---|---:|
| F | -0.000396 ± 0.000804 |
| M | -0.000236 ± 0.000666 |
| FM | -0.000425 ± 0.000803 |

All four paired raw records, checkpoints and unrounded metrics are preserved. These L3 probes are scoped diagnostics; no old G42–G47 threshold is re-adjudicated, and the shorter L3 grid is not interchangeable with the historical full Stage 2C protocol.

## 15. Gate outcomes G48–G50

G48 PASS 8/8: pre-registered both-latent action-TV≥0.50 and ΔQ_A/B>0. G49 PASS 8/8: oracle M both action-TV≥0.50, entropy BS≥0.30, exact preference-reversing swap. G50 PASS 8/8: legal-history novel both action-TV≥0.50, entropy BS≥0.30, positive interaction. Thresholds and seeds were fixed in `configs/stage2c1_formal.yaml` after two dev seeds and before formal runs. Formal parameters remained frozen during every evaluation.

| Seed | L0 TV | L1 TV | L2 novel TV | M swap | L3 TV N16 | L3 BS N16 |
|---:|---:|---:|---:|---|---:|---:|
| 7201 | 0.99965 | 0.99914 | 0.99933 | True | 0.01205 | -0.00101 |
| 7202 | 0.99948 | 0.99926 | 0.99945 | True | 0.00819 | -0.00006 |
| 7203 | 0.99961 | 0.99921 | 0.99921 | True | 0.00530 | +0.00093 |
| 7204 | 0.99976 | 0.99938 | 0.99944 | True | 0.03146 | -0.00117 |
| 7205 | 0.99963 | 0.99893 | 0.99869 | True | 0.00518 | -0.00003 |
| 7206 | 0.99965 | 0.99931 | 0.99921 | True | 0.01427 | -0.00019 |
| 7207 | 0.99964 | 0.99925 | 0.99957 | True | 0.01254 | +0.00016 |
| 7208 | 0.99984 | 0.99889 | 0.99913 | True | 0.00381 | -0.00028 |

## 16. Failure localization

Joint lifetime action binding remains weak; endogenous F/M formation is not isolated from the interface/training failure. The L0–L2 success establishes conditional interface capacity and legal-history sufficiency. It does **not** establish endogenous F/M formation. If L3 action-TV itself is near zero, even the direct head has not retained oracle binding under joint lifetime training; one cannot uniquely blame the F/M law. Stage 2C's original Outcome C remains intact. This pattern is best labeled a mixed joint-training/interface–memory-chain blocker pending a controlled state-injection audit of the L3 checkpoint.

## 17. Next-stage recommendation

Do not enter language modeling or redesign F/M yet. Next isolate the L3 checkpoint with frozen H and injected oracle M / learned history M, quantify action gradient and read-path survival during lifetime training, then pre-register a controlled training-curriculum comparison. Keep the original entropy policy. Address H stability separately only if nonfinite/large-H trajectories actually dominate; this short-horizon ladder does not justify an H redesign.

## Direct answers to the 18 required questions

1. Yes: L0 distinguishes the two action consequences in all eight seeds.
2. Yes: mean L0 TV 0.999657 versus historical 2.59e-04.
3. Yes: ΔQ_A 0.999607 ± 0.000242, ΔQ_B 0.999706 ± 0.000111.
4. Yes: entropy-policy correct-action probability 0.958592 ± 0.000165.
5. Q-policy works (0.945636 ± 0.000016), with lower seed variance but less decisive mean behavior at the fixed temperature.
6. The frozen original head was nearly action-insensitive in Stage 2C; topology versus optimization is not isolated.
7. Modulation also binds; it is not materially needed over simple late-concat in L0.
8. Both heads bind by 500 steps in oracle L0; more steps improve small residual error, not the old joint-training diagnosis.
9. No persistent L0 action-gradient starvation was observed; L3 joint-training action gradients remain unmeasured.
10. Yes: oracle M changes read/H and forecasts through the inherited pathway.
11. Yes: exact M swap reverses preference in 8/8 seeds.
12. Yes for eight legal observed past records and held-out history surface combinations; not yet proof of spontaneous memory.
13. Yes in L0/L1/L2; interactions are positive and replicated.
14. Mixed joint-training/interface–endogenous-chain failure; a pure F/M blocker is not isolated.
15. The ladder warrants a controlled full-chain diagnostic, which L3 performed; it does not warrant a positive behavioral-memory claim.
16. No F/M architecture change is justified by these data alone.
17. Yes: direct action/consequence conditioning and its joint-training survival require targeted work.
18. No immediate H-stability redesign; monitor H separately after interface diagnostics.

## Artifacts, protocol integrity and limitations

Machine-readable summaries: `results/stage2c1/processed/summary.json`; per-seed raw logs/checkpoints: `oracle_latent/`, `oracle_memory/`, `history_encoder/`, `raw/lifetime/`; direct interventions: `action_interventions/`; curves: `trajectories/`; config/source/historical hashes: `configs/`, `manifests/`. A separately labeled 10-step L3 smoke run is preserved in `raw/l3_smoke_7101/` and excluded from formal aggregates. The historical 883-file Stage 2C inventory is SHA-256 verified after this stage. The 13 new Stage 2C.1 tests pass. The repository-wide suite has one inherited failure: `test_prior_frozen_artifacts_have_no_tracked_edits` compares README to commit `1367110`, whereas the already-committed Stage 2C HEAD `a6540c5` appended 45 README lines; Stage 2C.1 did not edit README or that test. We preserve and disclose this mismatch. L0–L2 are small synthetic sanity tests, not human-like memory, causal memory or language competence. L2 held-out surfaces are histories rather than current probes; L3 shorter evaluation grids do not revise frozen Stage 2C gates.
