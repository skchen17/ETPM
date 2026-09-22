# Stage 2D.5 — Conditional Interaction Transmission Audit

> **When candidate-specific history×action information is already present in persistent-memory reads but has little behavioral effect, where along the read→active-state→fusion pathway is that conditional interaction lost, and can restoring only the broken transmission step stabilize behavioral memory?**

> **当 persistent-memory read 中已经存在 candidate-specific 的 history×action 信息，但它几乎不影响行为时，这种 conditional interaction 究竟在 read→active state→fusion 的哪一步丢失？如果只修复这个断裂的传递步骤，能否稳定建立 behavioral memory？**

## Executive result

- **G97:** FAIL
- **G98:** FAIL
- **G99:** FAIL
- **G100:** PASS
- **G101:** PASS
- **G102:** FAIL
- **G103:** FAIL
- **G104:** FAIL
- **G105:** NOT_RUN_BY_PROTOCOL
- **G106:** NOT_RUN_BY_PROTOCOL
- **G107:** NOT_RUN_BY_PROTOCOL
- **G108:** NOT_RUN_BY_PROTOCOL
- **G109:** NOT_RUN_BY_PROTOCOL

**Outcome D — Upstream Interaction Is Mostly Epiphenomenal.** A1's upstream read interaction is real as a measured activation but not a replicated behavioral mediator. Eight of eight failed confirmatory A1 checkpoints can be rescued by output-aligned `fusion_post` interaction injection, and 8/8 healthy legacy checkpoints are destroyed by removing their native interaction there. Yet restoring at any earlier candidate-read, integration or pre-fusion node rescued **0/8**. This confirms the protected downstream causal node already found in Stage 2D.3; it does **not** causally localize a repairable read→H or fusion transmission edge. Architecture rescue was not authorized. F/M laws remain frozen and the formal handoff stays closed.

## Frozen protocol, cohorts and provenance

No architecture, memory law, query, gate, evaluator, dimensions, NULL/SELF_OUTPUT semantics, training objective or curriculum changed. The original Stage 2D.4 A0/A1 endpoints each contributed 8 matched runs. A new confirmatory cohort trained **16 A1 models** under the identical frozen Stage 2D.4 observed-only protocol (1500 updates, batch 16, episode lengths 4/6/8, gamma .50, rhoF .97, rhoM .9995, protected evaluator). It yielded **2 healthy, 4 partial and 10 shortcut** endpoints; the original A1 cohort had **2 healthy and 6 partial**. Thus only **4/24 A1** models are healthy. An independent historical legacy C0 set supplied 24 endpoints (12 healthy, 12 shortcut) to reach the diagnostic minimum of 8 healthy and 8 failed overall. It is clearly labelled as legacy, never passed off as A1 replication. Statistical units are independently trained models; 16 within-run history replicates are averaged before all run-level contrasts.

The primary causal set was fixed by sorted IDs: eight failed confirmatory A1 and eight healthy historical legacy C0. Six failed original A1, all four healthy A1 and six historical F2 controls were audited additionally. `results/stage2d5/processed/selection.json` records exact run IDs and checkpoint paths; `confirmatory/training_summaries.json` records all 16 seeds, endpoints, hashes, per-checkpoint losses, norms, parameter counts and protected-evaluator checks. All candidate branches and interventions were read-only; H/F/M, clocks, model parameters and protected evaluator hashes were verified unchanged. The audit reimplemented the exact frozen candidate computation and unit-tested its logits/read equality to numerical precision.

## Full 2×2 interaction flow

Each row is the mean absolute factorial interaction norm after per-run 2×2 aggregation. The same histories and candidate actions are used at every node; none of these magnitudes alone establishes causal mediation.

| Node | Original A1 (8) | Confirmatory A1 (16) | Combined A1 (24) |
|---|---:|---:|---:|
| q_F | 0.0093 | 0.0068 | 0.0077 |
| q_M | 0.0120 | 0.0074 | 0.0090 |
| r_F | 0.0431 | 0.0319 | 0.0357 |
| r_M | 0.0662 | 0.0412 | 0.0495 |
| normalized_F | 0.6518 | 0.6366 | 0.6416 |
| normalized_M | 0.2315 | 0.3132 | 0.2860 |
| gated_read | 0.5992 | 0.5801 | 0.5865 |
| read_projection | 0.3394 | 0.3175 | 0.3248 |
| candidate_update | 0.1204 | 0.0816 | 0.0945 |
| temporary_H | 0.1204 | 0.0816 | 0.0945 |
| evaluator_H_input | 0.0213 | 0.0110 | 0.0144 |
| fusion_input | 0.0213 | 0.0110 | 0.0144 |
| fusion_pre | 0.0143 | 0.0077 | 0.0099 |
| fusion_post | 0.2725 | 0.1649 | 0.2007 |
| logits | 0.4461 | 0.2193 | 0.2949 |
| probabilities | 0.1441 | 0.0705 | 0.0950 |
| policy_score | 0.4029 | 0.1922 | 0.2625 |

The first descriptive absolute reduction is `gated_read→read_projection` (mean ratio **0.539**), followed by `temporary_H→evaluator_H_input` (**0.170**). Neither distinguishes healthy from failed in two adequately powered A1 cohorts. `candidate_update→temporary_H` conserves absolute factorial interaction here; its normalized ratio can fall because common residual H increases state-main magnitude. Likewise `evaluator_H_input→fusion_input` has absolute ratio **1.000** (concatenation preserves the interaction) but normalized ratio **0.080** because the action main effect enters the denominator. These are not evidence of destroyed information. G97 therefore fails; A1 has only two healthy runs in each independent cohort and cannot furnish ≥6/8 same-architecture healthy/failed matched replications.

## F/M mixing, residual scaling and operator anatomy

Mean raw F/M interaction cosine was original/confirmatory **0.450/0.185**; negative in **2/8** and **6/16**. Native gate interaction was smaller than both F-only and M-only in only **1/8** and **1/16**. F-only, M-only, equal and norm-matched-sum conditions did not give replicated behavioral rescue, so G98 fails.

For failed A1, residual α=1→2 changed mean IHA from **0.110→0.110** (original) and **0.039→0.039** (confirmatory). No moderate α≤2 rescue replicated; G99 fails. The internal audit separates H/read/event projections, core preactivation/SiLU, gate preactivation/activation, candidate update, residual output, evaluator input, H/action fusion projections, additive merge, fusion preactivation/SiLU and postactivation. The complete per-node metrics and ratios are retained machine-readably.

## Finite path restoration and destruction

For each node, the source was the **same-architecture healthy cohort's mean logit-interaction vector**. Its direction was mapped into the target model's local coordinates by that model's output-Jacobian adjoint; amplitude was calibrated to the within-architecture healthy median native interaction. This is declared output-coordinate alignment, not a raw neuron swap across models. Each 2×2 intervention changed only the factorial interaction component; state/action main effects were preserved. λ was .25/.5/1 and the induced factorial norm never exceeded the matched healthy norm (below the 1.25 cap). Norm-matched state-only, action-only, random-interaction, shuffled-sign and generic-shift controls were run at the same scales. Healthy destruction removed/reversed the native interaction at each node with main-effect controls. A preliminary single-logit alignment pilot is retained under `path_restoration/pilot_unaligned/` but **excluded from all formal gates**.

The **earliest rescue node is `fusion_post`**: 8/8 primary failed A1 recovered healthy TV/IHA/BS, versus state 0/8, action 0/8, random 0/8, shuffled 0/8 and generic 0/8. Every earlier node was **0/8** under the same norm cap. The **earliest necessary node is `fusion_post`**: removal destroyed 8/8 healthy legacy behaviors, while state/action-main removal had negligible effect. All four healthy A1 checkpoints also lost behavior at this node, but four is below an independent 8-run A1 necessity cohort. Rescue and necessity coincide at the protected post-fusion node, not at a demonstrated broken transmission edge.

## F2 and bottleneck adjudication

F2-like stored-but-unused controls numbered **9** across cohorts (`{'confirmatory_query': 3, 'historical_legacy': 6}`). Their mean gated-read interaction was **0.266**, and fusion-post interaction **0.195**. Fusion-post injection rescued **9/9**, but earlier-node restoration rescued **0/9**. Legacy F2 often had weak interaction already near memory access, whereas some A1 F2 had strong reads but weak behavior; no single replicated F2 break edge was identified, so G104 fails.

G102 fails because no specific internal read→H node passed the 6/8 finite-restoration requirement. G103 fails because post-fusion rescue alone does not show that a healthy interaction was lost *inside* fusion: preactivation interaction is small and the SiLU can generate new state×action interaction from main effects. The primary result is **upstream interaction mostly epiphenomenal for behavior**, not a licensed fusion or integration redesign. G100 PASS plus G102/G103 FAIL prohibits architecture modification. Hence G105–G109 are `NOT_RUN_BY_PROTOCOL`: no rescue architecture, parameter-matched control, expanded 24-run cohort, behavioral dynamics, F/M mediation or continuous regression was run.

## Direct answers to the 34 required questions

1. **Query interaction:** mean qF/qM = 0.0077/0.0090.
2. **Raw F read:** 0.0357.
3. **Raw M read:** 0.0495.
4. **Gated read:** 0.5865.
5. **ΔH:** 0.0945.
6. **Temporary H:** 0.0945.
7. **Fusion input:** 0.0144.
8. **Fusion post:** 0.2007.
9. **Output probability interaction:** 0.0950.
10. **First clear attenuation:** descriptively gated-read→read-projection; no proven causal collapse edge.
11. **Healthy vs failed ratio:** weak and inconsistent within A1; only 2 healthy per cohort, below the 6/8 gate.
12. **F/M cancellation?** Not systematic; G98 FAIL.
13. **Gate mixing destroys interaction?** Not systematically and alternatives do not rescue behavior.
14. **Residual H dilution?** Normalized ratio falls, absolute interaction is conserved; no causal dilution claim.
15. **Moderate scaling rescue?** No; G99 FAIL.
16. **Earliest rescue:** `fusion_post`.
17. **Earliest necessity:** `fusion_post` in 8 healthy legacy; corroborated in 4 healthy A1, not an 8-run A1 gate.
18. **Same window?** Yes, post-fusion; this is the known downstream causal node, not a proven upstream transmission break.
19. **F2 first break?** No shared edge; legacy often weak at access, A1 may have high read interaction yet little behavioral use.
20. **Read→H bottleneck?** Not causally localized; G102 FAIL.
21. **Fusion bottleneck?** Not proven; G103 FAIL despite strong post-fusion positive control.
22. **Distributed bottleneck?** Not established; pre-fusion finite restorations were null.
23. **Interaction restoration beats controls?** Yes at `fusion_post`: 8/8 versus ≤0/8 controls; not at earlier nodes.
24. **Architecture modification authorized?** No.
25. **Minimal rescue selected?** None; selecting one would violate the stopping rule.
26. **Rescue architecture ≥6/8?** Not run by protocol.
27. **Parameter-matched control?** Not run; no architecture rescue was authorized.
28. **Transmission repaired?** Not demonstrated.
29. **Behavioral dynamics recovered?** Not run.
30. **F/M mediation recovered?** Not run.
31. **Continuous running worsened?** Not evaluated for a new architecture; none was trained.
32. **Most consistent category:** upstream interaction mostly epiphenomenal, with **no replicated causal transmission bottleneck**.
33. **Modify F/M law?** No.
34. **Reopen formal F→M handoff?** No.

## Integrity and limits

Stage 2D.5 tests passed **16/16**. The dedicated historical manifest verified **2445/2445** tracked frozen Stage 2C–2D.4 assets with zero changes. In the final committed tree, the full repository suite passed **234/240**; six inherited integrity-test failures compare cumulative later-stage files or README changes against older frozen snapshots. The sixth appears after Stage 2D.5 files become Git-tracked, because the older Stage 2D.4 snapshot test includes any subsequently tracked `stage2d*` files. None was modified or waived. No endpoint was reclassified to make a gate pass. A downstream output-aligned intervention is a causal test of that node, not proof that persistent memory itself was the source of the recovered information. These synthetic findings make no claim of human-like memory, consciousness or unlimited capacity.
