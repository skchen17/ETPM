# Stage 2D.3 — Conditional Interaction Anatomy and Minimal Binding Rescue

> **Where does genuine history/state × candidate-action interaction first emerge in successful ET-RCM computations, where does it fail in shortcut runs, and can restoring that specific interaction causally recover behavioral memory?**

> **成功 ET-RCM 中，“历史形成的内部状态 × 当前候选行为”这一条件计算究竟在哪一层首次形成？失败模型在哪一层断掉？如果只恢复这个具体 interaction，能否因果地恢复 behavioral memory？**

## Executive result

- **G79:** PASS
- **G80:** PASS
- **G81:** PASS
- **G82:** PASS
- **G83:** FAIL
- **G84:** FAIL
- **G85:** NOT_RUN_BY_PROTOCOL
- **G86:** NOT_RUN_BY_PROTOCOL
- **G87:** NOT_RUN_BY_PROTOCOL

**Formal outcome: Outcome B* — causal conditional interaction is identified, but no predefined A–D category is exactly satisfied because G83 and/or the training/dynamics chain failed.** Formal F→M handoff is not reopened. A redesign of the F/M law is **not** justified by this stage.

## Protocol and experimental details

The architecture, H/F/M topology, protected evaluator, external write, readout-conserving transfer (`gamma=.50`), decay (`rho_F=.97`, `rho_M=.9995`), NULL and SELF_OUTPUT semantics were frozen. Cohort A contains 32 existing C0 runs (16 healthy, 15 shortcut, 1 partial). Cohort B contains 24 newly trained C0 runs (8 initializations × 3 streams) and is independently labelled only from endpoint TV/IHA/BS. Each checkpoint was evaluated at steps 0, 25, 50, 100, 200, 300, 500, 750, 1000 and 1500 with 16 paired-history replicates per run.

The exact 2×2 factorial hooked pre-action H, action-conditioned incoming H, pooled H, action embedding, fusion input, first preactivation, first SiLU postactivation, pre-logit, logits, normalized probabilities and entropy-policy score. `S`, `A` and `I` were computed separately and first aggregated to the independent-run level. The causal layer was preregistered as `fusion_post`. Frozen cross-model restoration used protected-logit coordinate alignment, not raw hidden swapping. Finite cross-effects used 64 normalized H directions at 0.10/0.25 native scale and the real learned action-embedding contrast.

## Layerwise anatomy and emergence

At step 1500, normalized incoming-H interaction was A healthy/shortcut **0.109/0.024** and B **0.137/0.050**. At `fusion_post` it was A **0.268/0.029** and B **0.269/0.031**. The main-effect-adjusted residual difference remained positive in both cohorts, so G79 is not a relabelled state/action main effect comparison.

The earliest divergent computation location was **incoming_H** in A and **incoming_H** in B. Under the preregistered run-pair and endpoint criterion, the first divergent training checkpoint was **750** and **1000**. M-probe information became visibly different around the same broad 500–750-step window; it did not provide a stable much-earlier causal precursor.

## Frozen causal tests

In Cohort A, failed native IHA averaged **0.009**. A 0.25-scale interaction-only restoration raised it to **0.246**, versus state-only **0.010**, action-only **0.010**, random **0.045**, and shuffled-sign **0.010**. G81 counts were A **8/8** and B **8/8**. Healthy native IHA averaged **0.688**; exact post-fusion interaction removal reduced it to **0.001**, while reversal produced signed IHA **-0.687**. G82 removal/reversal counts were A **8/8/8/8** and B **8/8/8/8**.

## Finite cross-interaction and failure taxonomy

The finite evaluator H×action cross-response was extremely low rank (A rank-1 healthy/shortcut **0.9990/0.9998**; B **0.9990/0.9997**). It did **not** satisfy the required healthy>shortcut replicated direction; G83 therefore failed. This distinguishes the successful trajectory-conditioned interaction from generic local evaluator cross-sensitivity.

Run-level shortcut taxonomy: A `{'F1_state_formation_failure': 11, 'F2_stored_but_unused': 4}`; B `{'F2_stored_but_unused': 6, 'F1_state_formation_failure': 6}`. The dominant failure is therefore state formation plus conditional-use weakness, while the high-probe/low-interaction cases are explicitly retained as F2 stored-but-unused rather than folded into the dominant class.

## Conditional-binding curriculum

Development windows were 50/100/200/300 steps; the frozen selection was **50** steps using the preregistered highest-healthy-count/shortest-tie rule. The paired observational scaffold supplied two legal consequence samples for two actions from the same endogenous state. It supplied no z input, correct-action label, memory label or oracle state; the evaluator stayed protected; auxiliary weight was exactly zero after warmup and during evaluation.

Formal healthy counts were C0 **2/8**, C1 paired **4/8**, C2 shuffled **1/8**, C3 duplicate-compute **2/8**. Thus G84 is FAIL; G85 is NOT_RUN_BY_PROTOCOL.

## Reduced dynamics, mediation and continuous diagnostic

Per-run joint G86 criteria were gradual formation, meaningful D500 persistence, finite revision by R128 and predictive>matched-noise selectivity. Counts among eight C1 formal runs were gradual **NR/8**, persistence **NR/8**, revision **NR/8**, selectivity **NR/8**, with joint G86 **NR/8**. Formation-window F/M/FM read clamps mediated behavior in **NR/8** runs. Continuous diagnostic: **NOT_RUN_BY_PROTOCOL**.

## Required questions

1. **First layer?** {'A': 'incoming_H', 'B': 'incoming_H'} (the action-conditioned incoming-H boundary); the preregistered replicated causal window is `fusion_post`.
2. **Do failed models lack it?** Yes at run level: both cohorts show substantially smaller incoming/fusion interaction.
3. **State formation or conditional use?** Mostly state-formation weakness with an additional conditional-use bottleneck; not storage alone.
4. **Stored-but-unused classification?** F2, defined by high H/M probe but weak downstream interaction.
5. **Distinct from state main effect?** Yes: normalized and state/action-regressed separation replicated.
6. **Distinct from action main effect?** Yes: action embeddings/main effects remain large in shortcut runs while factorial interaction is weak.
7. **First training split?** A=750, B=1000 under the frozen criterion.
8. **M information or interaction first?** M probe differences appear in the same broad window and slightly before the strict interaction gate, not as a robust early predictor.
9. **Amplified after fusion?** Yes in healthy runs; normalized interaction rises from incoming H to first nonlinear fusion.
10. **Never formed or later lost?** Most shortcut runs never form a strong incoming interaction; the taxonomy preserves rare propagation/mixed cases.
11. **Interaction-only rescue?** G81: A=8/8, B=8/8.
12. **State-only rescue?** No; it was far weaker than interaction-only restoration.
13. **Action-only rescue?** No.
14. **Random/shuffled rescue?** No under the preregistered superiority criterion.
15. **Destruction harms healthy behavior?** G82 removal: A=8/8, B=8/8.
16. **Reversal reverses behavior?** A=8/8, B=8/8.
17. **Finite H×action distinguishes groups?** No; the generic finite cross metric was not replicated in the required direction.
18. **Is cross-interaction low dimensional?** Yes, essentially rank-1 locally in both groups; low rank alone is not diagnostic.
19. **Dominant failure taxonomy?** F1_state_formation_failure in A and F2_stored_but_unused in B.
20. **Does paired warmup improve healthy rate?** C1=4/8 versus C0=2/8.
21. **Does binding persist after exit?** Not established by G85.
22. **Do controls exclude compute-only effects?** C1 was directionally above shuffled and duplicate-compute controls, but formal stabilization was not established unless G84 passed.
23. **At least 6/8 healthy?** No.
24. **Gradual formation restored?** NOT_RUN/8.
25. **Persistence restored?** NOT_RUN/8.
26. **Revision restored?** NOT_RUN/8.
27. **Predictive selectivity restored?** NOT_RUN/8.
28. **F/M mediation retained?** G87=NOT_RUN/8.
29. **10k regression?** NOT_RUN_BY_PROTOCOL.
30. **Qualified to reopen F→M handoff?** No.
31. **Reason to modify F/M law?** No. Conditional-computation failure must not be misattributed to the memory law.

## Integrity and interpretation limits

Stage 2D.3 tests passed **12/12**. The full repository suite passed **205/208**; the three failures are inherited integrity-test design issues: the Stage 1.5 test treats the intentionally updated README as frozen, the Stage 2D snapshot includes its own manifest after creation, and the Stage 2D.1 snapshot does not exclude later Stage 2D.2 assets. None was edited or waived. The dedicated pre-stage manifest verified **2327/2327** historical Stage 2C/2D/2D.1/2D.2 files with zero changes.

The independent training run is the statistical unit. Partial runs never enter primary healthy-vs-shortcut causal comparisons. Frozen interventions changed no parameters or stored F/M state. These synthetic diagnostics do not establish human-like memory, consciousness, unlimited temporal capacity, or a general causal-memory mechanism.
