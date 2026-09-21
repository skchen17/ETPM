# Stage 2D.4 — Action-Conditioned Persistent Memory Access

> **Can current candidate actions condition persistent-memory retrieval strongly enough to make history × action behavioral memory form reliably, without changing the memory law or turning counterfactual reads into writes?**

## Executive result

- **G88:** FAIL
- **G89:** NOT_RUN_BY_PROTOCOL
- **G90:** PASS
- **G91:** FAIL
- **G92:** FAIL
- **G93:** FAIL
- **G94:** NOT_RUN_BY_PROTOCOL
- **G95:** NOT_RUN_BY_PROTOCOL
- **G96:** NOT_RUN_BY_PROTOCOL

**Formal outcome: Outcome B — Helps but Does Not Stabilize.** A1 increased the formal healthy count from **1/8 to 2/8**, but missed the preregistered 6/8 stabilization gate. It created strong action-specific queries and upstream read interaction, yet query swap/neutralization had essentially no causal behavioral effect. A1 should therefore **not be retained as a validated architecture change**. Formal F→M handoff remains closed, and no F/M-law redesign is justified.

## Protocol and experimental details

The frozen Stage 2D core used `gamma=.50`, `rho_fast=.97`, `rho_slow=.9995`, hidden dimension 32 and 8×8 F/M matrices. External delta write, readout-conserving F→M transfer, decay, persistent H recurrence, NULL/SELF_OUTPUT semantics, and protected evaluator training were unchanged. Candidate branches were ephemeral: they performed no write, consolidation, decay, clock advance, or H/F/M commit. A0–A3 used only the original observed-consequence objective; no latent z, correct-action target, memory target, habit label or oracle state was supplied. A4 exactly used the frozen 50-step C1 paired observational warmup and then zero auxiliary weight.

Development used 2 matched independent runs per arm. Formal inference used 8 matched independent initialization/stream pairs per arm, 1500 updates, batch 16, episode lengths 4/6/8, and checkpoints at [0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500]. Each checkpoint audit formed state over 32 noisy episodes with 16 paired replicates, then evaluated novel surface contexts. The independent trained run—not replicate rows—was the statistical unit. Healthy required action-TV≥.10, |IHA|≥.10, |BS|≥.10 and CFA>0.

## Architecture and fairness

A1 adds two bias-free 8×8 action-to-query maps (exactly 128 trainable parameters). A3 adds one bias-free 32×4 downstream projection (exactly 128), so parameter matching is exact. A2 adds 16 parameters. The analytical F+M matrix-vector read costs **256 FLOPs/candidate**; A1 adds **256 FLOPs/candidate** for its two query projections and requires one candidate branch per contemplated action. Parameter matching and compute matching are explicitly not conflated. Profiler totals below are operation-accounted estimates, not hardware-cycle claims.

| Arm | Total params | Protected-head trainable params | Candidate FLOPs | Total training FLOPs |
|---|---:|---:|---:|---:|
| A0 | 27817 | 26357 | 29137 | 82306809000 |
| A1 | 27945 | 26485 | 29409 | 82385625000 |
| A2 | 27833 | 26373 | 29171 | 82316337000 |
| A3 | 27945 | 26485 | 29490 | 82459497000 |
| A4 | 27817 | 26357 | 29137 | 84669059550 |

## Formal arm results

| Arm | Healthy | Partial | Shortcut | Interpretation |
|---|---:|---:|---:|---|
| A0 legacy | 1/8 | 4/8 | 3/8 | original objective |
| A1 conditioned query | 2/8 | 6/8 | 0/8 | small gain, not stabilization |
| A2 conditioned gate | 2/8 | 2/8 | 4/8 | same healthy count as A1 |
| A3 capacity control | 2/8 | 2/8 | 4/8 | same healthy count as A1 |
| A4 paired warmup | 3/8 | 1/8 | 4/8 | best count, still <6/8 |

The controls prevent attributing the modest 1-run A1 improvement specifically to query-dependent retrieval: A2 and A3 each also reached 2/8 and 2/8. A4 reached 3/8, so the evidence is most consistent with a **mixed architecture/training problem with no validated A1-specific benefit**.

## Candidate retrieval and causal tests

Mean A1 action distances were DqF=0.904, DqM=0.872, DrF=0.262, DrM=0.389. Mean factorial read interaction rose from A0 0.0011 to A1 0.1093; A1 exceeded its matched A0 in 8/8 runs. Healthy/nonhealthy A1 read-interaction means were 0.1660/0.0904, satisfying G90 descriptively and by the frozen matched direction criterion.

However, query interventions were behaviorally inert. Across A1 runs, mean IHA changes were swapped -0.0018, both-neutral +0.0005, shared -0.0008, F-neutral +0.0006, and M-neutral -0.0000. Only 0/2 healthy A1 runs met the preregistered joint harm criterion, far below 6/8. Thus G91 failed: explicit query diversity was learned, but behavior still depended mainly on the inherited/downstream path.

## Failure taxonomy and interaction timing

A0 taxonomy was `{'F1_state_formation_failure': 6, 'healthy': 1, 'F2_stored_but_unused': 1}`; A1 was `{'healthy': 2, 'F1_state_formation_failure': 4, 'F2_stored_but_unused': 2}`. F2 increased from 0.125 to 0.250 of all runs while F1 changed from 0.750 to 0.500. This is not a stored-but-unused reduction; G92 failed.

At endpoint, normalized candidate-read interaction increased from A0 0.013 to A1 0.167. This moves a descriptive interaction upstream, but temporary-H/fusion behavior did not show finite causal dependence on the explicit query because swap/neutralization was inert. G93 therefore failed. First full healthy-interaction checkpoint lists were A0 `[1000]` and A1 `[1500, 1500]`; no stable A1 timing advantage is established.

## Conditional experiments and stopping rules

G88 failed, so the ≥24-run expanded A0/A1 cohort was not run (G89). Because both G88 and G91 were required, formation/persistence/revision/selectivity were not rerun (G94). G95 memory mediation was prohibited because G94 was not available. The A1 success condition for the 1k/5k/10k continuous regression was absent, so G96 was not run. These are protocol-governed missing results, not silent omissions.

## Answers to the 33 required questions

1. **A0 healthy rate:** 1/8 (12.5%).
2. **A1 healthy rate:** 2/8 (25.0%).
3. **A2 same benefit?** Yes in healthy count (2/8), so A1 is not specific.
4. **A3 same benefit?** Yes (2/8), despite no action in its memory query.
5. **A4 better?** A4=3/8, above A1=2/8 but still unstable.
6. **A1 ≥6/8?** No; G88 failed.
7. **Expanded replication?** Not run by protocol after G88 failure.
8. **Initialization sensitivity reduced?** Not established; 6/8 A1 runs remained nonhealthy.
9. **Stream sensitivity changed?** Not adjudicated without the crossed expanded cohort.
10. **Different qF/qM?** Yes; mean DqF/DqM=0.904/0.872.
11. **Different rF/rM?** Yes; mean DrF/DrM=0.262/0.389.
12. **Memory-level history×action interaction?** Yes descriptively and matched (G90 PASS), but not behaviorally causal.
13. **Earlier than legacy?** Descriptively at candidate read, not as a causally supported behavioral circuit.
14. **Query swap harms behavior?** No meaningful replicated harm; G91 failed.
15. **Neutral/shared harms behavior?** No meaningful replicated harm.
16. **F-query contribution:** mean IHA change under F-neutralization +0.0006; negligible.
17. **M-query contribution:** mean IHA change under M-neutralization -0.0000; negligible.
18. **Is M-alone necessity required?** **No; it is explicitly not required.**
19. **F2 reduced?** No; 0.125→0.250.
20. **F1 reduced?** 0.750→0.500, but this partly shifted failures into F2 and did not stabilize behavior.
21. **Upstream replaces/supports downstream?** It appears upstream numerically but does not causally support the downstream behavioral interaction.
22. **Only extra parameters?** Cannot exclude generic capacity: A3 matched A1 at 2/8.
23. **Only extra compute?** Cannot credit compute; A2/A3 controls and inert query interventions rule out an A1-specific compute claim.
24. **Gradual formation recovered?** Not run; prerequisite gates failed.
25. **D500 persistence recovered?** Not run.
26. **Revision recovered?** Not run.
27. **Predictive selectivity recovered?** Not run.
28. **Formation-window F/M mediation replicated?** Not run.
29. **Continuous 10k worsened?** Not evaluated because A1 did not meet the success prerequisite.
30. **Best diagnosis:** mixed architecture/training problem, with **no validated benefit from candidate-conditioned retrieval**.
31. **Retain A1?** No, not as a validated default; keep only as an experimental branch.
32. **Reopen formal F→M handoff?** No.
33. **Evidence to modify F/M law?** No.

## Integrity and interpretation limits

Stage 2D.4 protocol tests passed **16/16**. Candidate interventions changed no persistent tensor, parameter, consolidation event, decay event, or clock. Historical Stage 2C/2D/2D.1/2D.2/2D.3 assets passed **2418/2418** hash checks with zero changes. The full repository suite passed **220/224**; four inherited integrity-test design failures treat later-stage additions or the intentionally cumulative README as historical mutation, and none was waived or edited. Null results and conditional non-runs are retained. These toy diagnostics do not establish human-like memory, consciousness, unlimited information capacity, or a general causal-memory mechanism.
