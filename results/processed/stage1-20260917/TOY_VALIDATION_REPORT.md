# ET-RCM Stage-1 toy validation report

> **Can a model with its own internal time transform transient experience into persistent computational state through repeated internal use, while keeping self-repetition from becoming new evidence?**
>
> **一个具有自身内部时间的模型，能否让短暂经历因为后续内部计算中的反复使用而自然转化为持久计算状态，同时避免把自己的重复思考误当成新的证据？**

## Execution scope and verdict

- Run: `stage1-20260917`; raw records: `results/raw/stage1-20260917/toy_records.parquet`.
- Records: 2104 total, 720 final episode summaries, 8 seeds.
- Stage-2 authorization: **FALSE**.
- This is a small synthetic validation. It is not evidence of human-like memory,
  consciousness, infinite information capacity, or an intervention-validated
  causal memory in a language model.

## Gates

| gate | result |
|---|---|
| G1_math | PASS |
| G2_repeated_exposure | PASS |
| G3_repeated_use | PASS |
| G4_idle_cognition | PASS |
| G5_no_self_evidence | PASS |
| G6_memory_revision | PASS |

## Negative criteria

| criterion | triggered |
|---|---|
| N1_single_memory_equivalent | True |
| N2_uniform_equivalent | False |
| N3_matched_compute_equivalent | True |
| N4_unknowable_confidence_rises | False |
| N5_revision_failure | False |

N1/N3 are decision-relevant null results: the single persistent matrix remains
competitive on this narrow retention endpoint, and deterministic idle compute
is exactly equivalent to moving the same transitions to query time. Idle
execution reduces future latency but does not improve matched-compute accuracy
or final state in Toy 5. These outcomes are retained rather than protocol-tuned.

## Required questions

1. **Readout conservation?** Yes within the unit-test tolerance when G1 passes;
   maximum accumulated Toy-2 conserving drift was `6.859e-16`.
   The non-conserving replay control drifted by `0.3366`.
2. **Repeated consolidation analytic formula?** `test_repeated_consolidation_analytic_solution`
   checks `F_K q=(1-gamma)^K F_0 q`; G1 result: `True`.
3. **Repeated real exposure improves retention?** `True`.
   Curve: 1: 0.0502, 2: 0.1172, 4: 0.2519, 8: 0.4679.
4. **Repeated internal use at matched exposure improves retention?**
   `True`. Curve: 0: 0.0110, 1: 0.0560, 2: 0.0945, 4: 0.1554, 8: 0.2323.
5. **Same input/different future utility learns different lifetime?** In the
   deliberately small differentiable policy toy, useful/unused retention was
   `0.1716` / `0.0013`. This is proof of implementation,
   not a general learned-importance result.
6. **Do idle ticks enable reasoning?** Toy-4 accuracy changed from
   `0.5000` to `1.0000`.
   This shows iterative latent computation, not autonomous human-like thought.
7. **Idle vs matched query-time compute?** Accuracy curves and final states were
   matched; maximum final-state difference was `0.000e+00`.
   Only readiness/latency placement differed.
8. **Does idle consolidation improve slow retention?**
   `True`; curve: 0: 0.0000, 2: 0.0927, 4: 0.1603, 8: 0.2457, 16: 0.3149.
9. **Unknowable-bit self-confidence amplification?** `False`.
   Accuracy: 0: 0.5013, 1: 0.5013, 2: 0.5013, 4: 0.5013, 8: 0.5013, 16: 0.5013, 32: 0.5013; confidence: 0: 0.5000, 1: 0.5000, 2: 0.5000, 4: 0.5000, 8: 0.5000, 16: 0.5000, 32: 0.5000.
10. **Can real new evidence correct old memory?** Best `P(new)` was
    `0.8417`; G6: `True`. Mean by new
    exposures: 1: 0.1719, 2: 0.2678, 4: 0.3388.
11. **Query-dependent better than uniform?** Negative-equivalence criterion N2
    is `False`. Full: 0: 0.0110, 1: 0.0560, 2: 0.0945, 4: 0.1554, 8: 0.2323;
    uniform: 0: 0.0199, 1: 0.0203, 2: 0.0208, 4: 0.0216, 8: 0.0230.
12. **Fast/slow better than single persistent memory?** Not established when N1
    is triggered (`True`). Single:
    0: 0.3159, 1: 0.3157, 2: 0.3156, 4: 0.3153, 8: 0.3146.
13. **Independent value of endogenous time?** It provides pre-query readiness
    and a slot for consolidation, but Toy 5 does not show a matched-compute state
    or accuracy advantage. Independent value is therefore **not established**.
14. **Enough evidence for Stage 2?** **FALSE**.
15. **Where is evidence insufficient?** The narrow single-memory baseline is not
    defeated; matched idle/query-time compute is equivalent; Toy 3 uses a tiny
    synthetic context policy; and no language-scale learned dynamics or causal
    state intervention has been tested.

## Temporary vs persistent structure

Toy-9 slow retention was `0.0498` for the episode-local rule and
`0.2457` for the cross-episode repeatedly used rule. Inputs had the
same form; only later task-use statistics differed.

## Artifacts

- Raw: `results/raw/stage1-20260917/toy_records.parquet`
- Summary: `results/processed/stage1-20260917/toy_summary.parquet` and `results/processed/stage1-20260917/summary.json`
- Figures: `results/processed/stage1-20260917/figures`

All negative and null arms remain in the raw Parquet. The protocol was not
modified in response to these results.
