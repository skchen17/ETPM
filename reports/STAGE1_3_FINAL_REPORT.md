# ET-RCM Stage 1.3 Final Report

> **Can a continuously running finite-state model integrate streaming evidence, reallocate memory lifetime through internal use, and emit selectively without treating its own output as new evidence?**

> **一个持续运行的有限状态模型，能否整合连续证据、通过内部使用重新分配记忆寿命，并在不把自身输出当成新证据的前提下选择性表达？**

Formal run `stage1_3-formal-v1`; generated 2026-09-18T09:25:56.645555+00:00. All formal seeds used the development-frozen learning rates and thresholds. The B6 threshold required a disclosed diagnostic fallback because the primary development criterion was infeasible; therefore G19 cannot pass by formal sampling luck.

## Outcome

| Gate | Status | Registered measurements |
|---|---|---|
| G18 | **FAIL** | {"correct_emission_rate": 0.50439453125, "margin_vs_B0": -0.0751953125, "sufficient_minus_insufficient": 0.4541015625, "replicate_seed_count": 0} |
| G19 | **FAIL** | {"development_threshold_primary_feasible": false, "precision": 0.5772701424349489, "recall": 0.5, "noise_false_emission_rate_10000": 0.0, "replicate_seed_count": 0} |
| G20 | **FAIL** | {"accuracy_margin_vs_R0": -0.0078125, "useful_emission_margin_vs_R0": -0.0078125, "false_emission_increase_vs_R0": 0.0, "replicate_seed_count": 0} |
| G21 | **FAIL** | {"mean_usage_retention_spearman": 0.14448473882021062, "replicate_seed_count": 1} |
| G22 | **FAIL** | {"b6_self_output_safety_subcriteria_pass": true, "maximum_external_write_count": 0.0, "maximum_mean_memory_relative_increase": 0.0, "maximum_mean_expression_score_increase": 0.0, "nonconserving_control_memory_increase": 0.0} |

Long stream authorized: **False**. Small language-model prototype recommended: **False**.

## Architecture and protocol

The only persistent cognitive state is `(H,F,M)`. External evidence alone uses the delta write; NULL and SELF_OUTPUT never use it. Consolidation transfers the currently accessed fast direction into slow memory while conserving `F+M` before decay. B6 reads `g*r_F + (1-g)*r_M` with one learned scalar gate. Expression is a thresholded observable action and never a halt, reset, solved flag, confidence claim, or new evidence.

Formal seeds: `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`. Each model trained for 1,000 equal-budget steps with batch 64. Development used two disjoint seeds, three learning rates, and 16 thresholds. Full experiment sizes and aggregation definitions are in `STAGE1_3_ANALYSIS_PLAN.md` and the specialized reports.

## Experiment-by-experiment design

| Experiment | Formal construction per model/seed | Primary recorded outcomes |
|---|---|---|
| A — evidence accumulation | 512 query-free 12-event streams; half contain 4 supporting events and half only 2, with support positions randomized among unrelated writes | first/correct/false emission, latency, expression trajectory, H/F/M/read/transfer diagnostics |
| B — pattern discovery | 256 episodes for each of stable, random-frequency, accidental, disappearing, reversing, and shifted 24-event streams | correct/false structured pattern emission and content accuracy |
| C — cross-time association | A→B, 512 unrelated distractors, then B→C and 8 NULL ticks; intact R0/M-only/B6 plus no-memory, M-lesion, gamma-zero, and random-query controls | A→C content accuracy and spontaneous correct emission |
| D — silence under noise | 16 parallel pure-noise streams for 1,000 ticks for all models; B6/B7 additionally 10,000 ticks | false spontaneous emission and score drift |
| E — self-output audit | 512 borderline-supported propositions followed by 0/1/2/4/8/16/32/64 ticks without new evidence | external-write count, memory-strength change, expression change, repeated emission; B7 is the non-conserving control |
| F — revision | 128 episodes with four old-value evidence events followed by 1…8 genuine contradictory new-value events | old/new accuracy, revision emission, memory norm, confidence and latency |
| G — interleaved input | 128 matched episodes; six external events with gaps 0/1/2/4/8/16 either blocked or fully interleaved, with equal transition counts | accuracy, emission, score and final H/F/M norms |
| H — read arbitration | 32 target memories evaluated at 0/32/128/512/2048/8192 accumulated distractors in target and absent contexts | recovery accuracy, useful/false emission, F/M gate, F/M read norms and slow retention |
| I — thought-driven persistence | 256 episodes × 8 once-exposed facts, 32 NULL ticks, 512 distractors, then F cleared | fact-level usage attribution, slow retention and per-episode Spearman association |
| J — long stream | 1e3/1e4/1e5 mixed streams only if every authorization condition passes | otherwise explicitly `NOT_RUN_BY_PROTOCOL` |

## Training summary

| model                        |   final_training_loss |
|:-----------------------------|----------------------:|
| B0_no_persistent             |                0.0003 |
| B1_gru                       |                0.0003 |
| B2_single_persistent         |                0.0019 |
| B3_joint                     |                0.0002 |
| B4_m_only                    |                0.0057 |
| B5_f_only                    |                0.0012 |
| B6_arbitration               |                0.0002 |
| B7_nonconserving_self_replay |                0.0002 |

## Scientific interpretation

Only registered gates support confirmatory claims. A failed gate remains failed. Diagnostic fallback thresholds, structured toy success, persistent state, or nonzero internal dynamics do not imply calibrated epistemic confidence, causal memory, consciousness, autonomous human thought, infinite capacity, or a language model.

## Answers to the 20 required questions

1. **New relations without a query?** Evidence accumulation correct-emission rate was 0.5044; pattern and cross-time controls remain secondary. This is not established by the registered gate.
2. **Does expression score change systematically?** See the accumulated-support trajectory and formal PR tables; score is not interpreted as truth probability.
3. **Emit when sufficient and remain silent when insufficient?** Sufficient-minus-insufficient emission margin was 0.4541; G18=FAIL, G19=FAIL.
4. **Long-noise false emission rate?** B6 at 10,000 ticks: 0.000000.
5. **Does state continue after output?** Yes structurally and in tests; emit does not reset or halt H/F/M, and later transitions run normally.
6. **Does self-output enter external write?** No; maximum cumulative post-output external writes was 0.
7. **Does repeated self-output amplify memory/score?** No systematic B6 amplification was observed: maximum mean memory change=0.0000, score change=0.0000, and external writes=0. The compound G22 nevertheless failed because the B7 pathological-control net increase was 0.0000, below its registered +0.10 validation floor.
8. **Can stored M affect behavior?** B6 high-pressure content accuracy was 0.0273; lesion and read-mode tables show how much was behaviorally accessible.
9. **Is learned arbitration better than historical F+M?** G20=FAIL; accuracy margin=-0.0078, useful-expression margin=-0.0078.
10. **When does it read F versus M?** The registered scalar is fast weight g; its distractor-conditioned means are reported in `MEMORY_ARBITRATION_STAGE1_3.md`. This is descriptive, not a semantic proof.
11. **Natural old-memory revisit without query cue?** The NULL-tick usage metric is nonzero by construction only when learned queries align and fast reads occur; persistence is credited only through G21's retention association.
12. **Does internal reuse predict slow retention?** Mean Spearman=0.1445; G21=FAIL.
13. **Cross-time association?** B6 condition means: [{'condition': 'A_to_B_gap_B_to_C', 'accuracy': 0.0478515625, 'correct_emission': 0.0}, {'condition': 'M_lesion', 'accuracy': 0.046875, 'correct_emission': 0.0}, {'condition': 'gamma_zero_intervention', 'accuracy': 0.041015625, 'correct_emission': 0.0}, {'condition': 'random_query_intervention', 'accuracy': 0.0390625, 'correct_emission': 0.0}].
14. **Can genuine new evidence revise prior output?** Revision audit=False; count-8 accuracy=0.6348, correct revision emission=0.6338.
15. **Excess duplicate output?** Across sampled post-output ticks with no new evidence, B6 repeated the target at rate 0.0780. No permanent already-said database was used; revision after genuine evidence is reported separately.
16. **Is interleaving harder than block input?** B6 matched results: [{'condition': 'block', 'accuracy': 0.0390625, 'emitted': 0.908203125, 'expression_score': 0.836696333864893}, {'condition': 'interleaved', 'accuracy': 0.037109375, 'emitted': 0.9091796875, 'expression_score': 0.8315552881995245}].
17. **Stable at 1e3/1e4/1e5?** Not established; Experiment J is NOT_RUN_BY_PROTOCOL.
18. **Main bottleneck?** The development expression precision/recall separation is a demonstrated bottleneck.
19. **Enough evidence for a small LM prototype?** **False**. No Stage-2 training was started.
20. **What remains toy-scale?** Every positive result here: structured symbols, synthetic streams, fixed small state, short training, and controlled distributions. None establishes consciousness, sentience, human-like autonomous thought, infinite memory/context, or general intelligence.

