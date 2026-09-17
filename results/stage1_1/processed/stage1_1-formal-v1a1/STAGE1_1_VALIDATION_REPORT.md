# ET-RCM Stage 1.1 Complete Validation Report

> **Can a learned recurrent state autonomously decide what to revisit, thereby allocating limited persistent-memory lifetime preferentially to information that remains useful, and can computation performed between external events change how later events are processed rather than merely shifting compute earlier in time?**

> **一个可学习的持续状态模型，能否自主决定接下来重新访问什么，从而把有限的长期记忆寿命优先分配给未来仍有用途的信息；同时，外部事件之间发生的内部计算，是否能够真正改变模型随后吸收新事件的方式，而不只是把相同计算提前执行？**

Formal run: `stage1_1-formal-v1a1`. Generated: 2026-09-17T15:19:55.489853+00:00. Stage 2 authorization: **False**.

## Protocol and implementation details

The frozen protocol, split salt, formal seeds (2101/2102/2103), learned-rate choice, all failed/null outcomes, and the disclosed A1 correction were preserved. The selected learning rate was 1e-3 from development seeds 1101/1102. Formal memory/graph/unknowable training used 700/900/700 optimizer steps, batch size 64, AdamW, full BPTT, gradient clipping 1.0, and no detach interval. The full model used H=4x128 and F/M=32x32, with gamma=.12, rho_fast=.97, rho_slow=.9995, eta=.6. All prediction heads, queries, access strengths, and recurrent dynamics were learned; external delta write and readout-conserving transfer remained fixed laws.

Inputs contained only the current structured event. Query events carried a key but no answer value; NULL ticks carried no event. Final associative-memory tests scrubbed H before query. The no-history schema/signature/lesion audit and the full 20-test suite passed.

## Gate adjudication

| Gate | Result | Frozen measurements |
|---|---:|---|
| G7_learned_query | **FAIL** | alignment_margin=-0.7875; accuracy_margin=0.3125; retention_endpoint_change=0.1611 |
| G8_selective_persistence | **PASS** | accuracy_margin_vs_B2=0.0590; retention_margin_vs_B2=0.7652 |
| G9_usage_over_frequency | **FAIL** | useful_over_unused=-0.1844; useful_over_uniform=0.3134 |
| G10_idle_reasoning | **FAIL** | accuracy_K16_minus_K0=0.0026 |
| G11_interleaved_time | **FAIL** | best_schedule_accuracy_margin=0.0000 |
| G12_no_self_evidence | **PASS** | accuracy_K32=0.5032; confidence_inflation=-0.0121; ece_degradation=-0.0089 |
| G13_revision | **PASS** | best_balanced_cell_new_probability=0.9793; reference_unrelated_retention=0.3009 |

## Experiment details and results

### A — One exposure, then 0/1/2/4/8 genuine downstream retrieval uses without restating the value; 64 interference events; H scrub before final query. Measures learned query alignment, access, transfer, slow retention, and accuracy.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| A1_gamma_zero | 0.0181 ± 0.0214 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.4060 ± 0.0183 (n=3 seeds) |
| A2_uniform_transfer | 0.0243 ± 0.0262 (n=3 seeds) | 0.8383 ± 0.0320 (n=3 seeds) | 0.4038 ± 0.0178 (n=3 seeds) |
| A3_equal_timescales | 0.0479 ± 0.0307 (n=3 seeds) | 0.8473 ± 0.0052 (n=3 seeds) | 0.4532 ± 0.0296 (n=3 seeds) |
| A4_no_null_dynamics | 0.1250 ± 0.0473 (n=3 seeds) | 0.8179 ± 0.0060 (n=3 seeds) | 0.0619 ± 0.0087 (n=3 seeds) |
| A5_random_query | 0.0375 ± 0.0150 (n=3 seeds) | 0.1427 ± 0.0171 (n=3 seeds) | 0.3491 ± 0.0306 (n=3 seeds) |
| A6_frozen_H | 0.0139 ± 0.0060 (n=3 seeds) | 0.8733 ± 0.0124 (n=3 seeds) | 0.0233 ± 0.0011 (n=3 seeds) |
| A7_nonconserving | 0.3653 ± 0.0710 (n=3 seeds) | 0.8375 ± 0.0117 (n=3 seeds) | 0.4783 ± 0.0125 (n=3 seeds) |
| B0_no_memory_mlp | 0.0208 ± 0.0104 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0191 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0208 ± 0.0104 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 0.0590 ± 0.0120 (n=3 seeds) | 0.1284 ± 0.0181 (n=3 seeds) | 0.4389 ± 0.0246 (n=3 seeds) |
| B3_uniform | 0.0243 ± 0.0262 (n=3 seeds) | 0.5870 ± 0.0098 (n=3 seeds) | 0.3649 ± 0.0204 (n=3 seeds) |
| B5_no_idle | 0.1181 ± 0.0332 (n=3 seeds) | 0.7956 ± 0.0688 (n=3 seeds) | 0.3987 ± 0.0123 (n=3 seeds) |
| B6_full | 0.2132 ± 0.0980 (n=3 seeds) | 0.8480 ± 0.0026 (n=3 seeds) | 0.4099 ± 0.0051 (n=3 seeds) |

### B — Eight useful facts are used in tasks, then 32/128/512 distractors arrive. No useful flag is present. B2 matches the F+M matrix-state float count; lesion rows zero F/M after the same history.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B0_no_memory_mlp | 0.0104 ± 0.0180 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0191 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0104 ± 0.0104 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 0.1100 ± 0.0236 (n=3 seeds) | 0.1568 ± 0.0084 (n=3 seeds) | 0.4802 ± 0.0119 (n=3 seeds) |
| B3_uniform | 0.0394 ± 0.0100 (n=3 seeds) | 0.4665 ± 0.0111 (n=3 seeds) | 0.4091 ± 0.0203 (n=3 seeds) |
| B5_no_idle | 0.1019 ± 0.0203 (n=3 seeds) | 0.7344 ± 0.1194 (n=3 seeds) | 0.4519 ± 0.0198 (n=3 seeds) |
| B6_full | 0.1331 ± 0.0231 (n=3 seeds) | 0.8227 ± 0.0108 (n=3 seeds) | 0.4666 ± 0.0240 (n=3 seeds) |
| B6_full_memory_lesion | 0.0104 ± 0.0104 (n=3 seeds) | 0.8227 ± 0.0108 (n=3 seeds) | 0.1482 ± 0.0164 (n=3 seeds) |

### C — A useful fact has 1/2 exposures and eight downstream uses; an unused competitor has 8/16/32 exposures. Neither event carries a utility marker.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B0_no_memory_mlp | 0.0208 ± 0.0208 (n=3 seeds) | NA | 0.0191 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0174 ± 0.0060 (n=3 seeds) | NA | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 0.0278 ± 0.0046 (n=3 seeds) | NA | 0.4514 ± 0.0038 (n=3 seeds) |
| B3_uniform | 0.0249 ± 0.0020 (n=3 seeds) | NA | 0.3969 ± 0.0064 (n=3 seeds) |
| B5_no_idle | 0.2593 ± 0.0350 (n=3 seeds) | NA | 0.4581 ± 0.0114 (n=3 seeds) |
| B6_full | 0.3785 ± 0.0297 (n=3 seeds) | NA | 0.4849 ± 0.0080 (n=3 seeds) |

### D — Graph paths are supplied as edge events, followed by a query and 0/1/2/4/8/16 NULL ticks. Training support is length 1–6; formal evaluation includes 7–8 as unseen longer paths.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B1_gru | 0.4901 ± 0.0119 (n=3 seeds) | NA | 0.5204 ± 0.0133 (n=3 seeds) |
| B5_no_idle | 0.4857 ± 0.0197 (n=3 seeds) | NA | 0.5047 ± 0.0020 (n=3 seeds) |
| B6_full | 0.5483 ± 0.0333 (n=3 seeds) | NA | 0.5580 ± 0.0476 (n=3 seeds) |

### E — Compares internal ticks before versus after a matched interruption event. External events and transition counts are identical; behavior, not state distance, adjudicates the gate.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B0_no_memory_mlp | 0.0174 ± 0.0060 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0191 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0035 ± 0.0060 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 1.0000 ± 0.0000 (n=3 seeds) | 1.0000 ± 0.0000 (n=3 seeds) | 0.9534 ± 0.0042 (n=3 seeds) |
| B3_uniform | 1.0000 ± 0.0000 (n=3 seeds) | 1.0000 ± 0.0000 (n=3 seeds) | 0.9180 ± 0.0070 (n=3 seeds) |
| B5_no_idle | 1.0000 ± 0.0000 (n=3 seeds) | 1.0000 ± 0.0000 (n=3 seeds) | 0.9755 ± 0.0039 (n=3 seeds) |
| B6_full | 1.0000 ± 0.0000 (n=3 seeds) | 1.0000 ± 0.0000 (n=3 seeds) | 0.9567 ± 0.0051 (n=3 seeds) |

### F — Moves the same query/NULL block before versus after 128 distractors, preserving facts and total step count.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B0_no_memory_mlp | 0.0104 ± 0.0104 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0192 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0139 ± 0.0159 (n=3 seeds) | 0.0000 ± 0.0000 (n=3 seeds) | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 0.0174 ± 0.0159 (n=3 seeds) | 0.0573 ± 0.0207 (n=3 seeds) | 0.4458 ± 0.0110 (n=3 seeds) |
| B3_uniform | 0.0260 ± 0.0052 (n=3 seeds) | 0.3737 ± 0.0166 (n=3 seeds) | 0.3861 ± 0.0216 (n=3 seeds) |
| B5_no_idle | 0.0434 ± 0.0210 (n=3 seeds) | 0.6545 ± 0.1264 (n=3 seeds) | 0.4367 ± 0.0240 (n=3 seeds) |
| B6_full | 0.0816 ± 0.0314 (n=3 seeds) | 0.7232 ± 0.0235 (n=3 seeds) | 0.4412 ± 0.0166 (n=3 seeds) |

### G — Moves eight learned reasoning ticks across a matched interruption after a five-edge graph query.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B1_gru | 0.5046 ± 0.0079 (n=3 seeds) | NA | 0.5133 ± 0.0104 (n=3 seeds) |
| B5_no_idle | 0.5026 ± 0.0118 (n=3 seeds) | NA | 0.5039 ± 0.0025 (n=3 seeds) |
| B6_full | 0.5846 ± 0.0184 (n=3 seeds) | NA | 0.5563 ± 0.0479 (n=3 seeds) |

### H — A jointly trained learned binary head sees either a knowable parity task or an independently sampled target bit absent from all input. Accuracy, confidence, entropy, ECE and Brier are swept over 0–32 ticks.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B6_full | 0.7513 ± 0.0016 (n=3 seeds) | NA | 0.7535 ± 0.0042 (n=3 seeds) |

### I — Full 5x5x5 old-exposure/new-exposure/new-reuse grid. Old memory is used four times before genuinely contradictory evidence; unrelated retention is retained for every cell.

| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |
|---|---:|---:|---:|
| B0_no_memory_mlp | 0.0417 ± 0.0180 (n=3 seeds) | NA | 0.0191 ± 0.0006 (n=3 seeds) |
| B1_gru | 0.0104 ± 0.0180 (n=3 seeds) | NA | 0.0251 ± 0.0017 (n=3 seeds) |
| B2_single_persistent | 0.9929 ± 0.0092 (n=3 seeds) | NA | 0.9415 ± 0.0048 (n=3 seeds) |
| B3_uniform | 0.9999 ± 0.0001 (n=3 seeds) | NA | 0.9276 ± 0.0057 (n=3 seeds) |
| B5_no_idle | 0.9959 ± 0.0043 (n=3 seeds) | NA | 0.9521 ± 0.0020 (n=3 seeds) |
| B6_full | 0.9720 ± 0.0080 (n=3 seeds) | NA | 0.9238 ± 0.0072 (n=3 seeds) |

## Training diagnostics and resource accounting

The combined formal artifact contains 322,944 episode-level evaluation rows and 486 training-log rows. Every row stores the model/seed/run revision; evaluation rows also store persistent-state bytes, parameter count, compute budget, H/F/M norms, query vector, access/transfer fields where applicable, loss, accuracy and calibration fields. Checkpoints are separate tensor artifacts and are not embedded in Parquet.

## Scientific conclusion

Only the frozen gates determine the conclusion. A failed gate remains failed; no threshold was tuned after inspecting formal outcomes. Toy-scale success would not imply human-like memory, consciousness, infinite capacity, autonomous human-like thought, or causal memory. Stage 2 is authorized only when the machine-readable adjudication says TRUE.

## Direct answers to the 15 required questions

1. **Task-meaningful autonomous query? No under the frozen definition.** B6 improved task accuracy, but target-key alignment was -0.7875 below the random-query arm; G7 failed.
2. **Does query change with active state? Yes, but that is insufficient.** Mean per-dimension query variance was 0.0287, so it was not a fixed direction. Its changes did not satisfy semantic target alignment.
3. **Did future use strengthen slow retention through the learned path? Partly.** B6 reuse-8 minus reuse-0 slow retention was 0.1611, but the failed alignment component prevents the stronger learned-query claim.
4. **Did fast/slow beat matched single persistent memory under pressure? Yes at the frozen 512-distractor endpoint.** Accuracy and retention margins were 0.0590 and 0.7652; G8 passed.
5. **Was that only extra state or compute? Not in the adjudicating comparison.** B2 uses two persistent matrices matching F+M floats and the same runner transition budget/training examples. Exact parameter/state-byte counts remain in every row; unmatched B0/B1 are not used to decide G8.
6. **Which won, frequent-useless or rare-useful? The frequent useless trace.** Useful-minus-useless retention was -0.1844; G9 failed even though B6 beat uniform on useful retention.
7. **Did the learned core reason usefully on NULL events? No.** K16 minus K0 accuracy was 0.0026, below the frozen 0.10 margin, including a separately reported unseen-longer stratum.
8. **Did idle-before-event differ behaviorally from event-before-idle? No.** The best B6 schedule accuracy margin was 0.0000. State distance alone was not counted.
9. **Does endogenous time remain compute/latency scheduling here? Yes.** N8 triggered; this protocol found no independent matched-compute behavioral advantage.
10. **Did learned autonomous dynamics amplify self-evidence? No detected amplification.** Unknowable K32 accuracy was 0.5032, confidence changed -0.0121, and ECE changed -0.0089; G12 passed and N9 did not trigger.
11. **Can real new evidence revise memory? Yes in preregistered balanced cells.** Best eligible mean P(new) was 0.9793; the complete grid is retained, and G13 passed.
12. **Fair-baseline result?** GRU/no-memory arms were near chance on delayed associative recall; matched B2 was weaker than B6 at the frozen high-pressure endpoint; uniform transfer was weaker on learned-reuse accuracy. However B2/uniform remained extremely strong on immediate/simple revision settings, and the frequency-utility test favored raw frequency. No broad dominance claim is warranted.
13. **Which mechanism was necessary?** Consolidation, query dependence and evolving H affected learned-reuse behavior in the registered ablations, while non-conserving replay performed strongly only by violating the core law. No component was shown universally necessary because the main learned-query and endogenous-time gates failed.
14. **What can be removed without behavioral loss?** For the failed idle/interleaving tasks, NULL dynamics added no registered benefit; globally removing it is not justified because the no-idle arm was weaker on learned reuse. No globally redundant module was established.
15. **Proceed to decoder LM Stage 2? No.** `STAGE2_LANGUAGE_MODEL_AUTHORIZED = FALSE`; G7, G10 and G11 (and secondary G9) failed, with N6 and N8 triggered. No Stage-2 training was run.
