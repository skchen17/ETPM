# ET-RCM Stage 1.2 Final Report

> **Does ET-RCM actually learn functional memory addressing and selective lifetime allocation, and can learned internal recurrence perform necessary sequential computation without converting self-generated activity into new evidence?**
>
> **ET-RCM 是否真正学会了功能性的记忆寻址与选择性的记忆寿命分配；同时，可学习的内部递归是否能够承担必要的逐步计算，而不会把自身产生的内部活动误当成新的外部证据？**

Formal run `stage1_2-formal-v1` uses eight fresh seeds. Stage-1/1.1 conclusions are unchanged. Generated 2026-09-17T18:27:13.676193+00:00.

## Gate summary

| Gate/status | Result | Key measurement |
|---|---:|---|
| G14 Functional addressing | **FAIL** | target-removal accuracy drop 0.0176; loss increase -0.0463 |
| G15 Selective scaling | **FAIL** | high-pressure accuracy margin 0.0050; slope margin 0.0266 |
| G16 Sequential computation | **FAIL** | OOD K8-K1 0.0078 |
| G17 No self-evidence | **FAIL** | knowable gain 0.0007; unknown K64 0.5005 |
| Endogenous time | **ET_STATUS_SUPPORTED** | before-after accuracy 0.2275 |
| Stage-2 Memory LM | **False** | no decoder training executed |
| Stage-2 Continuous Cognition LM | **False** | no decoder training executed |

## Experimental execution

All five baselines received an equal 3-LR × 2-seed development search. An initial 180-step search was preserved as underpowered; amendment A1 extended every architecture equally to 600 steps before formal evaluation. Formal training used the frozen architecture-specific selections, 600 memory steps, and for B6 separate 700/900/600-step autonomous/sequential/no-evidence models. Full BPTT was used. The combined artifacts contain 998,400 evaluation rows and 896 training-log rows.

## Experiment details

### Frozen formal design

| Item | Frozen value |
|---|---|
| Formal seeds | 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208 |
| State dimensions | H=4×128; F/M=32×32 |
| Memory constants | gamma=0.12; rho_fast=0.97; rho_slow=0.9995; eta=0.6 |
| Batch / BPTT | 48 / full |
| Formal training steps | memory=600; autonomous=700; sequential=900; no-evidence=600 |
| Independently selected LRs | B1_gru=0.001, B2_single_persistent=0.001, B3_uniform=0.001, B5_no_idle=0.001, B6_full=0.001 |
| Endogenous-time interference | 512 distractors, frozen from development |
| Formal scaling maximum | 2048 distractors; 32768 was excluded before formal outcomes for the recorded two-GPU budget reason |

### A — Functional query intervention

Every intervention cloned the same memory and active state. The seven conditions changed only the query geometry. The target-projection removal is the adjudicating intervention; signed cosine is descriptive.

| condition | accuracy | loss | target_retrieval_score | downstream_hidden_change | target_projection | absolute_cosine | squared_projection |
|---|---|---|---|---|---|---|---|
| original | 0.1445 | 3.6319 | 0.0778 | 2.8814 | 0.2188 | 0.8412 | 0.7090 |
| parallel | 0.1211 | 3.9055 | 0.0625 | 2.8470 | 0.2188 | 0.8412 | 0.7090 |
| perpendicular | 0.1270 | 3.5856 | 0.0524 | 2.7730 | 0.2188 | 0.8412 | 0.7090 |
| random | 0.0605 | 4.3566 | -0.0029 | 2.8576 | 0.2188 | 0.8412 | 0.7090 |
| signflip | 0.0049 | 5.6551 | -0.0778 | 2.8826 | 0.2188 | 0.8412 | 0.7090 |
| strongest_nontarget_zero | 0.1465 | 3.6205 | 0.0788 | 2.8690 | 0.2188 | 0.8412 | 0.7090 |
| target_zero | 0.1270 | 3.5856 | 0.0524 | 2.7730 | 0.2188 | 0.8412 | 0.7090 |

The learned query clearly mattered globally: sign-flip and random-query interventions were destructive. However, selectively removing the target-key projection changed accuracy by only 0.0176, changed loss by -0.0463, and replicated in only 1/8 seeds. That does not establish target-direction-specific functional addressing.

### B — Exposure × reuse phase diagram

The full 6×7 grid used external-exposure counts 1–32 and value-free reuse counts 0–32, matched interference, then H/F scrub before slow-only recall. Representative cells are below; all 42 cells and heatmaps are in the dedicated report and processed artifacts.

| exposure_count | reuse_count | retention | accuracy | transfer_mass |
|---|---|---|---|---|
| 1 | 0 | 0.5875 | 0.3262 | 0.0529 |
| 1 | 8 | 0.9441 | 0.9824 | 0.1897 |
| 1 | 32 | 0.9489 | 0.9883 | 0.2030 |
| 8 | 0 | 0.8951 | 0.8965 | 0.3801 |
| 8 | 8 | 0.9816 | 1.0000 | 0.5034 |
| 8 | 32 | 0.9832 | 1.0000 | 0.5280 |
| 32 | 0 | 0.8973 | 0.9023 | 0.6447 |
| 32 | 8 | 0.9854 | 1.0000 | 0.7294 |
| 32 | 32 | 0.9864 | 1.0000 | 0.7551 |

The descriptive log-model reuse/exposure coefficient ratio was 1.1543, seed-bootstrap 95% CI [1.0713, 1.2701]. Both axes improved retention, with a negative interaction indicating saturation/substitution. This ratio is not a universal exchange rate.

### C — Selective persistence scaling

Sixteen future-used facts competed with 0, 32, 128, 512, or the preregistered feasible maximum of 2048 same-format distractors. Persistent state remained fixed. Each architecture independently selected its LR from the same development budget.

| model | distractor_count | accuracy | retention | interference | selective_persistence_efficiency | persistent_state_bytes |
|---|---|---|---|---|---|---|
| B1_gru | 0 | 0.0165 | 0.0000 | 1.0000 | 0.0000 | 2048.0000 |
| B1_gru | 32 | 0.0165 | 0.0000 | 1.0000 | 0.0000 | 2048.0000 |
| B1_gru | 128 | 0.0165 | 0.0000 | 1.0000 | 0.0000 | 2048.0000 |
| B1_gru | 512 | 0.0165 | 0.0000 | 1.0000 | 0.0000 | 2048.0000 |
| B1_gru | 2048 | 0.0165 | 0.0000 | 1.0000 | 0.0000 | 2048.0000 |
| B2_single_persistent | 0 | 0.9541 | 0.8874 | 0.1126 | 0.0001 | 10240.0000 |
| B2_single_persistent | 32 | 0.2180 | 0.3680 | 0.6320 | 0.0000 | 10240.0000 |
| B2_single_persistent | 128 | 0.0172 | 0.0580 | 0.9420 | 0.0000 | 10240.0000 |
| B2_single_persistent | 512 | 0.0193 | 0.0301 | 0.9699 | 0.0000 | 10240.0000 |
| B2_single_persistent | 2048 | 0.0145 | 0.0226 | 0.9774 | 0.0000 | 10240.0000 |
| B3_uniform | 0 | 0.7372 | 0.8857 | 0.1143 | 0.0001 | 10240.0000 |
| B3_uniform | 32 | 0.0396 | 0.7856 | 0.2144 | 0.0000 | 10240.0000 |
| B3_uniform | 128 | 0.0204 | 0.4676 | 0.5324 | 0.0000 | 10240.0000 |
| B3_uniform | 512 | 0.0220 | 0.2069 | 0.7931 | 0.0000 | 10240.0000 |
| B3_uniform | 2048 | 0.0171 | 0.0790 | 0.9210 | 0.0000 | 10240.0000 |
| B5_no_idle | 0 | 0.9641 | 0.7056 | 0.2944 | 0.0001 | 10240.0000 |
| B5_no_idle | 32 | 0.0968 | 0.5786 | 0.4214 | 0.0000 | 10240.0000 |
| B5_no_idle | 128 | 0.0337 | 0.4531 | 0.5469 | 0.0000 | 10240.0000 |
| B5_no_idle | 512 | 0.0260 | 0.3539 | 0.6461 | 0.0000 | 10240.0000 |
| B5_no_idle | 2048 | 0.0181 | 0.2265 | 0.7735 | 0.0000 | 10240.0000 |
| B6_full | 0 | 0.7927 | 0.7091 | 0.2909 | 0.0001 | 10240.0000 |
| B6_full | 32 | 0.0653 | 0.6250 | 0.3750 | 0.0000 | 10240.0000 |
| B6_full | 128 | 0.0382 | 0.5442 | 0.4558 | 0.0000 | 10240.0000 |
| B6_full | 512 | 0.0326 | 0.4920 | 0.5080 | 0.0000 | 10240.0000 |
| B6_full | 2048 | 0.0195 | 0.3647 | 0.6353 | 0.0000 | 10240.0000 |

ET-RCM had a better mean degradation slope than B2 by 0.0266 and a positive byte-normalized efficiency margin, but its 2048-distractor accuracy margin was only 0.0050, below the frozen 0.05 threshold. G15 therefore failed rather than being rescued by partial metrics.

### D — Autonomous multi-memory selection

After writes and an H scrub, TASK_CUE contained the operation only—no key IDs, values, or retrieval schedule—and all later events were NULL. The query trajectory visited task-relevant key directions and improved behavior for the first few ticks, although later ticks could degrade performance.

| operation | internal_tick | accuracy | projection_A | projection_B | projection_C | confidence |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.4688 | 0.9662 | 0.4141 | 0.1597 | 0.8535 |
| 0 | 1 | 0.7794 | 0.3597 | 0.9747 | -0.2196 | 0.8113 |
| 0 | 2 | 0.9129 | 0.9725 | 0.2776 | 0.2844 | 0.9265 |
| 0 | 4 | 0.9028 | 0.8379 | 0.2780 | 0.1699 | 0.9208 |
| 0 | 8 | 0.8737 | 0.7414 | 0.1562 | 0.1585 | 0.8866 |
| 0 | 16 | 0.7925 | 0.7497 | 0.1475 | 0.1154 | 0.8580 |
| 1 | 0 | 0.4495 | 0.9663 | 0.4489 | 0.1101 | 0.7118 |
| 1 | 1 | 0.7218 | 0.3199 | 0.9904 | -0.2430 | 0.8110 |
| 1 | 2 | 0.8287 | 0.9905 | 0.2488 | 0.1905 | 0.8633 |
| 1 | 4 | 0.8243 | 0.7963 | 0.2686 | 0.0255 | 0.8561 |
| 1 | 8 | 0.8141 | 0.7433 | 0.1888 | -0.0135 | 0.8231 |
| 1 | 16 | 0.7277 | 0.7216 | 0.1336 | 0.0216 | 0.8008 |
| 2 | 0 | 0.7278 | 0.9846 | 0.1515 | 0.2099 | 0.8640 |
| 2 | 1 | 0.8639 | 0.2797 | 0.9791 | -0.2592 | 0.9344 |
| 2 | 2 | 1.0000 | 0.2078 | -0.2492 | 0.9856 | 0.9982 |
| 2 | 4 | 0.9749 | 0.0221 | -0.2137 | 0.3040 | 0.9837 |
| 2 | 8 | 0.8166 | 0.2389 | -0.0824 | 0.2118 | 0.9333 |
| 2 | 16 | 0.7322 | 0.4212 | 0.0491 | 0.0627 | 0.9047 |

### E — Sequential computation under a hard per-tick bottleneck

The graph history was scrubbed from H; each tick emitted one query and received one addressed read. Lengths 6–8 were held out. The bypass audit passed, but OOD accuracy did not systematically improve with tick budget.

| internal_tick | accuracy | query_path_alignment | confidence | entropy |
|---|---|---|---|---|
| 0 | 0.5076 | 0.9616 | 0.6549 | 0.6079 |
| 1 | 0.5094 | 0.8815 | 0.5645 | 0.6736 |
| 2 | 0.5474 | 0.8634 | 0.5546 | 0.6799 |
| 4 | 0.5472 | 0.8091 | 0.5659 | 0.6703 |
| 8 | 0.5173 | 0.7572 | 0.5685 | 0.6672 |
| 16 | 0.5116 | 0.6997 | 0.5717 | 0.6625 |

### F — Endogenous-time timing intervention

The two schedules used identical event content, transition count, and total compute. Only whether K NULL transitions happened before or after resource-competing interference changed.

| condition | accuracy | retention | loss | confidence |
|---|---|---|---|---|
| after_interference | 0.4795 | 0.7162 | 2.1435 | 0.2586 |
| before_interference | 0.7070 | 0.8254 | 1.5908 | 0.3183 |

The matched-compute behavioral margin was 0.2275, replicated in 8/8 seeds. This supports pre-interference consolidation timing in this toy; it does not establish general autonomous reasoning.

### G — No self-evidence without an explicit cue

Knowable and unknowable examples used the same event kinds, shapes, query format, and output head. Only evidence availability differed.

| condition | internal_tick | accuracy | confidence | entropy | ece | brier |
|---|---|---|---|---|---|---|
| knowable | 0 | 0.9993 | 0.8794 | 0.3470 | 0.1199 | 0.0194 |
| knowable | 1 | 1.0000 | 0.9021 | 0.3028 | 0.0979 | 0.0129 |
| knowable | 2 | 1.0000 | 0.9065 | 0.2924 | 0.0935 | 0.0119 |
| knowable | 4 | 1.0000 | 0.9032 | 0.2971 | 0.0968 | 0.0130 |
| knowable | 8 | 1.0000 | 0.8867 | 0.3265 | 0.1133 | 0.0181 |
| knowable | 16 | 1.0000 | 0.8565 | 0.3758 | 0.1435 | 0.0289 |
| knowable | 32 | 0.9953 | 0.8209 | 0.4257 | 0.1744 | 0.0446 |
| knowable | 64 | 0.9838 | 0.7896 | 0.4641 | 0.1942 | 0.0611 |
| unknowable | 0 | 0.4997 | 0.6591 | 0.6154 | 0.1594 | 0.2867 |
| unknowable | 1 | 0.5002 | 0.6572 | 0.6170 | 0.1570 | 0.2856 |
| unknowable | 2 | 0.5000 | 0.6548 | 0.6185 | 0.1548 | 0.2846 |
| unknowable | 4 | 0.5027 | 0.6507 | 0.6213 | 0.1481 | 0.2829 |
| unknowable | 8 | 0.5021 | 0.6459 | 0.6251 | 0.1438 | 0.2810 |
| unknowable | 16 | 0.5016 | 0.6409 | 0.6293 | 0.1393 | 0.2790 |
| unknowable | 32 | 0.5002 | 0.6364 | 0.6330 | 0.1361 | 0.2774 |
| unknowable | 64 | 0.5005 | 0.6328 | 0.6359 | 0.1323 | 0.2761 |

Unknowable K=64 accuracy was 0.5005; confidence changed by -0.0263 and ECE by -0.0271. The safety/calibration part passed. The compound G17 nevertheless failed because the already-near-ceiling knowable stratum improved by only 0.0007, below the frozen usefulness threshold.

### H — Stored versus used

Interventions were evaluated after 2048 distractors. Pre-lesion storage and post-lesion behavior are named separately.

| condition | lesion_component | accuracy | loss | pre_lesion_slow_retention | post_lesion_slow_retention | pre_lesion_accuracy | post_lesion_accuracy |
|---|---|---|---|---|---|---|---|
| lesion_both | both | 0.0146 | 4.2280 | 0.7141 | 0.0000 | 0.0449 | 0.0146 |
| lesion_fast | fast | 0.2539 | 3.6282 | 0.7141 | 0.7141 | 0.0449 | 0.2539 |
| lesion_slow | slow | 0.0107 | 4.6175 | 0.7141 | 0.0000 | 0.0449 | 0.0107 |
| query_original | none | 0.0449 | 4.0330 | 0.7141 | 0.7141 | 0.0449 | 0.0449 |
| query_target_zero | none | 0.0342 | 4.1473 | 0.7141 | 0.7141 | 0.0449 | 0.0342 |

Slow-memory lesion sharply reduced both slow retention and accuracy, so M was behaviorally used. F-only lesion improved accuracy in this long-interference setting, indicating fast-state interference. Target-query projection removal remained weak, consistent with G14 failure.

### I — Long continuous stream

`NOT_RUN_BY_PROTOCOL`. The frozen authorization condition required the Memory-LM path gates, which were not satisfied. No 1e3/1e4/1e5 result is implied.

## Direct answers to the 17 required questions

1. **Was G7 mainly a sign artifact?** No evidence supports that reinterpretation. G14=FAIL; target-direction removal was weak, and historical G7 remains FAIL.
2. **Does target-component removal hurt?** Mean accuracy drop=0.0176, with 1/8 registered seed replications.
3. **Is q≈-k functional gauge addressing?** Not established. Original accuracy=0.1445 and sign-flip accuracy=0.0049, so query sign matters globally, but target projection was not specifically necessary. The fresh-seed mean signed cosine was positive, not stable evidence for q≈-k.
4. **Reuse relative to exposure?** The descriptive log-model reuse/exposure coefficient ratio is 1.1543 (seed-bootstrap CI is in the phase report); it is toy-distribution-specific.
5. **Retention surface shape?** The full 6×7 grid, interaction/log fits and heatmaps are preserved; it is not assumed linear.
6. **Scaling versus permanent storage?** G15=FAIL at the preregistered 2048-event endpoint and over the degradation curve.
7. **Fair after bytes/tuning?** B2 matches F+M matrix floats; all architectures received identical search budgets. Exact bytes/parameters/compute are per row.
8. **Autonomous multiple retrieval?** K8-K0 task accuracy changed by 0.2871; trajectories contain A/B/C projections with no post-goal key cues.
9. **Sequential learned ticks?** G16=FAIL under the passed=True bottleneck audit.
10. **Pre-event benefit?** ET_STATUS_SUPPORTED; behavioral margin=0.2275. State distance alone is not counted.
11. **Calibration without unknowable cue?** The unknowable stratum remained near chance and became no more confident, with same-format audit=True. The compound G17 still failed because its knowable-benefit clause was not met.
12. **Stored versus used?** Query-removal and F/M lesion behavior is reported separately from pre/post stored retention in `MEMORY_STORAGE_VS_USE_STAGE1_2.md`.
13. **Stable through 1e5 events?** Not established: Experiment I was NOT_RUN_BY_PROTOCOL because its authorization rule failed.
14. **Necessary parts?** Only intervention-specific losses support necessity; strong storage without behavioral use is not counted. See G14, G15 and the lesion report.
15. **Keep endogenous internal time?** Keep it only as a tested consolidation-scheduling option: timing mattered under interference, but G16 failed, so continuous cognition is not justified as a core claim.
16. **Stage-2 Memory LM authorized?** **False**.
17. **Stage-2 Continuous Cognition LM authorized?** **False**.

## Scientific boundary

These are synthetic bounded-state experiments. Functional projection is called functional only when an intervention changes behavior. Autonomous memory selection is not autonomous thought; long operation is not infinite context; no result establishes human-like memory, consciousness, or causal state in the unrestricted sense.
