# ET-RCM Stage 1.6 Final Report

> **Is ET-RCM failing to use persistent memory because its current integration architecture is incapable of doing so, or because the training process never forces the recurrent core to learn memory-dependent computation?**

> **ET-RCM 当前无法有效利用持久记忆，究竟是因为现有 memory-to-H integration 架构本身做不到，还是因为训练过程从未真正迫使 recurrent core 学会依赖 memory 进行计算？**

Formal run `stage1_6-formal-v1`: 48 training cells, 8 fresh training seeds, 552,960 intervention rows. Parent Stage 1.5 reports and frozen results were not modified. No integration architecture or memory-law change.

## Frozen gates

| gate   | outcome   |
|:-------|:----------|
| G34    | FAIL      |
| G35    | PASS      |
| G36    | FAIL      |
| G37    | FAIL      |

## Paired causal effects

| effect                                          |        mean | ci95                                             |   invalid_seeds |   positive_seeds |   threshold |
|:------------------------------------------------|------------:|:-------------------------------------------------|----------------:|-----------------:|------------:|
| D_R_3000_minus_160                              | 0.614565    | [4.522779490798712e-05, 1.7387914092485175]      |               0 |                2 |       0.01  |
| D_M_3000_minus_160                              | 0.280345    | [-0.00011986367753706873, 0.6815422703842927]    |               0 |                2 |       0.01  |
| oracle_vs_zero                                  | 2.81491     | [2.0779377373114998, 3.5661207667105086]         |               0 |                8 |       0.05  |
| oracle_vs_random                                | 3.63797     | [2.8793404879515188, 4.468620885442243]          |               0 |                8 |       0.02  |
| oracle_vs_shuffled                              | 4.98985     | [4.128236408240672, 5.805042752040317]           |               0 |                8 |       0.02  |
| no_memory_vs_full                               | 0.436838    | [-4.374398558866232e-05, 1.0397187691125964]     |               0 |                2 |       0.05  |
| M_lesion_vs_full                                | 0.280365    | [4.970747977495194e-06, 0.6814563841453349]      |               0 |                2 |       0.02  |
| curriculum_fraction                             | 1.03255     | [1.0001770284552625, 1.0971926644838594]         |               5 |                3 |       0.5   |
| curriculum_learned_vs_zero                      | 1.63107     | [0.4596438921686929, 2.8540544090032633]         |               0 |                6 |       0.025 |
| curriculum_vs_oracle_trained_fraction_secondary | 0.503372    | [0.12672949486452134, 0.910045675874835]         |               0 |                3 |       0.5   |
| curriculum_F_lesion_secondary                   | 4.26258     | [1.7101605419311596, 6.803371011067173]          |               0 |                6 |       0.02  |
| curriculum_M_lesion_secondary                   | 0.000170435 | [2.5843104012324195e-07, 0.00039943971581664047] |               0 |                0 |       0.02  |

Rows labeled `secondary` were examined after formal gate outcomes and cannot change them. The curriculum F/M-lesion contrast is descriptive about which component carried its learned-read benefit.

## Final condition outcomes

| training_arm   | condition   |      CE |   accuracy |
|:---------------|:------------|--------:|-----------:|
| curriculum     | F_lesion    | 4.9454  |    0.20264 |
| curriculum     | M_lesion    | 0.68299 |    0.72217 |
| curriculum     | learned     | 0.68282 |    0.72217 |
| curriculum     | oracle      | 1.54642 |    0.60596 |
| curriculum     | random      | 2.59771 |    0.4541  |
| curriculum     | shuffled    | 3.00732 |    0.44434 |
| curriculum     | zero        | 2.31389 |    0.45654 |
| gru            | F_lesion    | 2.08482 |    0.11914 |
| gru            | M_lesion    | 2.08482 |    0.11914 |
| gru            | learned     | 2.08482 |    0.11914 |
| learned        | F_lesion    | 2.20081 |    0.20947 |
| learned        | M_lesion    | 1.92726 |    0.22705 |
| learned        | learned     | 1.64689 |    0.28125 |
| learned        | oracle      | 2.64512 |    0.16162 |
| learned        | random      | 2.32816 |    0.19287 |
| learned        | shuffled    | 2.46508 |    0.1709  |
| learned        | zero        | 2.26136 |    0.20312 |
| no_memory      | F_lesion    | 2.08373 |    0.11475 |
| no_memory      | M_lesion    | 2.08373 |    0.11475 |
| no_memory      | learned     | 2.08373 |    0.11475 |
| oracle         | F_lesion    | 5.34118 |    0.4248  |
| oracle         | M_lesion    | 5.21544 |    0.49805 |
| oracle         | learned     | 5.38943 |    0.50049 |
| oracle         | oracle      | 0.13756 |    0.95605 |
| oracle         | random      | 3.77552 |    0.41064 |
| oracle         | shuffled    | 5.12741 |    0.36963 |
| oracle         | zero        | 2.95246 |    0.40039 |
| single_memory  | F_lesion    | 2.08369 |    0.11475 |
| single_memory  | M_lesion    | 2.08372 |    0.11475 |
| single_memory  | learned     | 2.08369 |    0.11475 |

## Model and state sizes

| training_arm   |   parameter_count |   state_bytes |
|:---------------|------------------:|--------------:|
| curriculum     |            151877 |          2560 |
| gru            |            151877 |          2560 |
| learned        |            151877 |          2560 |
| no_memory      |            151877 |          2560 |
| oracle         |            151877 |          2560 |
| single_memory  |            151877 |          2560 |

Parameter counts include instantiated but potentially inactive baseline modules; state bytes count H/F/M float32 slots per episode.

## Experimental details

**Task.** B∈0–7, A∈8–15 and C∈16–23 are sampled independently per episode. Four genuine external B→A exposures update F by the unchanged delta rule. Only H is reset to initial H; F/M are preserved exactly. Eight identical non-writing context distractors follow, then a bridge exposes B and C. The unseen future class is Y=(A−8+C−16) mod 8. Thus current H alone cannot identify Y; the historical A is necessary, while A without later C is not the answer. Model output head receives only H. Counterfactual leakage and no-memory identifiability are tested.

**Read interventions.** At the bridge, the oracle supplies the raw (F+M) read at the observed historical B, saved immediately after exposure 4. It has no episode-specific C or Y. Oracle training uses the existing slow-read normalization/arbitration and clamps fast read to zero. Learned training uses the existing fast/slow route. Zero clamps both read channels; random replaces the slow input with an independent equal-norm vector; shuffled uses a derangement across episodes. These controls use the same interface. F/M lesions zero exactly one component immediately after H scrub, before distractor reads can carry it into H. An induced downstream state change is permitted; external event/write law is unchanged.

**Optimization and pairing.** Development seeds 8601–8602 each tested LR .001/.0003 across all six arms for 160 AdamW steps and selected common LR .001 by equal-arm/seed held-out CE (2.10779 versus 2.12833). Formal seeds 8701–8708 are independent and disjoint. Each of six arms sees the same 3000 world draws per seed, batch 32 (96,000 train episodes per arm/seed), gradient clip 1.0, weight decay .0001, FP32. B5 arms share exact initialization per seed. Curriculum p_oracle=1/.75/.5/.25/0 over five 600-step blocks; the realized per-step draws are stored. Checkpoints 0/50/100/160/300/500/1000/2000/3000 use the same 256 held-out episodes/seed across arms, conditions and checkpoints. These are repeated paired evaluations, not independent new episodes.

**Statistics and diagnostics.** Episode CE is averaged within seed; seeds receive equal weight. Gate margins require at least 6/8 seed-level replications at preregistered thresholds; 95% intervals bootstrap independent seeds 2000 times. Finite interventions decide use; gradient/JVP, gate saturation, read norm, train loss and raw state norms are only explanatory diagnostics. A bridge H-step delta in raw records includes event input; isolated raw/gated candidate norms are in the JVP diagnostic file. The pre-formal lesion-timing correction and post-first-seed G37 interpretation caveat are documented in separate amendment notes; no formal threshold changed. Checkpoint tensors, train logs, raw Parquet, source/config/report hashes and integrity manifest live under `results/stage1_6`.

## Answers to the 18 registered questions

1. No evidence that simply extending the original objective solves read use: on its 128-distractor long-gap slice, held-out CE fell by 0.724 from 160 to 3000 steps, but oracle-vs-no-read benefit changed by -0.00041 and cleared .01 in only 1/8 seeds at 3000. This is exploratory and narrower than the full Stage 1.5 grid.

2. Not reliably in ordinary learned-read training: G34=FAIL; D_R and D_M gains from 160 to 3000 averaged 0.615/0.280 CE but crossed .01 in only 2/8 and 2/8 seeds, respectively.

3. Yes, for supplied past-only read on this toy: G35=PASS; oracle-trained oracle CE=0.138 versus zero CE=2.952, with all three registered contrasts clearing margin in 8/8 seeds.

4. Yes under registered margins: oracle-vs-zero/random/shuffled CE advantages were 2.815/3.638/4.990, each 8/8 seeds.

5. The existing integration operator demonstrably has capacity to use a correct historical read on this task; no absolute architecture ceiling is established. This does not imply learned routing or persistent M use.

6. Not reproducibly with the ordinary full model: G36=FAIL; full-vs-no-memory and M-lesion effects cleared thresholds in only 2/8 and 2/8 seeds, respectively. The curriculum did induce F-sensitive computation, not replicated M necessity.

7. The no-memory baseline remained near chance (CE=2.084, 8-class chance ≈2.079) after H scrub; ordinary full CE=1.647, but the paired advantage crossed .05 in only 2/8 seeds.

8. Not robustly: ordinary M lesion crossed .02 CE harm in 2/8 seeds; curriculum M lesion did so in 0/8 versus F lesion in 6/8 (the latter two are post-formal descriptive checks).

9. Replicated oracle benefit first appeared at step 2000; ordinary learned benefit never reached 6/8 at any checkpoint, while curriculum learned benefit did so at 3000.

10. No within-curriculum oracle-first sequence was established: its oracle benefit never crossed .01 in 6/8 seeds, whereas learned benefit did at step 3000. Cross-arm oracle training succeeded earlier, but that is not a within-model learning order.

11. No complete gradient starvation: at step 3000 the learned/oracle memory-branch gradient norms averaged 0.0101/0.0435; finite interventions, not gradient magnitude, establish use. Relative optimization weakness remains possible.

12. No universal gate collapse: step-3000 slow-read gate means were 0.409/0.750 for learned/oracle arms; recurrent gate saturation fractions were 0.163/0.429. Some saturation warrants study but is not a causal verdict.

13. H scrub removes direct historical H information on the new task, and no-memory chance behavior confirms that shortcut is blocked there. The original Stage 1.5 world's shortcut was not itself isolated, but its CE improvement without read benefit is consistent with objective bypass.

14. G37=FAIL; ratio requires a positive oracle-benefit denominator. Separate auxiliary F trigger=True, with 2/8 learned-arm seeds clearing D_R≥.01. Triggered F subsequently cleared learned D_R≥.025 in 6/8 seeds, without revising G37.

15. Oracle integration passed in the present gated-residual operator, but ordinary learned-read training and persistent M necessity did not replicate. The preregistered B3 curriculum had learned-read benefit in 6/8 seeds; a post-formal component dissection found F-lesion benefit in 6/8 but M-lesion benefit in 0/8. This suggests curriculum-assisted fast-memory use, not proven slow persistent-memory integration. Frozen G36=FAIL and G37=FAIL are unchanged.

16. No integration-operator redesign is currently justified by G35: the existing gated residual can use supplied read. A future Stage 1.7 should prioritize retrieval training and slow-state necessity; separate long-NULL instability remains unresolved.

17. Retain the gated-residual operator as the next toy-test baseline because G35 passed, but do not treat current F/M persistence or ordinary learned retrieval as validated; G34/G36/G37 failed.

18. No. Stage 1.5's G27/G28/G31 failures and the narrow toy scope preclude sequence/LM prototype authorization.

## Conditional auxiliary Experiment F

TRIGGERED_AND_COMPLETED. The equal-oracle-budget three-phase curriculum is a separately labeled post-trigger experiment, not a fifth gate. Learned-read benefit ≥.025 replicated in 6/8 seeds; M/F-lesion benefit ≥.02 replicated in 2/8 and 4/8. See `reports/AUXILIARY_MEMORY_USE_CURRICULUM_STAGE1_6.md`; G34–G37 are unchanged.

## Original-objective exploratory scaling

See `reports/TRAINING_LENGTH_SCALING_ORIGINAL_OBJECTIVE_STAGE1_6.md`. No evidence that simply extending the original objective solves read use: on its 128-distractor long-gap slice, held-out CE fell by 0.724 from 160 to 3000 steps, but oracle-vs-no-read benefit changed by -0.00041 and cleared .01 in only 1/8 seeds at 3000. This is exploratory and narrower than the full Stage 1.5 grid.

## Interpretation and limits

Oracle integration passed in the present gated-residual operator, but ordinary learned-read training and persistent M necessity did not replicate. The preregistered B3 curriculum had learned-read benefit in 6/8 seeds; a post-formal component dissection found F-lesion benefit in 6/8 but M-lesion benefit in 0/8. This suggests curriculum-assisted fast-memory use, not proven slow persistent-memory integration. Frozen G36=FAIL and G37=FAIL are unchanged. The historical oracle is a saved pre-distractor F+M read, so it is an upper-bound delivery intervention, not proof that current slow M retrieves the same content. The H-scrub compositional task is intentionally simpler and more memory-forcing than Stage 1.5's four-family distribution; success here does not retroactively erase its negative results. M/F lesions are one-time at the scrub boundary; the remaining component may later reconsolidate, so their effects are conservative about sustained component necessity. The gates are toy-specific and do not imply human-like memory, consciousness, unlimited information capacity or language-model readiness. Missing/negative outcomes are retained.
