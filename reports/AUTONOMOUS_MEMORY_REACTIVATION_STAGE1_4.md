# Autonomous Memory Reactivation — Stage 1.4

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**


Each long-gap episode has early A→B evidence (1/2/4 true exposures), 128/512/
2048 unrelated writes, a genuine C→B bridge event, and a future A-dependent
token. The model sees the bridge, then four NULL ticks, but **never** receives
a target-key query, target-key auxiliary loss, future-use flag, or future token
as input. Same-checkpoint B5 M/F lesions, random q_M and shuffled M are made
before the bridge. Other baselines were trained independently with equal
development and formal budgets. CE below averages the 512/2048 gaps.

| B5 condition | Future-event CE |
|---|---:|
| full | 1.7987 |
| M_lesion | 1.7987 |
| F_lesion | 1.7988 |
| random_q_M | 1.7988 |
| shuffled_M | 1.7990 |

| Trained architecture | Full-condition CE |
|---|---:|
| B0_no_memory | 1.7519 |
| B1_gru | 2.7195 |
| B2_single_memory | 1.7702 |
| B3_joint | 1.7773 |
| B4_shared | 1.7961 |
| B5_separate | 1.7987 |
| B6_gamma_zero | 1.5576 |
| B7_random_query | 1.4551 |

All registered B5 gap/condition means:

| Distractors | B5 condition | Future-event CE |
|---:|---|---:|
| 128 | full | 0.9492 |
| 128 | M_lesion | 0.9460 |
| 128 | F_lesion | 0.9480 |
| 128 | random_q_M | 0.9459 |
| 128 | shuffled_M | 0.9497 |
| 512 | full | 1.5407 |
| 512 | M_lesion | 1.5407 |
| 512 | F_lesion | 1.5409 |
| 512 | random_q_M | 1.5410 |
| 512 | shuffled_M | 1.5414 |
| 2048 | full | 2.0567 |
| 2048 | M_lesion | 2.0567 |
| 2048 | F_lesion | 2.0568 |
| 2048 | random_q_M | 2.0566 |
| 2048 | shuffled_M | 2.0566 |

Descriptive B5 read/state diagnostics (not substitutes for lesion effects):

| Distractors | q_M coordinate variance | Mean g_M | Raw M-read norm | Effective M contribution norm | M-lesion future-H distance |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.059381 | 0.5487 | 0.0995 | 2.2480 | 2.2970 |
| 512 | 0.058378 | 0.4999 | 0.1164 | 2.0507 | 1.9924 |
| 2048 | 0.059035 | 0.5252 | 0.0714 | 2.1417 | 2.1376 |

G24 **FAIL**. Registered control-minus-full
margins: `{"M_lesion": -2.3996341042220592e-05, "no_persistent": -0.04685665329452604, "random_q_M": 4.320358857512474e-05}`. Seed-bootstrap
95% CIs: `{"M_lesion": [-0.0003163101093377918, 0.00034336221870034933], "no_persistent": [-0.6427297350019217, 0.5378010083222762], "random_q_M": [-0.0001529525383375585, 0.000250955554656684]}`. Nonzero q_M,
gate weight or read norm is descriptive access, not causal proof. Only paired
lesion and prediction effects count. The M lesion is applied just before the
bridge with H held identical; it tests *reactivation at that point*, not every
possible earlier M→H influence. Learned keys are not orthogonal, and 2048
distractors are an OOD gap relative to the 16-step training worlds.
