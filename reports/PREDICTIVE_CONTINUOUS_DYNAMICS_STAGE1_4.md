# Predictive Continuous Dynamics — Stage 1.4

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**


Eight formal B5 models forecast horizons 1/2/4/8 after the same external
history and K=0/1/2/4/8/16 NULL ticks. Each prefix/target is paired across
K; the only changing input is internal time. Frozen-H and fixed random
recurrent NULL controls use the same pre-event transition count. The matched
compute timing arm places K ticks either before or after the next external
event and forecasts the subsequent event; post-event ticks incur K ticks of
future latency and cannot count as pre-event forecasting.

| K | Mean h=1 CE |
|---:|---:|
| 0 | 0.9571 |
| 1 | 0.9543 |
| 2 | 0.9529 |
| 4 | 0.9530 |
| 8 | 0.9588 |
| 16 | 0.9781 |

Full family/horizon curve (mean CE over eight trained seeds):

| World family | Horizon | K0 | K1 | K2 | K4 | K8 | K16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| latent_transition | 1 | 2.2432 | 2.2374 | 2.2340 | 2.2309 | 2.2310 | 2.2385 |
| latent_transition | 2 | 2.2832 | 2.2809 | 2.2796 | 2.2783 | 2.2780 | 2.2792 |
| latent_transition | 4 | 2.3227 | 2.3197 | 2.3179 | 2.3160 | 2.3154 | 2.3168 |
| latent_transition | 8 | 2.4132 | 2.4127 | 2.4127 | 2.4132 | 2.4149 | 2.4173 |
| long_gap_relation | 1 | 0.5018 | 0.5034 | 0.5061 | 0.5136 | 0.5327 | 0.5747 |
| latent_regime | 1 | 0.9653 | 0.9589 | 0.9547 | 0.9506 | 0.9513 | 0.9670 |
| latent_regime | 2 | 1.0850 | 1.0794 | 1.0765 | 1.0752 | 1.0814 | 1.1036 |
| latent_regime | 4 | 1.2360 | 1.2324 | 1.2315 | 1.2337 | 1.2448 | 1.2723 |
| latent_regime | 8 | 1.4371 | 1.4327 | 1.4307 | 1.4306 | 1.4374 | 1.4579 |
| distractor_heavy | 1 | 0.1183 | 0.1173 | 0.1169 | 0.1170 | 0.1201 | 0.1321 |

| Seed | L0−L4 | Frozen−learned at K4 | Random−learned at K4 |
|---:|---:|---:|---:|
| 7501 | 0.0126 | 0.0126 | 0.0324 |
| 7502 | 0.0089 | 0.0089 | 0.0385 |
| 7503 | -0.0012 | -0.0012 | 0.0360 |
| 7504 | -0.0111 | -0.0111 | 0.0147 |
| 7505 | 0.0013 | 0.0013 | 0.0330 |
| 7506 | 0.0086 | 0.0086 | 0.0391 |
| 7507 | -0.0034 | -0.0034 | 0.0352 |
| 7508 | -0.0014 | -0.0014 | 0.0088 |

G23 **FAIL**: mean L0−L4=0.0018,
seed-bootstrap 95% CI `['-0.0030', '0.0069']`,
positive seeds=4/8, frozen margin=0.0018,
random margin=0.0297. The registered criterion is not
changed for non-monotone curves. Matched timing mean losses:
`{"idle_pre_event_then_event": 1.6065779526058275, "post_event_ticks": 1.6044805343941941}`.
The full equal-compute timing table is:

| K | Pre-event K, CE | Post-event K, CE | Post−pre CE |
|---:|---:|---:|---:|
| 0 | 1.6115 | 1.6115 | 0.0000 |
| 1 | 1.6071 | 1.6064 | -0.0007 |
| 2 | 1.6044 | 1.6031 | -0.0013 |
| 4 | 1.6018 | 1.5996 | -0.0022 |
| 8 | 1.6023 | 1.5990 | -0.0033 |
| 16 | 1.6123 | 1.6072 | -0.0051 |

Post-event compute requires K future-latency ticks; a lower post-event CE
does not prove pre-event autonomous predictive benefit.
These are toy-world predictive losses, not evidence of human-like thought.
