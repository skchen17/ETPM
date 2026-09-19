# Causal Usage Versus Retention — Stage 1.4

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**


After 128 interference writes, F is excluded and each fact's slow-memory
read is compared by cosine to its originally written value vector. Mean M-only
retention=0.2991. Fourfold within-seed held-out rank
regression compares exposure+read against exposure+read+CU; seeds, not items,
are replication units. Spearman coefficients are descriptive.

| Seed | Exposure–retention ρ | Read–retention ρ | CU–retention ρ | Incremental R² |
|---:|---:|---:|---:|---:|
| 7501 | 0.2066 | 0.2826 | -0.3273 | 0.1381 |
| 7502 | 0.0778 | 0.3281 | 0.0557 | -0.0650 |
| 7503 | 0.3148 | 0.1979 | -0.1441 | -0.1201 |
| 7504 | 0.2838 | 0.1921 | 0.1129 | -0.1037 |
| 7505 | -0.0117 | 0.0630 | -0.0795 | -0.1151 |
| 7506 | 0.4028 | -0.1250 | 0.0733 | -0.1705 |
| 7507 | 0.0518 | -0.0517 | -0.0128 | -0.0366 |
| 7508 | 0.3168 | 0.2287 | 0.3567 | 0.0336 |

G26 **FAIL**: mean incremental held-out R²
-0.0549 (95% CI
`['-0.1155', '0.0138']`), seeds at registered +0.02 floor
2/8. Correlation cannot substitute for the
required consolidation intervention.

Exploratory access-frequency sensitivity (not a G26 redefinition): replay
uses a hard query-key cosine ≥0.5 count and a soft cosine sum. It compares
CU incremental R² after exposure, read magnitude and hard count controls.

| Seed | Hard-count–retention ρ | Soft-count–retention ρ | CU incremental R² after count |
|---:|---:|---:|---:|
| 7501 | -0.0319 | 0.1712 | 0.1178 |
| 7502 | 0.0913 | 0.3446 | -0.0756 |
| 7503 | -0.1520 | -0.1004 | -0.0776 |
| 7504 | 0.0921 | 0.0590 | -0.0828 |
| 7505 | -0.0670 | -0.1081 | -0.2405 |
| 7506 | 0.1218 | -0.1514 | -0.0754 |
| 7507 | 0.1581 | 0.0301 | -0.0927 |
| 7508 | 0.3445 | 0.2419 | -0.0245 |

Mean count-adjusted incremental R²=-0.0689.
This secondary threshold was not tuned or used to authorize G.
