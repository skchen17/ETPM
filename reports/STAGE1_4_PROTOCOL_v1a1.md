# ET-RCM Stage 1.4 corrected protocol v1a1

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**

This is an append-only correction to `STAGE1_4_PROTOCOL.md`, **not** a
post-hoc change to G23–G26. Amendment A4 explains why the first attempted
training/evaluation run is scientifically invalid and retained separately.

The corrected latent-transition event contains only the noisy observation in
both visible key/value fields. The true latent state exists solely in generator
audit metadata, never in `ContinuousEvent` or `model.step`. The long-gap
relation generator enforces distinct B and C symbols so the later bridge does
not accidentally overwrite its own B→A association. All other world families,
model equations, future horizons/weights, evaluation interventions and gate
thresholds remain as in the original frozen protocol and analysis plan.

Corrected config: `configs/stage1_4_v1a1.yaml`. Development run:
`stage1_4-development-v1a1` with new seeds 6501/6502. Formal run:
`stage1_4-formal-v1a1` with new seeds 7501–7508. There are 8 trainable
architectures, the same two candidate learning rates and per-architecture
development budget, 100 development and 160 formal steps, batch 32.
Development causal selection uses a distinct v1a2 run. All new source/config,
selection and prior-artifact hashes are frozen before corrected formal
training. Experiment G remains conditional on **corrected** development CU
incremental held-out prediction, not on any invalid old run or formal result.

All invalid first-run shards remain machine-readable and explicitly excluded
from gate adjudication. A corrected run can fail every gate; a changed world
does not grant permission to adjust success criteria or start an LM.
