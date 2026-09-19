# Self-Output Pathological Positive Control — Stage 1.4

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**


The normal validated SELF_OUTPUT transition has zero external writes across
all formal records (count=0). In a separate
explicitly unsafe B_bad arm, each repeated self-output deliberately executes
the external delta write. From tick 1 to tick 8, its memory readout strength
changes by 0.3701; a **fixed
diagnostic proxy**, sigmoid(4×readout−2), changes by
0.2778. Positive-control
sensitivity is **VALID**.
The proxy is not a trained expression policy, confidence estimate, or claim
that an agent would literally speak more often. If INVALID, the normal arm's
zero writes cannot by itself validate a full amplification audit.
