## 实验方法细节

### 状态转移与指标

Associative-memory toys 只允许 external event 执行 delta write：
`F <- F + eta[v-(F+M)k]k^T`。Internal use 执行
`Delta=gamma(Fq)q^T, F<-F-Delta, M<-M+Delta`，在 decay 前检查 `F+M`
守恒，再应用 fast/slow decay。NULL_EVENT 不得进入 external write。长期
retention 在清除 F 后定义为 `<Mq,v>/<v,v>`；同时保存 cosine、interference、
state norms、transfer、query、dynamics、latency 和 compute metrics。

默认 single-head 配置为 `hidden=64`、`key=value=32`、`gamma=.1`、
`rho_fast=.95`、`rho_slow=.9995`、`eta=.5`。正式运行使用 seeds 42–49。

### Toy protocols

1. **Repeated exposure:** 同一 fact 真实出现 1/2/4/8 次，随后 32 个随机
   interference events；最终清 F 测 M。
2. **Repeated use:** 8 个正交-key facts 各真实出现一次，目标只被内部使用
   0/1/2/4/8 次，再运行 32-step interference；比较 B2–B6。
3. **Same input/future utility:** 相同 fact、不同 task context；单层 sigmoid
   gate 通过 downstream-use loss 训练 300 Adam steps，比较 useful/unused
   slow retention。它是机制演示，不是一般 importance-learning 结论。
4. **Idle reasoning:** 每 seed 128 个平衡 reachability episodes；路径长度 1–8，
   使用 `h<-clamp(h+hA,0,1)` 运行 0/1/2/4/8/16 ticks。
5. **Matched compute:** 将完全相同的 K 次 graph transition 放在 idle period
   或 query time，比较 accuracy、最终 state、latency 和 compute budget。
6. **Idle consolidation:** fact 只出现一次，运行 0/2/4/8/16 个无输入
   consolidation ticks；逐 tick 检查守恒，最终清 F。
7. **Unknowable bit:** 每 seed 4096 个未提供证据的 Bernoulli(.5) targets；
   0–32 ticks 后检查 accuracy/confidence/calibration。
8. **Revision:** old exposures 1/4/8，随后 8 次旧版本巩固；new exposures
   1/2/4、new reuses 0/2/8；清 F 后从 M 计算 `softmax(5·read)` 的 P(new)。
9. **Temporary/persistent:** 输入形式相同的两个 facts 分别使用 1 次和跨 episode
   使用 8 次，无 temporary/permanent 标签，最终比较 M retention。

### Baselines 与判定规则

B0/B1 已实现并做 API/shape smoke test；未训练网络不强行赋予 associative
retention 分数。Toy 2 正式比较 B2 single persistent、B3 uniform、B4
non-conserving、B5 no-idle 和 B6 full。B3 使用 `gamma/key_dim` 的各向同性
每方向预算；B2 使用强基线 `rho_slow`。

G2/G3 要求 group mean 随 sweep 非下降且末端严格提高；G4 要求 accuracy
至少提高 .1；G5 要求 unknowable accuracy 偏离 .5 小于 .05 且 confidence
range 小于 .01；G6 要求某 revision cell 的 P(new) 至少 .70。N1/N2 的末端
equivalence 容差为 .05；N3 要求 accuracy 相同且 state difference <1e-12；
N4 是 confidence 增长 >.02；N5 是没有 revision cell 达到 .70。Gate 是必要
条件，重大 negative criterion 仍可否决 Stage 2。

