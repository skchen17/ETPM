# ET-RCM Stage-1 toy validation report

> **Can a model with its own internal time transform transient experience into persistent computational state through repeated internal use, while keeping self-repetition from becoming new evidence?**
>
> **一个具有自身内部时间的模型，能否让短暂经历因为后续内部计算中的反复使用而自然转化为持久计算状态，同时避免把自己的重复思考误当成新的证据？**

## Execution scope and verdict

- Run: `stage1-20260917`; raw records: `results/raw/stage1-20260917/toy_records.parquet`.
- Records: 2104 total, 720 final episode summaries, 8 seeds.
- Stage-2 authorization: **FALSE**.
- This is a small synthetic validation. It is not evidence of human-like memory,
  consciousness, infinite information capacity, or an intervention-validated
  causal memory in a language model.

## 实验环境与可复现性

| 项目 | 当前正式运行值 |
|---|---|
| Host | `222.20.126.223` |
| Python | `3.13.12` |
| PyTorch | `2.6.0+cu124` |
| pandas / pyarrow | `3.0.3` / `25.0.1` |
| CUDA | 可用；本轮 toy tensor 明确创建在 CPU，主要使用 `float64` |
| 初始源码 revision | `b0d64f3188121289867edef88bc7608b6f26f4f2` |
| Config SHA-256 | `45db1ea51ce43b48803b7a463259c048bf6795bad10beba895e8e138d9de504b` |
| Seeds | `42..49`，共 8 个 |
| Raw rows / final rows | `2104 / 720` |

默认配置为：`hidden_dim=64`、`latent_slots=1`、`key_dim=value_dim=32`、
`heads=1`、`gamma=0.1`、`rho_fast=0.95`、`rho_slow=0.9995`、
`eta_external=0.5`、`max_idle_ticks=16`。Toy 3 的访问策略训练 300 步，
Adam learning rate 为 `0.08`；Toy 1/2 的 interference stream 长度为 32。

正式运行命令：

```bash
cd /data/CSK/ETPM/et-rcm
.venv/bin/python -m pytest
.venv/bin/python experiments/run_toy.py \
  --toy all --config configs/toy_default.yaml --run-id stage1-20260917
.venv/bin/python experiments/analyze_toys.py --tests-passed
```

原始记录和汇总文件均有 SHA-256 校验，见
`results/processed/stage1-20260917/SHA256SUMS`。参考项目
`/data/CSK/J-space-project` 只进行了读取，没有复制结果或建立 symlink。

## 统一状态转移协议

所有 associative-memory toy 使用相同的证据/使用分离：

1. 只有真实 external fact 才执行
   `F <- F + eta [v-(F+M)k]k^T`；`k` 单位归一化，`M` 不直接写入。
2. internal use 使用单位 query 执行
   `Delta=gamma(Fq)q^T, F<-F-Delta, M<-M+Delta`。
3. conservation 在 decay 前测量；随后才执行
   `F<-rho_fast F, M<-rho_slow M`。
4. `NULL_EVENT` 不进入 external write 路径。内部 tick 只推进 latent state、
   consolidation 和 decay。
5. slow-retention 实验在最终 query 前将 `F` 清零，只用 `M@q`，避免把尚未
   巩固的 fast trace 算作长期记忆。

主要 retention 定义为目标方向投影：

`Retention = <M q, v> / <v, v>`。

同时记录 cosine、L2 interference、`||H||/||F||/||M||/||F+M||`、每 tick
`||Delta||`、transfer fraction、query、reuse count、累计访问量、状态变化、
readout drift、latency 和 compute budget。完整列保存在 raw Parquet；无意义的
字段保留为空值，而不是伪造为零。

## 实验规模清单

| Toy | 总记录 | final 记录 | 主要 sweep |
|---|---:|---:|---|
| 1 Repeated Exposure | 1176 | 32 | exposure `1,2,4,8` |
| 2 Repeated Use | 200 | 200 | reuse `0,1,2,4,8` × 5 methods |
| 3 Same Input / Future Utility | 16 | 16 | useful vs unused context |
| 4 Idle Reasoning | 48 | 48 | ticks `0,1,2,4,8,16` |
| 5 Matched Compute | 96 | 96 | 6 budgets × 2 schedules |
| 6 Idle Consolidation | 280 | 40 | ticks `0,2,4,8,16` |
| 7 Unknowable Bit | 56 | 56 | ticks `0..32` |
| 8 Memory Revision | 216 | 216 | `3×3×3` grid per seed |
| 9 Temporary/Persistent | 16 | 16 | local vs cross-episode |

## 各 Toy 的具体协议

### Toy 1 — Repeated Exposure Consolidation

每个 seed 生成一个随机 unit key/value fact。它被真实 external exposure
`1/2/4/8` 次；每次 exposure 均执行一次 external delta write、一次当前 fact
query 的 conserving consolidation 和一次 decay。随后加入相同长度 32 的随机
interference stream，每个干扰 fact 同样只作为一次真实 external event 处理。
最终清除 `F`，以 `M@target_key` 测量 retention。每个 exposure 条件有 8 个
独立 seed；intermediate exposure/interference tick 也写入 raw records。

### Toy 2 — Repeated Use Consolidation

每个 seed 首先生成 8 个正交 key 的事实，目标 fact 与 7 个 distractor 各只
external exposure 一次。之后目标事实不再输入，只在内部被 query
`0/1/2/4/8` 次，再经过固定 32-step interference stream，最终清 `F` 后读取
`M`。该设计保证不同 reuse 条件的 external target evidence 相同。

量化比较五种方法：B2 single persistent、B3 uniform transfer、B4
non-conserving replay、B5 no-idle、B6 full ET-RCM。B3 每 tick 对所有方向转移
`gamma/key_dim`，匹配一次 rank-one query 在各向同性假设下的每方向预算；若对
所有方向直接使用完整 `gamma`，其总 transfer budget 将是 ET-RCM 的 32 倍。
B2 使用 `rho_slow`，是有意设置的强基线。B4 只向 `M` 加副本而不从 `F`
扣除，用于暴露 self-amplification。

### Toy 3 — Same Input, Different Future Utility

两个 episode 接收完全相同的 fact event，不含 remember/importance/permanent
标签。它们只有二维 task-context 不同，分别对应未来任务会使用或不使用该
fact。一个单层 sigmoid access gate 通过下游任务损失训练 300 Adam steps；
useful context 有四次可微 consolidation opportunity，unused context 只承担
slow-state pollution cost。报告的是学得 gate 导致的 slow retention。

这是一个很小的 mechanism demonstration：下游损失本身提供 future utility
学习信号，因此它不能证明一般场景中模型会自动发现“重要性”。

### Toy 4 — Idle Reasoning

每个 seed 生成 128 个平衡 graph episodes：一半存在长度 `1..8` 的可达路径，
另一半由两个断开分量构成。外部 graph 给定后不再输入新 event。latent
reachability state 使用确定性一步更新
`h <- clamp(h + h @ adjacency, 0, 1)`，运行 `K=0/1/2/4/8/16` 次后判断
target 是否可达。

该更新是为了隔离“内部时间能否承载迭代计算”，不是训练得到的完整 ET-RCM
gated MLP；因此 Toy 4 的 positive result 只说明多步 latent transition 有用。

### Toy 5 — Query-Time Compute Control

使用与 Toy 4 同分布的 128 episodes。Condition A 在 idle period 运行 K 次；
Condition B 冻结 state，在最终 query 时运行完全相同的 K 次 transition。
两者 compute budget 和 transition operator 完全匹配，比较 accuracy、最终 state
差异和 future latency。这个 control 用来区分真正的新能力与单纯提前计算。

### Toy 6 — Idle Consolidation

单个 fact 只 external exposure 一次。随后在没有新 event 时沿其 key 运行
`K=0/2/4/8/16` 次 conserving consolidation，每 tick 之后 decay。每次 transfer
前后检查 `F+M`；最终清 `F`，只测 slow retention。因此 K 增长只能重新分配
已有内容的寿命，不能把该 fact 当作重新观测。

### Toy 7 — Unknowable Bit Negative Control

每个 seed 生成 4096 个独立 `Bernoulli(0.5)` target，模型没有获得任何相关
evidence。对 `K=0/1/2/4/8/16/32`，预测头固定为 maximum-entropy `p=0.5`。
记录 accuracy、confidence 和 calibration。这个版本验证评估/动力学不会凭空
改变信念；它不是对训练后复杂模型 hallucination 的充分检验。

### Toy 8 — Memory Revision

同一 key 先绑定 old one-hot value，old external exposures 为 `1/4/8`，随后有
8 次巩固形成 slow trace。再输入真实 new one-hot evidence，new exposures 为
`1/2/4`，并给予 `0/2/8` 次 new-version internal reuse。最终丢弃 `F`，对
`M@key` 的前两个 value coordinates 应用 `softmax(5·read)`，得到 `P(new)`。
每 seed 共 `3×3×3=27` 个条件。报告同时保留强旧记忆、弱新证据失败的格点。

### Toy 9 — Temporary vs Persistent Structure

生成两个输入形式相同、key 正交的 facts。episode-local fact 只在第一个
episode 被访问一次；cross-episode fact 在 8 个 episode 中各使用一次，每个
episode 后进行 decay。无 temporary/permanent 标签。最终清 `F` 并比较 slow
retention，用以检查使用统计能否产生不同寿命；尚未训练 context-dependent
write suppression。

## Baseline 覆盖范围

- **B0 No Memory MLP** 和 **B1 GRU**：已实现并通过 shape/API smoke test，但
  本轮没有把未训练的网络硬套到 associative-retention 指标上。对它们做公平
  behavioral comparison 需要单独冻结训练预算和 sequence-task protocol。
- **B2 Single Persistent Matrix**：一个 delta-rule matrix，所有内容以
  `rho_slow` 衰减；在 Toy 2 正式比较。
- **B3 Fast/Slow Uniform Transfer**：守恒但不依赖 query；在 Toy 2 比较。
- **B4 Non-Conserving Replay**：允许自我重复增加 readout 的负对照。
- **B5 ET-RCM without idle dynamics**：NULL_EVENT 不更新状态。
- **B6 Full ET-RCM**：query-dependent conserving transfer 与 idle dynamics。

因此“至少建立 B0–B6”已经完成，但目前只有与 associative retention 端点直接
匹配的 B2–B6 进入量化表；这也是 Stage 2 未获授权的限制之一。

## Gates

| gate | result |
|---|---|
| G1_math | PASS |
| G2_repeated_exposure | PASS |
| G3_repeated_use | PASS |
| G4_idle_cognition | PASS |
| G5_no_self_evidence | PASS |
| G6_memory_revision | PASS |

判定规则在读取正式结果前已编码在 `experiments/analyze_toys.py`：G2/G3 要求
group mean 随 sweep 非下降且末端严格高于起点；G4 要求最大 K accuracy 比
K=0 至少高 `0.1`；G5 要求 unknowable accuracy 与 `0.5` 的最大偏差小于
`0.05` 且 confidence range 小于 `0.01`；G6 要求至少一个 revision cell 的
`P(new) >= 0.70`。G1 来自完整 pytest 数学/状态测试，而不是从行为曲线推断。

## Negative criteria

| criterion | triggered |
|---|---|
| N1_single_memory_equivalent | True |
| N2_uniform_equivalent | False |
| N3_matched_compute_equivalent | True |
| N4_unknowable_confidence_rises | False |
| N5_revision_failure | False |

N1/N2 的 equivalence 容差为末端 retention 落后 full ET-RCM 不超过 `0.05`；
N3 要求 matched-compute accuracy 完全相同且最终 state difference `<1e-12`；
N4 定义为最大 K confidence 比 K=0 增长超过 `0.02`；N5 与 G6 使用相同的
`0.70` revision threshold。所有 G gate 都是必要条件，但重大 negative
criterion 仍可否决 Stage 2，因此本轮出现“G1–G6 通过但不授权扩展”并不矛盾。

N1/N3 are decision-relevant null results: the single persistent matrix remains
competitive on this narrow retention endpoint, and deterministic idle compute
is exactly equivalent to moving the same transitions to query time. Idle
execution reduces future latency but does not improve matched-compute accuracy
or final state in Toy 5. These outcomes are retained rather than protocol-tuned.

The Toy-2 uniform baseline uses `gamma/key_dim` on every direction per tick,
matching the isotropic per-direction budget of one rank-one ET-RCM access. B2
uses the slow decay coefficient and is therefore an intentionally strong single
persistent-memory baseline. B0/B1 are implemented and smoke-tested but are not
assigned associative-retention scores without a separately trained sequence
protocol.

## Required questions

1. **Readout conservation?** Yes within the unit-test tolerance when G1 passes;
   maximum accumulated Toy-2 conserving drift was `6.859e-16`.
   The non-conserving replay control drifted by `0.3366`.
2. **Repeated consolidation analytic formula?** `test_repeated_consolidation_analytic_solution`
   checks `F_K q=(1-gamma)^K F_0 q`; G1 result: `True`.
3. **Repeated real exposure improves retention?** `True`.
   Curve: 1: 0.0502, 2: 0.1172, 4: 0.2519, 8: 0.4679.
4. **Repeated internal use at matched exposure improves retention?**
   `True`. Curve: 0: 0.0110, 1: 0.0560, 2: 0.0945, 4: 0.1554, 8: 0.2323.
5. **Same input/different future utility learns different lifetime?** In the
   deliberately small differentiable policy toy, useful/unused retention was
   `0.1716` / `0.0013`. This is proof of implementation,
   not a general learned-importance result.
6. **Do idle ticks enable reasoning?** Toy-4 accuracy changed from
   `0.5000` to `1.0000`.
   This shows iterative latent computation, not autonomous human-like thought.
7. **Idle vs matched query-time compute?** Accuracy curves and final states were
   matched; maximum final-state difference was `0.000e+00`.
   Only readiness/latency placement differed.
8. **Does idle consolidation improve slow retention?**
   `True`; curve: 0: 0.0000, 2: 0.0927, 4: 0.1603, 8: 0.2457, 16: 0.3149.
9. **Unknowable-bit self-confidence amplification?** `False`.
   Accuracy: 0: 0.5013, 1: 0.5013, 2: 0.5013, 4: 0.5013, 8: 0.5013, 16: 0.5013, 32: 0.5013; confidence: 0: 0.5000, 1: 0.5000, 2: 0.5000, 4: 0.5000, 8: 0.5000, 16: 0.5000, 32: 0.5000.
10. **Can real new evidence correct old memory?** Best `P(new)` was
    `0.8417`; G6: `True`. Mean by new
    exposures: 1: 0.1719, 2: 0.2678, 4: 0.3388.
11. **Query-dependent better than uniform?** Negative-equivalence criterion N2
    is `False`. Full: 0: 0.0110, 1: 0.0560, 2: 0.0945, 4: 0.1554, 8: 0.2323;
    uniform: 0: 0.0199, 1: 0.0203, 2: 0.0208, 4: 0.0216, 8: 0.0230.
12. **Fast/slow better than single persistent memory?** Not established when N1
    is triggered (`True`). Single:
    0: 0.3159, 1: 0.3157, 2: 0.3156, 4: 0.3153, 8: 0.3146.
13. **Independent value of endogenous time?** It provides pre-query readiness
    and a slot for consolidation, but Toy 5 does not show a matched-compute state
    or accuracy advantage. Independent value is therefore **not established**.
14. **Enough evidence for Stage 2?** **FALSE**.
15. **Where is evidence insufficient?** The narrow single-memory baseline is not
    defeated; matched idle/query-time compute is equivalent; Toy 3 uses a tiny
    synthetic context policy; and no language-scale learned dynamics or causal
    state intervention has been tested.

## Temporary vs persistent structure

Toy-9 slow retention was `0.0498` for the episode-local rule and
`0.2457` for the cross-episode repeatedly used rule. Inputs had the
same form; only later task-use statistics differed.

## 统计解释与主要限制

曲线值是 8 个 seed 的 arithmetic mean；逐 seed 记录、标准差和 count 位于
`toy_summary.parquet`。本轮是机制验证，不进行多重假设校正，也没有把小样本
置信区间包装成确认性统计结论。Toy 4/5 使用确定性 reachability operator，
Toy 7 使用固定最大熵头，Toy 3 使用极小 context gate；它们分别隔离某一性质，
不等同于端到端训练的通用状态模型。

Toy 2 的 single-memory 优势表明当前 protocol 没有同时测量“选择性长期保存”
与“全局写入造成的污染/可塑性代价”。Toy 8 虽证明某些 grid cell 可修正旧
memory，但按 new-exposure 聚合的均值明显低于最佳 cell，不能据此声称已经
解决 stability/plasticity tradeoff。Toy 5 的完全等价结果意味着 endogenous
time 当前只改变计算发生的时间和 query latency，没有独立 accuracy/state
优势。以上限制直接构成 Stage 2 的 blocker。

后续 Stage-1.1 应预注册：匹配总写入/衰减预算的 single-memory
stability-plasticity Pareto、经过训练的 B0/B1 sequence baselines、具有 state
noise 或 deadline 的 idle-readiness task、以及 Toy 3/7 的端到端 learned
dynamics 版本。不得在看到结果后放宽现有 gate。

## Artifacts

- Raw: `results/raw/stage1-20260917/toy_records.parquet`
- Summary: `results/processed/stage1-20260917/toy_summary.parquet` and `results/processed/stage1-20260917/summary.json`
- Figures: `results/processed/stage1-20260917/figures`

Raw Parquet 的核心 schema 包含：`toy/condition/seed/step/phase`、sweep 参数、
behavioral metrics、四个 state norms、transfer/readout/query/dynamics metrics
及 notes。`manifest.json` 记录 seed、config hash、Git revision 和运行环境；
`latest_run.json` 与 `latest_summary.json` 只作为指针/便捷副本，正式不可变
artifact 位于带 run-id 的目录。

All negative and null arms remain in the raw Parquet. The protocol was not
modified in response to these results.
