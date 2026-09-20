# ET-RCM Stage 1.5 Preregistered Protocol

> **What dynamical and causal structure does the ET-RCM architecture itself possess, and can a learned state-dependent routing policy select the peripheral information whose finite use actually improves future computation?**

> **ET-RCM 架构自身究竟具有怎样的动力学与因果结构；同时，一个可学习的状态依赖路由机制，能否从外围持续状态中选择那些经真实有限干预验证、确实能够改善未来计算的信息？**

This independent version starts from immutable Stage 1.4 revision
`13671106e0602326e9fd84de9493568e24ff003f`. Its existing source,
configs, results and reports are read-only reference data. All new outputs
live under `stage1_5` paths. No historical result is overwritten or
readjudicated. This protocol and `configs/stage1_5.yaml` must be committed
before formal jobs begin; later scientific changes require an append-only
amendment and a new run ID.

## Frozen design

State remains `S=(H,F,M)`. Only validated external EVIDENCE/NOISE may use the
delta write. NULL and SELF_OUTPUT may not. Transfer remains
`Δ=γa(Fq_F)q_Fᵀ`, `F'=F−Δ`, `M'=M+Δ`, with `F'+M'=F+M` before decay.
No LM, planner, halting policy, utility-weighted transfer, extra memory tier,
or independent M recurrent dynamics is introduced.

Two development seeds (`8401–8402`), two learning rates (0.001, 0.0003)
and equal 100-step budgets are assigned to each of eight inherited
architectures. Per-architecture learning rate is chosen by held-out
equal-family multi-horizon CE before formal evaluation. Eight fresh
independent seeds (`8501–8508`) receive 160 steps each at the selected rate.
Train batches are 32, optimizer AdamW, histories 16 external events, four
world families and horizons 1/2/4/8. Capacity scaling uses the sparse grid
in the config; its additional model cells receive the same development and
formal budget. There is no target/history query in model inputs.

Each paired comparison aggregates episodes first, then equal-weights seeds;
six of eight seeds must show the specified direction, in addition to the
mean margin. Confidence intervals are seed bootstrap (2,000 resamples),
reported but not substituted for the registered thresholds. FP32 is primary.

## Track A: architecture characterization

A1: initialize from real external histories; run NULL ticks through 1024.
Record every-tick state/read/query/transfer/prediction metrics. Classify
fixed-point-like, bounded drift, oscillatory or divergent descriptively.
A2: independently perturb H, F and M by finite epsilon 0.001/0.01 and
measure H/full-state growth to tick 128; JVP cannot replace finite curves.
A3: train-only ridge probes predict historical visible event values at lags
1–512 from H, F or M separately; report held-out log-loss gain above
train-frequency constant predictor, never call it mutual information. Repeat
with equal/altered decay, separating imposed timescale from dynamics/usage.
A4: sparse `(d_H,d_M)` grid `(32,8),(64,16),(128,32),(256,64),(256,128)`;
measure items, distractors 0/32/128/512/2048/8192, retention, interference,
future CE, state bytes and compute. No infinite-capacity inference.
A5: exactly construct H, F, M, HF, HM, FM, HFM swaps between paired states;
measure future H difference, target CE and prediction JS at ticks 1/2/4/8;
calculate signed CE interactions `E_AB−E_A−E_B` and optional third order.
A6: train-only regularized linear probes predict future H, logits and events
from H, HF, HM, HFM; use strictly held-out episodes. Observational
sufficiency is not causal anatomy.

## Track B: peripheral routing

B1 localizes the Stage 1.4 joint F/M effect using A5, including F×M.
B2 swaps F or M while restoring only the corresponding *effective read*
to the intact branch at each tick; stored memory remains swapped. Report
absolute JS/CE and `1−effect_restore/effect_swap` only when denominator
exceeds the registered minimum. An unmediated effect is labeled
`UNMODELED_PERIPHERAL_PATHWAY`.
B3 on long-gap worlds compares learned, zero, norm-matched random,
cross-episode shuffled and correct historical oracle reads. Oracle is
constructed solely from the early actually observed key/value and a saved
pre-interference memory state; no future target, label or post-gap state is
consulted. Oracle and controls enter the *same* read interface, checkpoint,
episode, bridge and tick budget. B4 recomputes an oracle choice from the
updated H every tick and compares static/closed-loop/learned/no-read.

B5 is permitted only after G31 passes. Candidate reads must come from
actual historical memory snapshots; each finite intervention is scored
`A_i = CE(no-read)−CE(inject i)` at a fixed future target. Gradient, JVP,
query cosine, read norm and exposure count cannot replace this label.
B6 is permitted only if B5's held-out advantage signal is stable; a
state-conditioned router is trained from `softmax(βA)` without target keys,
future-use tags or remember labels. B7 is permitted only if G32 passes;
every NULL tick routes anew. Conditional non-execution is recorded as
`NOT_RUN_BY_PROTOCOL`, never imputed as failure or success.

Secondary state-conditioned causal-utility versus retention analysis is
permitted only after G32. Consolidation law is never changed this stage.
An exploratory FP32/BF16/FP32-shadow precision regression cannot change any
primary gate.

## Gates and decisions

Numerical thresholds are in `configs/stage1_5.yaml`, frozen here.
G27 requires all-finite, bounded norm/JS/finite-perturbation growth.
G28 requires reproducible held-out lag-profile separation beyond trivial
readout and explicit decay ablations. G29 requires replicated predictive
effects from a single F or M channel and quantified pairwise interactions.
G30 requires read restoration to remove at least half the relevant swap JS.
G31 requires oracle CE better than zero, random, shuffled and learned by
0.01, with 6/8 positive seeds. G32 requires state-only distilled routing
to beat zero/random by 0.01 on held-out worlds and recover ≥50% of oracle
gain. G33 requires K4 versus K0 CE gain 0.01, 6/8 seeds, beyond
frozen/random NULL controls. G27–G33 must all pass before recommending a
small sequence prototype; recommendation does not authorize training it.

For G28, lag-profile AUPRC means trapezoidal area of nonnegative held-out
decoding gain versus log2(lag), normalized to unit width; H/F and F/M
differences require 0.02 in the prespecified ordering, 6/8 seeds. Equal
decay ablation is reported and must not give an indistinguishable profile
if a nontrivial dynamics/usage claim is made.

For G29, paired absolute prediction JS at tick 4 must exceed 1e-5 in
F or M alone and FM jointly, 6/8 seeds. G27 checks maximum norms and JS
over the complete 1024-tick sample, and empirical perturbation growth at
tick 128; values exceeding thresholds or any nonfinite outcome fail.

All raw rows include run/model/seed/episode/world/state, clocks, state and
read metrics, intervention identity, CE/JS, exposure/read/retention, model
size/compute, and paths/hashes for large tensors. Negative outcomes remain
in machine-readable files. The final report answers all 20 requested
questions and distinguishes stored, read, predictive, causally useful and
retained information.
