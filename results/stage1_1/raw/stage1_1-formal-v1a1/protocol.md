# ET-RCM Stage 1.1 Protocol — Frozen Before Formal Execution

> **Can a learned recurrent state autonomously decide what to revisit, thereby allocating limited persistent-memory lifetime preferentially to information that remains useful, and can computation performed between external events change how later events are processed rather than merely shifting compute earlier in time?**
>
> **一个可学习的持续状态模型，能否自主决定接下来重新访问什么，从而把有限的长期记忆寿命优先分配给未来仍有用途的信息；同时，外部事件之间发生的内部计算，是否能够真正改变模型随后吸收新事件的方式，而不只是把相同计算提前执行？**

Protocol version: `stage1.1-v1`. This file, `configs/stage1_1.yaml`,
`configs/stage1_1_splits.json`, and the adjudication rules below are frozen
before any formal-test execution. Development-only learning-rate selection is
allowed; formal outcomes cannot change gates, splits, or metrics. Amendments
must be additive, versioned, hashed, and must preserve failed/null runs.

## Scope and Stage-1 continuity

Stage 1 established the external-write/internal-use separation, numerical
readout conservation, exposure/use-dependent slow retention, revision in some
cells, and functioning fast/slow mechanics. It did not establish a learned
query, learned idle reasoning, superiority to a strong single persistent
memory, independent value of compute placement, trained recurrent baselines,
or learned no-self-evidence. `N1_single_memory_equivalent` and
`N3_matched_compute_equivalent` remain formal blockers.

Stage-1 artifacts under `results/*/stage1-20260917` and
`reports/TOY_VALIDATION_REPORT.md` are immutable. Their hashes are recorded in
`artifacts/stage1_frozen_assets.sha256`.

## Fixed model law

State remains `S=(H,F,M)`. Only a true external write event may execute

`F <- F + eta [v-(F+M)k] k^T`.

Every `NULL_EVENT` has `write_mask=false`. Consolidation remains

`Delta=gamma(Fq)q^T; F'<-F-Delta; M'<-M+Delta`,

and must conserve `F+M` before decay. The learned components are event encoding,
initial active state, `q=normalize(W_q RMSNorm(pool(H)))`, shared recurrent
gated-MLP dynamics, optional access strength, and prediction heads. The same
recurrent parameters are used at every internal tick. No model receives a raw
history list, stored answer index, future-use marker, or importance bit.

## Training and credit assignment

AdamW, batch size, step counts, gradient clipping, seeds, and candidate learning
rates are frozen in `configs/stage1_1.yaml`. Learning-rate selection uses only
development seeds. Formal synthetic sequences use full BPTT: no detach interval
and no activation checkpointing. Claims therefore apply only to the executed
sequence lengths and do not imply ultra-long credit assignment.

## Formal experiments

- **A Learned Reuse:** one exposure per fact; downstream tasks require retrieval
  without restating values. Sweep actual reuse count and measure autonomous
  query alignment, cumulative learned access, transfer, slow retention, and
  task accuracy versus random query.
- **B Selective Persistence:** eight useful facts, no useful flag, followed by
  32/128/512 distractors. Compare trained B0/B1/B2/B3/B5/B6. G8 uses only
  state-byte-matched B2/B3 comparisons; unmatched capacity is disclosed.
- **C Frequency vs Utility:** high-frequency unused facts (8/16/32 exposures)
  compete with low-frequency useful facts (1/2 exposures plus real tasks).
- **D Learned Idle Reasoning:** graph edges are external events; after a query,
  shared learned `Phi` receives 0/1/2/4/8/16 NULL ticks. Report in-distribution
  and unseen longer paths, query/H trajectories, entropy, and accuracy.
- **E Interleaved Time:** compare `Phi_B(Phi_0^K(S_A))` with
  `Phi_0^K(Phi_B(S_A))`; events and transitions are matched. Behavioral margin,
  not state distance alone, determines G11.
- **F Consolidate Before Interference:** move matched internal ticks before or
  after a distractor stream and compare delayed retention/task accuracy.
- **G Reason Before Interruption:** move learned reasoning ticks before or after
  an interruption and measure answer accuracy/readiness.
- **H Learned Unknowable:** a learned head/core is jointly trained on knowable
  tasks and random-label unknowable tasks. Sweep NULL ticks; record accuracy,
  confidence, entropy, ECE, and Brier.
- **I Stability/Plasticity Pareto:** sweep old/new exposures and new reuse for
  ET-RCM, single, uniform, and GRU; report all cells and a Pareto frontier.
- **J Long stream:** 1e3/1e4/1e5 events runs only if the intermediate frozen
  adjudication authorizes it. Otherwise a report records `NOT_RUN_BY_PROTOCOL`.

## Baselines and capacity accounting

B0 no-memory recurrent MLP, B1 GRU, B2 two-head single persistent delta memory,
B3 uniform fast/slow transfer, B5 ET-RCM without NULL updates, and B6 full
ET-RCM are trained with the same examples, optimizer steps, batch size, and
external event counts when the task applies. B2 has two persistent matrices so
its memory-state float count matches `F+M`; it is not given a smaller memory.
Parameter count, persistent-state bytes, transitions, and compute budget are
recorded. B0/B1 state sizes cannot be simultaneously matched with both
parameter count and matrix-state size; the exact mismatch must be reported and
they cannot alone adjudicate G8.

## Ablations

A1 gamma=0; A2 uniform transfer; A3 rho_fast=rho_slow; A4 no NULL dynamics;
A5 random query; A6 learned query with frozen H; A7 non-conserving replay. Each
uses the same evaluation episodes as its parent formal arm.

## Frozen gates

- **G7 learned query:** full model exceeds random-query alignment by >=0.10 and
  task accuracy by >=0.05; reuse-to-retention endpoint increase >=0.02.
- **G8 selective persistence:** in at least one frozen high-pressure regime,
  full ET-RCM exceeds state-byte-matched B2 in both useful accuracy and useful
  retention by >=0.05.
- **G9 usage over uniform/frequency:** lower-frequency useful facts exceed both
  uniform transfer and high-frequency unused retention/utility by >=0.05.
- **G10 learned idle reasoning:** learned shared recurrence improves Kmax over
  K0 accuracy by >=0.10 on formal data, with autonomous queries.
- **G11 interleaved time:** one preregistered schedule improves the target
  behavior by >=0.05 under matched events/transitions/FLOP accounting.
- **G12 learned no-self-evidence:** unknowable accuracy remains within 0.05 of
  chance; confidence inflation <=0.03; ECE degradation <=0.05.
- **G13 revision Pareto:** at least one preregistered balanced cell reaches
  P(new)>=0.70 without unrelated-retention collapse >0.20.

## Frozen negative criteria

N6 query collapse: alignment improvement <0.10 or query-direction variance
collapses. N7: B2 is not worse than ET-RCM in every capacity-pressure regime.
N8: no >=0.05 interleaved behavioral advantage. N9: unknowable confidence rises
>0.03 without accuracy gain. N10: no balanced revision cell reaches 0.70.
N11: memory lesion reduces task accuracy by <0.05, or a model/runner history
bypass is detected. N11 invalidates the affected experiment.

## Authorization

Stage 2 requires G1–G6 to remain valid, all core gates G7/G8/G10/G11/G12 to
pass, no critical negative criterion, fair trained baselines, and a passing
no-history-bypass audit. Otherwise
`STAGE2_LANGUAGE_MODEL_AUTHORIZED = FALSE`. No language-model training belongs
to Stage 1.1.

