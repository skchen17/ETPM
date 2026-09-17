# ET-RCM Stage 1.2 Protocol — Frozen Before Formal Evaluation

> **Does ET-RCM actually learn functional memory addressing and selective lifetime allocation, and can learned internal recurrence perform necessary sequential computation without converting self-generated activity into new evidence?**
>
> **ET-RCM 是否真正学会了功能性的记忆寻址与选择性的记忆寿命分配；同时，可学习的内部递归是否能够承担必要的逐步计算，而不会把自身产生的内部活动误当成新的外部证据？**

Protocol `stage1.2-v1`; formal run `stage1_2-formal-v1`. This protocol,
`configs/stage1_2.yaml`, splits, seeds, metrics and thresholds are frozen before
formal outcomes. Stage 1 and Stage 1.1 reports remain immutable and are not
re-adjudicated. Their hashes are in `artifacts/stage1_2_prior_assets.sha256`.
J-space remains read-only background.

## Scope and hypotheses

Only four hypotheses are adjudicated: H1 functional addressing, H2 selective
persistence scaling, H3 sequential internal computation, and H4 no
self-evidence. Signed query cosine is descriptive and cannot retroactively
change Stage-1.1 G7. Stage 2 decoder training is outside this run.

## Preserved memory laws

Only a true external write executes
`F <- F + eta [v-(F+M)k]k^T`. A NULL transition cannot call this path.
Consolidation remains `Delta=gamma*a*(Fq)q^T`, `F'=F-Delta`, `M'=M+Delta`.
`F'+M'=F+M` is tested before decay. No result-driven modification of this law
is permitted.

## Reporting semantics and information audit

Stage-1.2 lesion records use four non-overloaded names:
`pre_lesion_slow_retention`, `post_lesion_slow_retention`,
`pre_lesion_accuracy`, and `post_lesion_accuracy`. The historical Stage-1.1
snapshot field is not changed. Models receive only the current event and
state. Query events never contain answer values; knowable and unknowable
adjudication episodes use identical FACT/QUERY kinds and shapes.

For sequential tasks, graph writes are followed by an H reset. The goal event
contains start/target only, never graph edges or label. One memory query/read is
allowed per transition. Consequently H cannot carry the written graph into the
reasoning phase and a single tick cannot read multiple independent edges.

## Development and formal separation

Each B1/B2/B3/B5/B6 architecture receives the identical development budget:
three candidate learning rates, two development seeds, and 180 steps per run.
Each architecture selects its own mean-best LR before formal execution. Formal
training uses eight fresh seeds 3201–3208 and full BPTT over executed toy
sequences. The endogenous-interference count is selected only on development
data by closeness to the center of the frozen 40–80% accuracy band.

The capacity sweep is frozen at 0/32/128/512/2048 distractors. A 32768-event
sweep across five learned architectures and eight seeds exceeded the fixed
two-GPU budget; 2048 was selected before formal outcomes, not after observing a
curve.

## Experiments

- **A Functional query intervention:** clone an identical memory/state and use
  original, target-parallel, target-perpendicular, sign-flipped, norm-matched
  random, target-zeroed, or strongest-nontarget-zeroed q. Record accuracy,
  loss, read, H change, signed/absolute cosine, squared projection and
  target/nontarget margin.
- **B Exposure x reuse:** evaluate the complete 6x7 frozen grid. Reuse presents
  a key/task cue but never its value. After matched interference, H and F are
  scrubbed for the slow-only endpoint. Fit linear, interaction and log models;
  report seed/bootstrap uncertainty and an exposure-equivalent reuse estimate.
- **C Selective scaling:** 16 used facts compete with the frozen distractor
  sweep. Compare independently tuned B1/B2/B3/B5/B6. Efficiency is exactly
  `mean useful-fact final behavioral accuracy / persistent_state_bytes`.
- **D Autonomous multi-memory selection:** after three writes and H reset, one
  TASK_CUE identifies an operation but supplies no key IDs or values. Only NULL
  transitions follow. Evaluate XOR/equality/modular comparison tasks and query
  trajectories against A/B/C memory directions.
- **E Sequential computation:** independently stored path edges, H reset, one
  query/read per transition, lengths 1–8 and K=0/1/2/4/8/16. Train on L<=5;
  L=6–8 is the frozen OOD stratum. A bypass-test failure invalidates G16.
- **F Endogenous-time necessity:** the same goal and interference events are
  used in both schedules; K NULL transitions move from before to after
  interference. Total transitions are equal and only timing differs.
- **G No self-evidence:** both strata use one FACT-shaped event and one QUERY;
  the unknowable target is independent and the relevant key was never written.
  No condition bit or special event kind is used. Sweep K through 64.
- **H Storage versus use:** repeat query interventions after the high-pressure
  stream and lesion F-only, M-only, or both with explicit pre/post fields.
- **I Long stream:** run 1e3/1e4/1e5 only if G14/G15/G17 and no-bypass pass.
  Otherwise record `NOT_RUN_BY_PROTOCOL`.

## Frozen gates

### G14 Functional Addressing

PASS iff original-minus-target-removed accuracy >=0.08, target-removal loss
increase >=0.10, target-removal drop exceeds strongest-nontarget-removal drop
by >=0.05, and at least 6/8 seeds show >=0.05 target-removal accuracy drop.
Signed cosine cannot adjudicate this gate.

### G15 Selective Persistence Scaling

At 2048 distractors, B6 must exceed state-byte-matched B2 accuracy by >=0.05,
have a degradation-slope advantage >=0.015 per log(1+Nd), and have strictly
higher frozen efficiency. Both behavioral and state-normalized criteria are
required.

### G16 Sequential Internal Computation

On OOD L=6–8, K=8 minus K=1 accuracy must be >=0.15, the aggregate tick curve
must increase systematically, and at least 6/8 seeds must show >=0.10 gain.
The shared learned dynamics and hard-bottleneck audit must pass.

### G17 No Self-Evidence

The same model must improve knowable K=16 over K=0 by >=0.05; unknowable K=64
accuracy must remain within 0.05 of chance; confidence inflation <=0.03; ECE
degradation <=0.05. Any explicit condition cue invalidates the gate.

## Endogenous-time and authorization decisions

`ET_STATUS_SUPPORTED` requires before-minus-after accuracy >=0.05 and >=6/8
seeds with >=0.03; state distance or retention alone cannot pass it.

Memory-LM Path A requires G14/G15/G17, healthy inherited revision, and no
bypass. Continuous-Cognition Path B additionally requires G16 and
`ET_STATUS_SUPPORTED`. Failure of endogenous time does not veto the simpler
memory-only path. No language model is trained in Stage 1.2.

## Amendments and reporting

Implementation bugs are corrected only through additive amendments preserving
all earlier outputs. All failures/nulls, selected hyperparameters, exact
capacity/compute accounting, raw Parquet, summaries, manifests, configs, git
revisions and SHA256 lists are retained.
