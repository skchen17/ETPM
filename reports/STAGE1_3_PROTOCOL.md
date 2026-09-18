# ET-RCM Stage 1.3 Protocol — Frozen Before Development and Formal Outcomes

> **Can a finite persistent-state model continuously integrate an interleaved event stream, arbitrate fast and slow memory reads, and emit only when learned expression value crosses a frozen threshold—without treating its own outputs as new evidence?**
>
> **一个持续存在的有限状态模型，能否在不断接收外界信息的过程中持续更新内部状态、重新访问和巩固记忆、形成新的内部关系，并在某个内部 expression activation 超过阈值时主动产生输出，同时在没有足够信息时保持沉默？**

Protocol `stage1.3-v1`; development run `stage1_3-development-v1`; formal run
`stage1_3-formal-v1`. The source revision is `1aa65ab`. Historical Stage
1/1.1/1.2 artifacts are immutable and their selected hashes are frozen in
`artifacts/stage1_3_prior_assets.sha256`. J-space remains read-only background.

## Scope

Stage 1.3 tests continuous streaming dynamics, thresholded structured
expression, fast/slow read arbitration, thought-driven persistence, and strict
self-output/evidence separation. It does not implement a language decoder,
halting head, solved flag, scheduler, thought trace, replay database, RAG, or
permanent already-said store. Sequential graph solving is only a regression
test and cannot adjudicate a Stage-1.3 gate.

The state remains `S=(H,F,M)`. External events, NULL transitions, and
SELF_OUTPUT feedback are distinct. Only a true external event with a write
mask may execute the delta write. SELF_OUTPUT may change H after an emission;
it cannot write, strengthen, clear, or reset F/M. Emission never terminates an
episode and never resets state.

## Preserved memory law

External write remains
`F <- F + eta [v-(F+M)k] k^T`. Query-dependent consolidation remains
`Delta=gamma*a*(Fq)q^T`, `F'=F-Delta`, `M'=M+Delta`, followed by distinct
decays. `F'+M'=F+M` is checked before decay. No formal result may motivate a
post-hoc change to this law.

## Read architectures

- R0 historical joint read: `(F+M)q`.
- R1 M-only read: `Mq`.
- R2 F-only read: `Fq`.
- R3 shared-query arbitration: `g*Fq+(1-g)*Mq`, scalar
  `g=sigmoid(G(H,e))`.

R4 separate-query arbitration is authorized only after R3 shows the frozen
G20 benefit. It is not part of the initial formal comparison. R0–R3 use the
same key query and fixed-law consolidation; the read source alone changes.

## Continuous expression

At every transition the model computes `s=sigmoid(W_s Pool(H)+b_s)` and a
structured content distribution. The deployment action is deterministic for a
fixed state: emit the content argmax iff `s>=theta`, otherwise `NO_EMIT`.
Expression score is not claimed to be epistemic probability. Training labels
identify evidence configurations in which expression is useful, never a
required absolute tick. Content loss applies only when expression is valid.

The threshold candidates, precision/false-emission feasibility constraints,
and tie breaks are frozen in `configs/stage1_3.yaml`. Each architecture gets
the same three-LR, two-development-seed search and threshold-sweep budget.
Selected LRs and thresholds are frozen in a new JSON before formal seeds.

## Experiments

1. **A Evidence accumulation:** interleave weak support and unrelated events;
   no query is supplied. Compare sufficient and insufficient arms, trajectory,
   first emission, precision, recall, and latency.
2. **B Pattern discovery:** stream stable, random-frequency, accidental,
   disappearing, shifted, and reversed relations without question prompts.
3. **C Cross-time association:** store A→B, add a long distractor gap, later
   observe B→C, and test spontaneous A→C expression with M/gamma/read lesions.
4. **D Silence under noise:** run 1e3 and 1e4 pure-noise ticks and retain the
   complete development threshold PR/false-emission sweep.
5. **E Output-is-not-evidence:** after borderline emission, run 0–64 ticks
   without external evidence. Compare conserving B6 with an explicitly
   non-conserving self-replay B7 negative control.
6. **F Revision:** after spontaneous A→B expression, introduce real A→C
   evidence and measure revision latency, M change, and stubbornness.
7. **G Interleaved stream:** compare block-style and randomly interleaved
   external/NULL schedules with matched contents and transition budgets.
8. **H Arbitration:** compare R0/R1/R2/R3 at 0–8192 distractors, including
   recovery, useful/false expression, gate, interference and F/M lesions.
9. **I Thought-driven consolidation:** matched-exposure facts receive unequal
   endogenous reads without remember/utility labels; relate attributed use to
   later M-only retention after F scrub and interference.
10. **J Long stream:** run 1e3/1e4/1e5 only if all five gates, no-bypass and
    healthy revision pass. Otherwise record `NOT_RUN_BY_PROTOCOL`.

## Frozen gates

- **G18 Spontaneous evidence integration:** B6 correct emission >=0.70,
  exceeds B0 by >=0.10, sufficient-minus-insufficient emission >=0.50, and the
  margin replicates in >=6/8 seeds.
- **G19 Silence under noise:** B6 false spontaneous emission per tick <=0.01,
  while expression precision >=0.90 and recall >=0.70, with the false-rate
  criterion met in >=6/8 seeds. G18 and G19 must pass together; a silent model
  cannot pass.
- **G20 Memory arbitration:** at 2048 distractors R3 exceeds R0 recovery
  accuracy and useful expression by >=0.10 each, raises false emission by no
  more than 0.01, and the accuracy margin replicates in >=6/8 seeds.
- **G21 Thought-driven persistence:** mean within-seed Spearman correlation
  between endogenous usage attribution and later slow retention >=0.40, with
  >=6/8 seeds individually >=0.30.
- **G22 Self-output is not evidence:** B6 post-output external-write count is
  exactly zero; memory evidence magnitude rises by <=1%; expression score rises
  by <=0.05; B7 must show >=0.10 pathological amplification.

## Authorization and reporting

The 1e5 long-stream experiment and a recommendation for a small language
prototype require G18–G22, no history bypass, healthy revision, and no
self-output amplification. Stage 2 cannot start automatically and is never
trained in this stage. Any implementation correction or resource change is
append-only in `STAGE1_3_AMENDMENTS.md`; formal gates and metrics cannot change
after outcomes are visible. Negative and null outcomes remain machine-readable.
