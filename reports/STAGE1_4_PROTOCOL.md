# ET-RCM Stage 1.4 Protocol — Frozen Before Development

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**

Protocol `stage1.4-v1`; immutable Stage 1–1.3 parent commit
`360693ab0ce635831bd714cf901a5702ffdbb0ae`. This protocol, the YAML
configuration, and the prior-asset manifest are committed before Stage 1.4
implementation or development results. Corrections are append-only amendments;
formal endpoints and gates cannot be edited after formal outcomes are visible.

## Boundaries and preserved laws

The persistent state is exactly `(H,F,M)`. Only an external event executes
`F += eta * [v-(F+M)k] k^T`; NULL and SELF_OUTPUT never execute this path.
Consolidation is `Delta=gamma*a*(F q_F)q_F^T`, `F'=F-Delta`, `M'=M+Delta`.
`F'+M'=F+M` is checked before `rho_fast/rho_slow` decay. Causal utility is
measured offline first; it does **not** alter this law in the formal model.
There is no history cache, target-key query label, importance/future-use label,
halting module, planner, or language decoder. Stage 1.3's expression head is
not optimized as the primary Stage 1.4 objective.

## Architectures and fair budget

B0 has recurrent H without persistent memory; B1 is a GRU; B2 has one
persistent delta-rule matrix; B3 uses historical `(F+M)q`; B4 uses one shared
query with scalar F/M arbitration; B5 is the primary separate-query model,
`q_F=Q_F(H)`, `q_M=Q_M(H)`, independently RMS-normalized reads, and a
two-way softmax gate conditioned on H and the current event. B6 sets `gamma=0`
in B5. B7 uses a fixed random M query and is independently trained. B8
shuffles M across episodes only at evaluation. B0–B7 receive identical seed,
step, batch, optimizer and learning-rate search budgets. Parameter count,
persistent bytes, actual transition count and elapsed compute are disclosed;
architecture differences are not silently described as parameter-matched.

## World and leakage control

Four frozen stochastic world families are generated from independent
per-episode RNG streams. Latent transition requires filtering a noisy hidden
state; long-gap relation has an early A→B event, unrelated distractors, later
B→C and an A-dependent future token; latent regime uses sparse early regime
evidence to predict later events; distractor-heavy streams have rare predictive
events amid frequent uninformative events. Models receive only the current
external event or a zero NULL event, never the future target or target key.
Prediction targets are shifted strictly after the current external step.
Horizon losses for `h=1,2,4,8` have weights `0.4,0.3,0.2,0.1`. The same
external history and targets are paired across NULL-tick conditions.

## Development and formal selection

Development uses seeds 6401/6402, two learning rates, 100 steps, batch 32,
length 16, and all eight trainable models. Choose each architecture's rate by
minimum mean held-out multi-horizon world loss, with lower LR on a tie.
Freeze selections and their hash before touching formal seeds. Formal uses
fresh training seeds 7401–7408, 160 steps each, the selected LR, 32 held-out
episodes per family/seed, and 32 intervention pairs/items per seed. Inference
world seeds are separate from training and development. Experiments at 2048
distractors may use a smaller paired confirmatory sample, never fewer than
eight independently trained model seeds. A resource amendment must precede
the affected run and retain aborted logs.

## Registered analyses

The episode is the observational unit and independent training seeds are
the replication units. Primary losses are mean future-event cross-entropy,
averaged over valid shifted horizon targets, then episode, then seed, then
equal-seed mean. Differences use paired episodes and a seed bootstrap.
No test target is used for model selection.

- **A: Predictive internal time.** Compare K=0,1,2,4,8,16 NULL ticks before
  the next external event, with frozen-H NULL, random recurrent NULL, and
  matched computation after that event where the target timing permits it.
  Report all four horizon losses and total transitions/latency. The after-event
  arm is not considered pre-event forecasting.
- **B: Autonomous reactivation.** Evaluate long-gap relations with 128/512/
  2048 distractors and no explicit query. Pair intact B5 with M lesion, F
  lesion, fixed random q_M, shuffled M, gamma-zero, B0, B3 and B4. Record
  q_F/q_M, raw/normalized reads, gates, effective contribution, H shift,
  prediction loss and target availability audit.
- **C: Same-H peripheral swap.** Clone a numerically identical H, substitute
  F and/or M from another independent episode, and apply the same zero event
  for 1/2/4/8 ticks. Record H-distance and prediction Jensen–Shannon
  divergence. State distance alone never establishes this gate.
- **D: Workspace mediation.** For the C pair, run one tick with B peripheral,
  then replace only its H by A's H after tick 1. Continue with B peripheral
  unchanged. Compare restored and unrestored effects at 2/4/8. Classify
  WORKSPACE_MEDIATED, PERIPHERAL_PERSISTENT, MIXED, or NOT_IDENTIFIED using
  the fraction of paired prediction effect remaining (≤0.25, ≥0.75,
  intermediate, or unstable/too small, respectively); this is descriptive,
  not a gate.
- **E: Causal use.** Attribute each identified memory-item direction by a
  one-item rank-one lesion; `CU_i(h)=L_future^(-i)-L_future^full` on identical
  future targets. Also measure future-H effect. Read count/magnitude and
  exposure are distinct covariates. Lesion touches one item/direction only.
- **F: Retention explanation.** Clear F after interference and measure M-only
  retention. Compare exposure, read usage and CU via Spearman and held-out
  rank regression. Incremental held-out R² of CU is against exposure+read
  predictors, with test folds grouped by episode/seed. Correlation is not
  itself a causal result.
- **G: Consolidation intervention.** Only if development F shows positive
  incremental CU, match high-/low-CU items on exposure/read count/approximate
  magnitude, block their F→M transfer separately, and measure paired future
  loss. If condition unmet mark NOT_RUN_BY_PROTOCOL, not negative evidence.
- **Safety control.** Normal SELF_OUTPUT cannot external-write. A separate
  `B_bad` intentionally applies the delta external-write equation to a
  SELF_OUTPUT key/value. The safety audit is VALID only if repeated output
  measurably increases memory readout and repeat-output propensity in that
  pathological arm. Otherwise audit INVALID, not a pass for the primary model.

## Gates and decision discipline

G23 requires B5 mean `L(0)-L(4)≥0.01`, ≥6/8 positive training seeds, and
positive margins versus frozen and random NULL controls at matched pre-event
compute. G24 requires B5 mean loss at 512/2048 distractors to be ≥0.01 lower
than each of M-lesion, random-q_M and B0, with ≥6/8 seeds positive for every
comparison and no explicit query. G25 requires same-H swaps to cause both
mean H distance and prediction JS difference ≥1e-5 at an a priori 1–8 tick
endpoint, replicated in ≥6/8 seeds. G26 requires incremental held-out R²
of CU ≥0.02 beyond exposure/read usage in ≥6/8 seeds **and** a ≥0.01 greater
future-loss harm from blocking high-CU than matched low-CU consolidation.
If G is not authorized, G26 cannot pass. A confidence interval and all
individual seed values accompany every gate; thresholds are not adjusted.

No small LM is trained. Recommendation can only be TRUE if G23–G26 pass,
information-path audit passes, and valid no-self-evidence safety audit passes.
Even a recommendation requires a new authorization before LM work. Null and
negative outcomes, failed controls, and amendments remain visible. Toy-scale
prediction, access, causal intervention and retention are never conflated
with consciousness, general intelligence or infinite capacity.
