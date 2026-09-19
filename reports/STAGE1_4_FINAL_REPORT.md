# ET-RCM Stage 1.4 Final Report

> **Can a continuously running predictive state autonomously reactivate old persistent information when that information causally improves future prediction, and does such causal usefulness explain which transient states should acquire longer memory lifetimes?**

> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**


Formal run `stage1_4-formal-v1a1` contains 64 independently
trained model/seed cells (8 trainable architectures × 8 new seeds),
64 corresponding evaluation shards, and
426,496 machine-readable evaluation records.
Development used 2 seeds × 2 learning rates × 8 architectures, 100 steps
each; all selected 0.001 by held-out multi-horizon CE. Formal training used
160 steps, batch 32, AdamW and four equal-budget world families. Source,
config, selection, checkpoint and record hashes are verified in
`results/stage1_4/processed/stage1_4-formal-v1a1/integrity.json` and the frozen manifest.

## Registered outcome

| Gate | Verdict | Primary observation |
|---|---|---|
| G23 | FAIL | ΔL4=0.0018; 4/8 |
| G24 | FAIL | M lesion margin=-0.0000 |
| G25 | PASS | JS=0.0013; 8/8 |
| G26 | FAIL | incremental R²=-0.0549; G=NOT_RUN_BY_PROTOCOL |

Mediation classification: **MIXED** (not a gate).
No target/history bypass: TRUE.
Normal SELF_OUTPUT no external write: TRUE.
Pathological-control audit valid: TRUE.
`SMALL_LM_PROTOTYPE_RECOMMENDED = FALSE`.
`LANGUAGE_MODEL_TRAINING_STARTED = FALSE`.

## Experimental methods and exact comparisons

**Worlds.** Latent-transition episodes filter a noisy 8-state trajectory;
latent-regime episodes infer one of four enduring token regimes from an early
event; long-gap relation episodes write B→A in 1/2/4 genuine exposures,
insert 128/512/2048 unrelated memory writes, observe C→B, then forecast an
A-dependent future token; distractor-heavy episodes hold a rare cue amid
frequent future-irrelevant writes. Training length is 16 external events.
Targets are future-shifted by horizons 1/2/4/8 and never supplied as current
events or query labels. The formal 2048 gap is out of training distribution.

**A, predictive internal time.** B5 predicts after 0/1/2/4/8/16 NULL ticks
from the same state/history. Frozen-H and fixed random recurrent controls use
the same tick count. An additional matched timing arm places K compute before
versus after the next event while forecasting the subsequent event; latency
differs and it is not substituted for pre-event prediction. See the full
family×horizon×K curve and all eight seed margins in
`PREDICTIVE_CONTINUOUS_DYNAMICS_STAGE1_4.md`.

**B, autonomous reactivation.** Paired same-checkpoint M/F lesions, random
q_M and shuffled M are applied before the bridge. B0/no-memory, B1/GRU,
B2/single memory, B3/joint, B4/shared, B6/gamma-zero and B7/random query are
separately trained with the same hyperparameter search budget. CE is measured
after four post-bridge NULL ticks. No explicit A key is input. See
`AUTONOMOUS_MEMORY_REACTIVATION_STAGE1_4.md` for every 512/2048 contrast.

**C/D, causal state and mediation.** H is cloned exactly, F/M are swapped
across episodes, then both states receive 1/2/4/8 zero-input ticks. C measures
future-H norm distance and prediction JS; state distance alone cannot pass.
D replaces only H after tick one in the swapped arm, leaving F/M swapped,
then measures the remaining tick-8 JS ratio. See the two intervention reports.

**E/F, causal use versus retention.** Thirty-two item directions per seed
have M rank-one key-direction lesions; future CE differences define CU.
Read usage, true exposures and M-only cosine retention after 128 distractors
are separate quantities. Fourfold held-out rank regression tests whether CU
adds value beyond exposure/read. Learned keys are nonorthogonal, so lesion
overlap is a limitation. All seed correlations and R² values are preserved.
An additional clearly marked exploratory replay measures hard/soft read
frequency and does not alter G26.

**G and safety.** The pre-formal development CU decision was
`FALSE`; high-/low-CU
consolidation blocking is `NOT_RUN_BY_PROTOCOL` when unauthorized.
This decision cannot be reversed by formal results. Normal SELF_OUTPUT
never calls external write; the separate B_bad positive control deliberately
does and is judged by memory readout plus a clearly labeled fixed propensity
proxy, not a learned expression policy.

## Answers to the 18 required questions

1. NULL ticks: G23 fails; K0−K4 CE=0.0018 (4/8 seeds positive).
2. Matched compute: frozen/random K4 margins are 0.0018/0.0297; post-event timing losses are {'idle_pre_event_then_event': 1.6065779526058275, 'post_event_ticks': 1.6044805343941941}, with post-event latency K.
3. Autonomous long-gap access is not established under G24; no target query was supplied.
4. M-lesion minus intact CE=-0.0000; compare direction and replication in G24 table.
5. Random-q_M minus intact CE=0.0000; a descriptive q trajectory alone is insufficient.
6. Separate B5 versus shared B4 CE at 512/2048 gaps: 1.7987 versus 1.7961; this is a separately trained architecture comparison.
7. Same-H peripheral swap changes future H by 2.7392 and prediction JS by 0.0013; G25 passes.
8. H restoration leaves tick-8 JS ratio 0.7355; classification MIXED.
9. Yes at this toy intervention's scope: G25 establishes that swapping F/M at identical H changes later H and prediction. This supports F/M as causal peripheral computational state here, not a general causal-memory claim. Classification: MIXED.
10. Exposure/read-magnitude/CU associations are listed seed-by-seed; the exploratory hard-count audit gives mean CU incremental R² -0.0689 after count control. No read norm alone proves causal use.
11. Registered CU incremental held-out R²=-0.0549; G26 fails. Count adjustment is secondary and cannot change G26.
12. High-CU consolidation block is NOT_RUN_BY_PROTOCOL; no harm contrast can be claimed.
13. Selective predictive reuse is not established by the combined reactivation and causal-use tests.
14. The world-prediction objective replaces Stage 1.3 expression training, but solving its objective mismatch requires predictive and retrieval gates, not low training loss alone.
15. Expression should remain secondary until predictive/retrieval controls validate content; it was not a Stage 1.4 gate.
16. Small LM prototype recommendation is FALSE; no LM was trained.
17. Gate-specific bottlenecks: predictive dynamics G23=FAIL, autonomous retrieval/read interface G24=FAIL, peripheral causal effect G25=PASS, and causal-utility persistence/consolidation G26=FAIL; model capacity is not isolated by this protocol.
18. All effects remain toy-scale: 24-symbol worlds, 64-d hidden state, short training histories, controlled interventions, and synthetic long gaps. No consciousness, general intelligence or infinite capacity is inferred.

## Scientific boundaries

Predictive usefulness, read/access, causal influence, and persistent retention
are reported separately. No result is called consciousness, human-like
thought, self-awareness, general intelligence, infinite context/capacity, or
causal memory merely because a prediction score improves. Failures and null
effects remain in the raw Parquet shards and gate JSON; no threshold was
changed after formal outcomes.

The first Stage 1.4 development/formal attempt is preserved but invalidated
by Amendment A4: its latent-transition event leaked hidden z in the key field.
The corrected v1a1 run uses new development/formal seeds and enforces a
partially observed latent key plus distinct B/C relation symbols. No metric
from the invalid attempt contributes to this adjudication.
