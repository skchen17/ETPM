# Stage 1.4 negative and null results

The corrected formal run is `stage1_4-formal-v1a1` (8 architectures × 8
fresh training seeds). The earlier v1 attempt was invalidated by Amendment
A4 after discovering a hidden-state leak in Family A; its 57 completed
training cells and one evaluation cell are retained but excluded.

- **G23 predictive internal time — FAIL.** Mean weighted/equal-family
  `L0−L4=0.00180`, seed-bootstrap 95% CI `[-0.00303,0.00693]`, positive
  in 4/8 seeds. The `0.01` floor and 6/8 replication rule were not met.
  Matched post-event compute had CE `1.60448` versus pre-event `1.60658`,
  but incurs K ticks of future latency and is not pre-event prediction.
- **G24 autonomous M reactivation — FAIL.** At 512/2048 distractors,
  M-lesion minus intact CE was `-0.000024`; random-q_M minus intact was
  `+0.000043`; no-persistent minus intact was `-0.046857`. None meets the
  required `+0.01` advantage. M read magnitude and query variation alone
  are not behavioral evidence. The lesion is immediately before the bridge,
  so it tests that retrieval point, not every earlier M→H influence.
- **G25 same-H peripheral causal state — PASS, narrowly.** Swapping F/M at
  identical H changed future-H by mean `2.73917` and prediction JS by
  `0.001316`, in 8/8 seeds. This is a toy-scale causal influence, not proof
  of useful autonomous retrieval or general causal memory. H restoration
  left a tick-8 JS ratio `0.73555`, classified `MIXED` (not a gate).
- **G26 causal utility/persistence link — FAIL.** Mean incremental held-out
  rank R² of CU beyond exposure and registered read-usage magnitude was
  `-0.05492` (95% CI `[-0.11548,0.01377]`), with 2/8 seeds at the `+0.02`
  floor. Corrected development incremental R² was negative in both seeds,
  so consolidation Experiment G is `NOT_RUN_BY_PROTOCOL`; no high-/low-CU
  blocking benefit is inferred. An exploratory hard-access-count adjustment
  also had negative mean incremental R² (`-0.06891`) and cannot change G26.

The normal SELF_OUTPUT arm had zero external writes. The deliberately unsafe
positive control increased memory readout by `0.37006` and a clearly labeled
fixed repeat-output propensity proxy by `0.27780` from tick 1 to 8. This
validates audit sensitivity but is not a trained expression result.

`SMALL_LM_PROTOTYPE_RECOMMENDED=FALSE`; no language model was trained. The
failed gates, invalid attempt, unrun conditional experiment and all
per-seed/null outcomes remain in the Parquet/JSON results and amendments.
