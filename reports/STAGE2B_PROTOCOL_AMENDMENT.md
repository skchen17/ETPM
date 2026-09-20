# Stage 2B protocol amendment before formal completion

The initial development runs (seeds 1101/1102, 100 steps/arm) used Stage 2A
auxiliary basic/reasoning templates. Review found that those templates could
co-occur a name and color/place in combinations reserved for lexical OOD.
Two just-started formal GRU jobs (seeds 2401/2402, less than 1000 steps and no
checkpoint) were interrupted and are excluded from all formal analyses.

Before restarting any formal arm, the auxiliary basic generator was replaced
with sentences that never pair a name and color/place, and the auxiliary
reasoning stream excluded name-color binding examples. All 5 formal seeds and
all 5 arms now use this revised, common data generator. The development runs
remain as feasibility checks only; their results cannot be used for OOD or
architecture adjudication. The formal seed IDs were retained because no
completed formal checkpoint or outcome was inspected. Exact checks are in
`tests/test_stage2b.py`.

This amendment changes data hygiene only. Architecture, optimizer, 1000-step
minimum, batch size, formal seeds, evaluation gaps and gate definitions are
unchanged.

Before contextual formal outcomes were inspected, the primary lexical-OOD
aggregate was restricted to `attribute`, `location`, `revision` and
`interference`: their answer-bearing name+value pairs are train/OOD-disjoint.
`temporal` remains reported by cell but is excluded from the strict OOD
headline, because an old meeting-day mention in training can contain a
name+day combination later queried as the updated answer. This is an analysis
definition correction, not a training-data change; all original raw rows are
retained. The `relation` family is also retained by cell but excluded from the
conservative primary OOD aggregate.

Before contextual-arm formal results were inspected, an optional `E0_late`
diagnostic was registered: raw-token KV with the E1/E2 read-before-write
ordering. It cannot replace frozen E0 in G38–G41, but can expose whether any
apparent contextual gain is actually an ordering effect. Its results are
reported separately and cannot retroactively change the original gates.
For the final Outcome A interpretation (not G38–G41), five available E0-late
seeds must be beaten by a contextual arm in at least 4/5 seeds with mean ID
candidate gain ≥0.03. If that attribution check is unavailable or fails,
Outcome A is disallowed even if the original gates pass.

The inherited GRU/RNN modes allocate parameters in unused memory/core
branches, so nominal parameter equality exaggerates active matching. A
separately labeled GRU-76 diagnostic is registered to approach contextual-arm
active parameter count. It uses the same five seeds, data and 1000-step
schedule; it does not alter the mandatory GRU-64 baseline or gates.
