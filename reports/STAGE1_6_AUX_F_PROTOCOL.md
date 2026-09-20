# Conditional auxiliary F protocol (triggered after frozen gates)

The frozen formal analysis gave G35 PASS while the ordinary learned-read arm had finite D_R≥.01 in only 2/8 seeds, so the user-specified condition for auxiliary Experiment F is met. This add-on does **not** revise G34–G37. It is labeled post-trigger exploratory, with schedule and comparisons written before its training outcomes.

Use the identical B5-separate architecture, gated residual integration, external delta write, readout-conserving F→M transfer, decay, AdamW LR .001, batch 32, gradient clip 1.0, weight decay .0001, same 3000 paired training worlds, model initialization and 256 held-out worlds for seeds 8701–8708. No query vector, key/path or importance label. The only change from formal B3 is the oracle/learned delivery schedule at the bridge:

| Steps | Delivery |
| --- | --- |
| 0–999 | 100% historical oracle read |
| 1000–1999 | 50% oracle, 50% learned, seeded Bernoulli |
| 2000–2999 | 100% learned read |

This has the same expected total oracle-delivery count (1500 of 3000) as formal B3's five-block schedule. At checkpoints 0, 1000, 2000, 3000, evaluate learned/oracle/zero/random/shuffled read and scrub-boundary M/F lesions on paired held-out episodes. Primary descriptive comparison: seed-paired `L_zero-L_learned` at 3000 versus formal B3 curriculum, plus the number of seeds with ≥.025 CE benefit. Oracle-reintroduction benefit and retained fraction are reported separately, without modifying G37. If extra phases fail, preserve them. This experiment cannot authorize Stage 2 or reverse a frozen gate.
