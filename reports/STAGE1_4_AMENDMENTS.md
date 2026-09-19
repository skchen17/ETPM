# Stage 1.4 append-only amendments

## A1 — pre-development Linux asset hash correction

The first Stage 1.4 prior-assets manifest (`7ff9a1c`) embedded SHA256 values
computed from a Windows sparse checkout with CRLF conversion. The canonical
Linux project files are unchanged but have LF bytes, so the original values
cannot validate the server. `artifacts/stage1_4_amendment1.freeze.json` records
the SHA256 values computed on the canonical server and supersedes **only** the
two affected hash fields. The parent Git commit and tree hashes in the original
manifest remain correct. This was identified by the first pre-training test;
there were no development or formal outcomes. No historical artifact, gate,
seed, budget or analysis criterion was changed.

## A2 — pre-formal diagnostic/intervention interface

After all 32 development training shards finished and before formal seeds,
the Stage 1.4 implementation gained an optional `block_transfer_key` argument
for Experiment G and a stricter structural guard that avoids even calling the
external-write function for an event whose validated `write_mask` is all false.
With the default `block_transfer_key=None`, all external/NULL transitions used
in development are mathematically unchanged. The development checkpoints and
their original source revision are retained. A unit test checks the blocked
direction and conservation. The formal source revision and all input hashes
will be frozen before formal training; this amendment changes no world,
training budget, selected LR, endpoint or gate.

## A3 — development causal-evaluation schema repair

The first two development causal-evaluation attempts (`6401`, `6402`) wrote
Parquet rows but stopped before summaries because the record constructor
omitted the `experiment` label required by the summarizer. These incomplete
outputs remain in `stage1_4-development-causal-v1`. Before any causal-usage
selection or formal outcome, the field was added and the same two seeds were
rerun under `stage1_4-development-causal-v1a1`. No model checkpoint, outcome
definition, gate or threshold changed.

## A4 — invalid hidden-state exposure, new independent run

During a pre-adjudication audit, after development and while the first formal
training run was in progress, we found that Family A's event `key_id` exposed
the true hidden latent `z_t` even when the `value_id` observation was noisy.
That violates the frozen partially observed world definition. We also found
that Family B sampled C=B in some episodes, so its bridge write could overwrite
the B→A association. These are world-generator defects, not desired negative
results. The original 32/32 development shards, 2/2 completed causal
development shards, 57/64 formal training shards, and one formal evaluation
shard remain under their original IDs but are **INVALID FOR STAGE 1.4 GATES**.
The formal drivers were stopped; no old output is overwritten or adjudicated.

The corrected generator exposes only the noisy observation in Family A's
visible key/value fields, retains hidden z solely as inaccessible audit
metadata, and enforces C≠B in Family B. The frozen gate thresholds, horizon
weights, architecture laws, equal training budget and analysis definitions
remain unchanged. A new `stage1_4_v1a1.yaml` uses fresh development seeds
6501/6502 and fresh formal seeds 7501–7508, with distinct run IDs. New
selection and formal manifests will be frozen before corrected formal
training. The first run's visible losses cannot influence this selection.

## A5 — formal analysis implementation follows equal-family aggregation

Before corrected formal adjudication, code review found that the draft G23
summarizer averaged all available horizon rows directly. Because two world
families have four valid horizons while two have only one at the registered
prefix, that would give unequal family weight. The already frozen analysis
plan says episode → valid horizons → **equal family** → equal seed. The
summarizer was corrected to implement precisely that rule. The gate threshold,
training/evaluation data, source checkpoints and pre-formal world/selection
hashes were not changed. The earlier draft analyzer was never run on the
corrected complete formal set; no corrected G23 verdict existed before this
change. The first invalid run remains excluded.
