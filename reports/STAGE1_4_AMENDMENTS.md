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
