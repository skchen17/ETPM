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

## A6 — apply frozen horizon weights inside each family

The same pre-adjudication aggregation review identified that the explicit
`horizon_weights=[0.4,0.3,0.2,0.1]` in the frozen configuration should be
applied to valid horizon CE within a family, then renormalized when only h=1
is available. The G23 analyzer now does this before equal-family and
equal-seed averaging. This implements the registered multi-horizon world
loss; it does not change any model, record, gate threshold, formal seed or
selection. No corrected formal adjudication had been run before the fix.

## A7 — report preregistered seed-bootstrap intervals

The original analysis plan specified 2,000 seed-bootstrap resamples and
confidence intervals for every gate. The draft analyzer had seed means/tables
but omitted interval output. Before corrected formal adjudication, it was
extended to report 95% percentile intervals for G23's three paired margins,
G24's three control margins, G25's H/JS effects, and G26's incremental R²,
resampling the eight independent training seeds with the frozen seed 1414.
This adds uncertainty reporting only; no gate value, threshold, sample,
selection or verdict formula changes.

## A8 — descriptive reactivation diagnostics in processed output

The formal Experiment B records already contain q_M vectors, gate values,
raw/effective M reads and paired H effects. The summarizer was extended to
include their per-gap means and query-coordinate variance in the topic report,
so the required reactivation diagnostics are visible without opening every
Parquet shard. These quantities are explicitly descriptive and do not enter
G24 or any other gate. No raw data or intervention changes.

## A9 — exploratory access-count sensitivity audit

The registered `read_usage` is cumulative query alignment times effective
M contribution, not a pure frequency count. To avoid calling a magnitude
surrogate a count, a separate post-formal **exploratory** replay measures
per-item soft count (sum of absolute query–key cosine) and hard count (cosine
≥0.5) on the already frozen B5 checkpoints and deterministic E/F worlds.
It compares retention associations and incremental held-out R² after
exposure, read magnitude and count controls. The 0.5 cutoff is diagnostic,
not a tuned model input or formal gate criterion. All eight seeds and rows
are saved separately. This audit cannot change G26 or authorize Experiment G.

## A10 — post-adjudication report wording only

The generated causal-usage topic report initially described the registered
composite `read_usage` as if an access count were separately logged there.
The report and its generator now state the exact metric and point to the
separate exploratory frequency audit. Final-report answer 9 was also made
direct: G25 supports a toy-scale causal peripheral-state effect, not a broad
causal-memory claim. No data, numeric result, gate or recommendation changed.

## A11 — matched-compute detail in the report

The predictive-dynamics topic report now prints the already recorded
pre-event versus post-event matched-compute CE at every K, rather than only
their aggregate means; the final report also states A/B episode sample sizes.
This is descriptive reporting from unchanged Parquet rows. It does not enter
G23 or alter any gate.
