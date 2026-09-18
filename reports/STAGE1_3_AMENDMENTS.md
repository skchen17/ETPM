# Stage 1.3 amendments

No amendments at protocol freeze. Any correction must append the timestamp,
affected run IDs and revisions, whether development or formal outcomes were
visible, and exactly which frozen fields remain unchanged.

## A1 — 2026-09-18, prior-asset byte-hash normalization before training

The first development launcher was rejected by its own integrity check before
any model training or result file was created. The prior-asset SHA list had
been calculated from a Windows checkout whose CRLF working-tree bytes differ
from the canonical LF bytes on the execution server. No historical Git content
had changed. The list is replaced by hashes calculated on the execution server.
Protocol, configs, splits, seeds, metrics, gates, model design and budgets are
unchanged. No development or formal outcome was visible.

## A2 — 2026-09-18, equal-budget development completeness correction

The complete initial development run `stage1_3-development-v1` is preserved.
Its threshold sweep found no feasible threshold for any architecture: the
noise false-emission rate was near one even where recall remained low. No
formal seed had been started. Audit found two implementation omissions relative
to the frozen protocol: training contained no long all-noise/silence sequence,
and the raw-score threshold evaluator treated every post-sufficiency state as a
new positive without applying the required self-output feedback.

The additive run `stage1_3-development-v1a1` gives every architecture the same
800-step development budget and adds a 32-step noise-silence objective. Formal
training, if development becomes feasible, uses 1000 steps for every
architecture. The threshold evaluator now labels the first sufficient world
state and excludes unexpressed post-sufficiency duplicates from recall; formal
evaluation still executes actual threshold feedback. External support scalars
are zero for unrelated events and 0.25 for genuinely weak supporting evidence.
This is evidence strength, not a remember/utility label. LR candidates, seeds,
threshold candidates, selection criteria, model equations and G18–G22 remain
unchanged. The failed initial development artifacts are not overwritten.

## A3 — 2026-09-18, first-sufficient-state threshold semantics

One B6 pilot from the additive development run showed that A2 successfully
eliminated long-noise score drift, but the threshold selector still included
every later no-feedback row whose cumulative count remained four. That
contradicted A2's declared first-sufficient-state semantics and penalized the
model for states that would have received SELF_OUTPUT feedback in deployment.
The evaluator now serializes `newly_sufficient`, and threshold precision/recall
uses that field as its positive stratum while retaining every pre-sufficient
state and all noise rows as negatives. No training run is discarded or
changed. No formal result exists; frozen threshold candidates, gates and
selection constraints are unchanged.

## A4 — 2026-09-18, diagnostic fallback after threshold infeasibility

After A3, B6 development diagnostics were evaluated before any formal run. At
800 training steps, no candidate met the frozen development requirement of
precision at least 0.90 with noise false-emission rate at most 0.01. A single
pre-declared stopping diagnostic extended B6 seed 4301 at learning rate 0.001
to 1600 steps. Noise false emission remained zero, but the best high-recall
point was threshold 0.20 (precision 0.817, recall 0.906), while threshold 0.40
gave precision 0.846 and recall 0.688. No candidate reached precision 0.90.
The exploratory checkpoint is retained and is not eligible for model or
hyperparameter selection.

No additional budget extension is allowed. To execute the requested formal
negative-control study without silently relaxing the scientific criterion, a
diagnostic-only fallback is frozen: among thresholds satisfying the original
noise false-emission ceiling, choose maximum F1, then maximum precision, then
the lower threshold. Each architecture records whether the primary rule was
feasible. If a fallback is used, G19 is forced to FAIL regardless of a chance
formal result. Threshold candidates, primary rule, training budgets, gates and
all formal seeds remain unchanged. No formal outcome was visible.

## A5 — 2026-09-18, complete secondary cross-time ablations

A pre-formal coverage audit found that Experiment C implemented the standard,
M-only and slow-memory-lesion conditions but had not yet executed three
explicitly requested secondary controls: no persistent memory, gamma=0, and a
random memory query. No formal seed had started. The no-memory B0 model is now
included in Experiment C. The B6 gamma-zero and random-query conditions are
same-weight evaluation interventions, replaying exactly the same external
events as the intact B6 condition; they are not retrained capacity controls.
They affect no core gate and are reported only as secondary causal diagnostics.
Training, development selection, thresholds, formal seeds, G18–G22 and long-
stream authorization rules are unchanged.

## A6 — 2026-09-18, amendment-chain verifier correction

While the equal-budget development shell was still running, the A5 evaluator
was installed before its freeze payload. Jobs launched in that short interval
were correctly rejected by the integrity verifier and produced no summary or
checkpoint. The verifier then continued comparing A5's evaluator with the A3
hash, because the supersession list omitted that file. This entry corrects the
amendment chain; all rejected `(model, LR, seed)` cells are rerun under the
same A2 budget. Existing completed cells are retained because A5 changes only
unused secondary evaluation code, not training. No formal outcome was visible,
and no metric, seed, threshold, model equation, budget, or gate changed.
