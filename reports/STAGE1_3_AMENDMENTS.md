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
