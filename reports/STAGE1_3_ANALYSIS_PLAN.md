# Stage 1.3 formal analysis plan

Frozen before formal seeds are run. Formal outcomes were not visible when this
plan was committed. Development threshold infeasibility is already disclosed
in `STAGE1_3_AMENDMENTS.md`.

## Units and aggregation

- An episode is the behavioral unit; tick rows are first reduced within an
  episode. The eight formal seeds are the replication units.
- G18 correct emission is an episode with at least one correct emission at or
  after sufficient evidence. Insufficient emission is any emission in an
  insufficient-arm episode. A seed passes only if B6 reaches the registered
  absolute rate, B6–B0 margin, and sufficient–insufficient margin.
- G19 precision uses eligible rows consisting of the first sufficient state
  and every not-yet-sufficient state, including the pre-sufficient prefixes of
  episodes that later become sufficient. A true positive is a correct emission
  at the first sufficient state; any other eligible emission is a false positive.
  Recall is true positives divided by first-sufficient states. Noise rate is
  the mean across all streams and ticks of the 10,000-tick B6 run. If B6 had
  no primary-feasible development threshold, G19 fails regardless of formal
  sampling variation.
- G20 uses the registered 2,048-distractor endpoint. Recovery is content-head
  accuracy in target context; useful expression is correct emitted content;
  false expression is any emission in absent context. A seed passes only if
  both B6–B3 positive margins and the false-emission ceiling hold.
- G21 first averages the per-episode Spearman statistic within each seed. The
  overall mean and the number of seeds above the registered per-seed floor are
  adjudicated.
- G22 uses the maximum, over audited post-output ticks, of the across-episode
  mean memory-relative change and expression-score change. All B6 cumulative
  external-write counts must be zero. The B7 non-conserving control must cross
  its registered memory-increase floor.

## Authorization audits

`no_history_bypass` requires the frozen schema/source tests to pass and the
event schema to contain no target, answer, history, solved, or halt field.
`revision_healthy` is a secondary pre-formal audit: at eight new-evidence
events B6 must have mean new-content accuracy at least 0.70, correct revision
emission at least 0.50, and accuracy at least 0.10 above the one-event state.
These criteria do not alter G18–G22.

The long stream is authorized only if every frozen gate and both authorization
audits pass. The language-model prototype additionally requires no self-output
amplification. Failure produces `NOT_RUN_BY_PROTOCOL`; it never triggers a
post-hoc threshold or metric change.
