# Stage 1.1 amendments

No amendments at protocol freeze. Any implementation-error correction or
resource-driven scope change must be appended here with timestamp, affected
run IDs, old/new hashes, and whether prior outcomes were visible.

## A1 — 2026-09-17, implementation-completeness correction

- Affected aborted run: `stage1_1-formal-v1` at implementation revision
  `8ba28a9`; all partial records, logs, and checkpoints are preserved.
- Trigger: after some baseline outcomes were visible, an audit found that
  Experiment D sampled only training-supported path lengths despite the frozen
  protocol requiring an unseen-longer-path stratum, and Experiment I did not
  serialize the already-required unrelated-memory retention measure.
- Correction: Experiment D now samples lengths 4--8 and explicitly records
  whether length is outside the trained 1--6 support. Experiment I records
  unrelated slow-read retention for every frozen grid cell. Contradictory new
  values are also guaranteed to differ from old values.
- No training rule, hyperparameter, seed, split, gate threshold, baseline,
  exposure grid, or outcome was changed. The corrected complete run ID is
  `stage1_1-formal-v1a1`.
- The correction is additive and disclosed because partial formal outcomes
  had become visible. The aborted run is excluded from gate adjudication but
  retained as an audit artifact.
