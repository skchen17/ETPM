# Connection to the read-only J-space reference project

Reference root: `/data/CSK/J-space-project/jstate-closure` (never modified).

## Components inspected

- `README.md`, `reports/FINAL_REPORT.md`, `reports/V11_COMPLETE_REPORT.md`, and
  `reports/V12_COMPLETE_REPORT.md`;
- V11/V12/V13 configs, processed records, and run manifests;
- `src/jclosure/interventions.py`, `geometry.py`, `experiments/jvp_v12.py`,
  `experiments/causal_v12.py`, and `experiments/geometry_v13.py`;
- strict replacement, local Jacobian/JVP, tangent geometry, and variance-vs-
  causal-geometry reports.

## Reused methodology, reimplemented locally

- immutable protocol/config provenance and deterministic seeds;
- append/retain raw outcomes, including failed or null arms;
- machine-readable Parquet plus JSON summaries;
- explicit negative controls and matched-compute controls;
- separate predictive observations from intervention-based causal wording;
- avoid interpreting a probe-restricted low rank as a global intrinsic rank;
- do not authorize a larger model from code existence alone.

No source file, result, model artifact, environment, or symlink is reused. The
new implementation is independent and cites the reference paths above as
methodological provenance only.

## Background observations carried forward as constraints

The reference reports show that the tested instantaneous measured-J state was
not sufficient, predictive compression did not establish writable causal
sufficiency, ordinary variance directions differed from restricted causal
directions, and local restricted spectra remained probe-scoped. ET-RCM thus
starts from an explicit full state and tests its update rules directly instead
of assuming a discovered readout is a complete causal state.

