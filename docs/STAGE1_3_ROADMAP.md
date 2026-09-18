# Stage 1.3 execution roadmap

1. Freeze prior hashes, protocol, config, splits, seeds, thresholds and gates.
2. Implement strict external/NULL/SELF_OUTPUT separation and expression without
   halting or reset.
3. Implement R0–R3 reads while preserving external write and conserving
   consolidation.
4. Add all structural, conservation, no-leakage and continuity tests; run the
   complete historical suite before formal work.
5. Run equal-budget LR and threshold development selection; freeze results.
6. Run eight fresh formal seeds for A, D, H, E and I, then B, C, F and G.
7. Run J only if its frozen authorization rule passes.
8. Adjudicate G18–G22 without changing metrics, preserve failures, generate
   reports/hashes, update README/architecture, and decide recommendation only.

Stage 2 remains unauthorized during this roadmap.

## Completion status — 2026-09-18

All roadmap steps through formal adjudication are complete. The development
matrix contained 48/48 unique finite-loss cells; the formal matrix contained
64/64 unique architecture/seed shards and 1,645,312 records. G18–G22 all
failed. B6 passed the narrower no-self-amplification safety subcriteria, but the
B7 pathological-control requirement did not validate, so G22 remains failed.
Revision was not healthy. Experiment J is `NOT_RUN_BY_PROTOCOL`, and Stage 2
remains unauthorized. See `reports/STAGE1_3_FINAL_REPORT.md` and machine-readable
`results/stage1_3/processed/stage1_3-formal-v1/adjudication.json`.
