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
