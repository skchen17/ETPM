# Stage 1.2 execution roadmap and completion status

1. **Complete:** froze prior-artifact hashes, protocol, config, splits, gates
   and eight fresh formal seeds.
2. **Complete:** added query interventions, explicit lesion semantics and
   implementation audits.
3. **Complete:** ran equal-budget architecture-specific development selection;
   all architectures independently selected LR `0.001`.
4. **Complete:** ran formal Experiments A–H and preserved raw shards,
   checkpoints, training logs and condition summaries.
5. **Complete:** adjudicated without moving thresholds: G14–G17 all FAIL;
   `ET_STATUS_SUPPORTED` for the separate timing intervention.
6. **Not run by frozen protocol:** Experiment I required the Memory-LM
   authorization conditions, which were not satisfied.
7. **Complete:** both `STAGE2_MEMORY_LM_AUTHORIZED` and
   `STAGE2_CONTINUOUS_COGNITION_LM_AUTHORIZED` are `false`; no decoder training
   was started.

Any follow-up should first repair the Stage-1.2 failures—especially target
addressing specificity, a non-ceiling knowable control, and train-to-length
generalization—under a newly frozen protocol. Stage-1/1.1/1.2 adjudications
must remain immutable.
