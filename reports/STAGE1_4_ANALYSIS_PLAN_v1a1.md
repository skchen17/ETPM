# Stage 1.4 corrected pre-formal analysis plan

This append-only overlay to `STAGE1_4_ANALYSIS_PLAN.md` is frozen before any
corrected formal seed 7501–7508 is trained. The old 7401–7408 attempt is
invalidated by Amendment A4 and excluded from all tables, regressions and
gates. No original G23–G26 threshold, outcome definition or 6/8 replication
criterion changes.

The corrected latent-transition world exposes the noisy observation, never
hidden z; metadata containing z is available to generator audits only. B and
C are distinct in the long-gap relation world. Corrected development run
`stage1_4-development-v1a1` has all 32 expected trainable cells. Each model
selected LR 0.001 by the original validation rule. The corrected development
causal run `stage1_4-development-causal-v1a2` found incremental held-out R²
`-0.09253` and `-0.07922`, so Experiment G is frozen
`NOT_RUN_BY_PROTOCOL`; G26 cannot pass. The optional causal-utility proxy is
also unauthorized. Formal status cannot change these decisions.

The formal run is `stage1_4-formal-v1a1`, 8 architectures × 8 fresh seeds,
160 steps and batch 32. Episode and seed aggregation, four horizon weights,
K=0/1/2/4/8/16 controls, 128/512/2048 gaps, same-H swaps, restoration,
direction lesion, safety positive control, and all interpretation limits
remain exactly as defined in the original plan. The new formal selection
manifest fixes code/config hashes before training. An omitted/invalid shard
cannot be imputed or replaced with old-run data.
