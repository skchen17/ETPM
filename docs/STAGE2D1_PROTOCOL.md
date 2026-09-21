# Stage 2D.1 — Success-vs-Failure Training Basin Audit

> **Why do only a minority of identically specified ET-RCM training runs enter the noisy behavioral-memory regime, and is the successful basin caused by slow-memory routing, by earlier conditional-binding dynamics, or by their interaction?**
>
> **为什么在完全相同的 ET-RCM 架构与训练协议下，只有少数训练运行能够进入 noisy behavioral-memory regime？成功 basin 是由 slow-memory routing 因果驱动、由更早的 conditional-binding dynamics 驱动，还是二者共同形成？**

## Frozen scope

Stage 2D.1 changes no external-write, consolidation, decay, H-recurrence, memory-size, evaluator, NULL, or SELF_OUTPUT law. The baseline remains A2 with a protected evaluator, `gamma=.50`, `rho_fast=.97`, `rho_slow=.9995`, `p=.70`, and 1,500 updates. Historical Stage 2C.x and Stage 2D assets are hash-protected and read-only.

## Factorial and trajectories

The confirmatory basin cohort is a complete 8-initialization × 4-data-stream crossing (32 runs). Initialization, data, and evaluation seeds are distinct. Checkpoints are frozen at steps 0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, and 1500. Outcome labels use final action-TV, absolute history×action interaction, and absolute behavioral separation; the M gate is never part of the label.

Every checkpoint receives the same held-out evaluation. It records behavioral interfaces, observed/conditional/marginal CE and CFA, H/F/M/FM centroid probes, distributional routing statistics, query/read geometry, writes, transfers, decay contributions, state increments, and the gradient groups associated with the preceding training update. Cross-seed geometry uses CKA and principal angles rather than raw coordinate comparison.

## Frozen interventions

Parameters and input memory tensors are frozen. The read sweep tests one-factor scales `alpha_F, alpha_M ∈ {0,.5,1,1.5,2}`, four limited joint points, and normalized `g_M ∈ {0,.2,.4,.6,.75,.9,1}`. G68 requires one identical finite intervention to improve action-TV, interaction, and absolute BS in at least 6/8 failed checkpoints without an H/read norm explosion.

Healthy-model necessity uses F, M, and FM read clamps at early/late formation, post-formation, mid/late delay, and probe. These interventions establish read-path mediation only; they do not reopen Stage 2D G66.

## Conditional curricula

Curriculum training is authorized only if the frozen diagnostics support a routing or conditional-binding hypothesis. C1 uses five equal update blocks with evidence reliability `.90 → .80 → .75 → .70 → .65`. C2 tests temporary M-gate floors `{.5,.65,.75}` for `{50,100,200}` updates, then removes the constraint. C3 uses `p=.90` for 300 updates followed by the unmodified target training environment. C4 is permitted only if C1 and C2 show independent development benefit. All final evaluation is fully endogenous and contains no latent, correct-action, memory, oracle, or routing override input.

## Gates and stopping rule

- G68: frozen routing rescue in at least 6/8 failed checkpoints.
- G69: at least 6/8 selected-curriculum models meet action-TV ≥ .10, interaction ≥ .10, and CFA > 0.
- G70: early < mid < late formation in at least 6/8 G69-healthy models.
- G71: meaningful D500 persistence, finite revision, and predictive-over-noise selectivity in at least 6/8 models.
- G72: one preregistered finite F/M/FM formation-window read intervention lowers BS in at least 6/8 models.

Only a replicated G69–G72 result authorizes a new formal fast→slow handoff phase. Null results are retained; thresholds and protocols are not altered after observing outcomes.
