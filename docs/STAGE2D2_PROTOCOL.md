# Stage 2D.2 — State-to-Behavior Alignment and Controllability

> **Do successful ET-RCM initializations enter the noisy behavioral-memory basin because history- and memory-induced active-state changes align with the downstream directions that can actually control action-conditioned predictions, and can this alignment be causally manipulated to stabilize training?**
>
> **成功的 ET-RCM 初始化是否因为历史和记忆诱导的 H 状态变化更容易对齐到 downstream evaluator 真正能够利用的行为方向，从而进入 noisy behavioral-memory basin？这种 state-to-behavior alignment 是否可以通过有限因果干预和短暂训练引导被稳定建立？**

## Frozen scope

The Stage 2D architecture, protected evaluator, H/F/M dimensions, external write, consolidation, decay, queries, gate, recurrence, NULL, and SELF_OUTPUT semantics are unchanged. Basin labels remain the Stage 2D.1 action-TV/interaction/BS thresholds and never use an alignment measure.

## Cohorts and checkpoints

Cohort A reuses the complete Stage 2D.1 8×4 factorial. Cohort B contains eight new initializations crossed with three new streams (24 runs). Initialization, stream, and evaluation seeds are disjoint. Steps 0, 25, 50, 100, 200, 300, 500, 750, 1000, and 1500 are analyzed.

## Finite behavioral subspace

At matched current-observation states, 128 normalized contrast directions are applied to H with central finite perturbations at `.05,.10,.25,.50 × native H scale`. The response vector includes both action-conditioned forecasts, action differences, entropy differences, interaction, and BS. A finite least-squares response operator is reconstructed from the direction/response matrix and decomposed by SVD. H-space ranks 1, 2, 4, and 8 are reported; rank 4 at epsilon .10 is the preregistered causal projection.

History alignment projects paired-history H contrast into the finite response subspace. F/M/FM alignment uses the actual difference between a native NULL transition and its corresponding finite read clamp. All ratios aggregate within each run before cross-run statistics.

## Causal interventions

Failed-state rescue decomposes the native M-induced H proposal into useful and orthogonal components. Useful-only, orthogonal-only, random rotation, and lambda `{.25,.50,.75,1}` rotations are row-wise norm matched to the native proposal; ratios above 1.25 are ineligible. F/M tensors and parameters are bit-exact frozen. G73 requires one common intervention to improve interaction and |BS| over native, random, and orthogonal controls in at least 6/8 failures.

Healthy destruction replaces the native proposal by its norm-matched orthogonal component. G74 requires damage in at least 6/8 healthy models beyond the within-useful control.

## Early controllability

Development candidates are external, F, M, and combined finite controllability at steps 0 and 25. The single highest development AUROC freezes feature and sign. G75 requires confirmatory AUROC ≥ .75 and same-direction AUROC ≥ .70 in at least two independent streams.

## Conditional training stage

Alignment warmup, random/orthogonal controls, G76, the reduced dynamics recheck (G77), peripheral recheck (G78), and selected-model continuous diagnostics are authorized only if G73, G74, or G75 passes. This stopping rule is applied before any warmup result exists. If authorized, auxiliary weight must be exactly zero after at most 200/1500 steps; at least 80% of updates and all formal evaluation are fully endogenous.
