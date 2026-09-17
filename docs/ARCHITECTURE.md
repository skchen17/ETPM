# Stage-1 architecture

## State and clocks

`ETState` stores `H`, `F`, `M`, internal time `tau`, and external-event time
`external_time`. Every call to `step` increments `tau`; only a non-null event
increments `external_time`.

`H` has shape `[latent_slots, hidden_dim]`. `F` and `M` have shape
`[value_dim, key_dim]`. Stage 1 is deliberately single-head.

## External transition

For encoded event `(k,v)` with unit `k`:

`F <- F + eta [v - (F+M)k] k^T`; `M` is not directly written.

An all-zero event is `NULL_EVENT` and cannot enter this code path.

## Internal transition

The latent core forms `q = normalize(W_Q mean(H))`, reads `r=(F+M)q`, and uses a
gated MLP residual update for `H`. Consolidation transfers
`Delta=gamma(Fq)q^T` from `F` to `M`. Decay is applied only after the conserving
transfer: `F <- rho_F F`, `M <- rho_M M`.

The order is explicit: external write (if any), latent read/update, conserving
transfer, then decay. Measurements that claim conservation are taken before
decay.

## Deliberate omissions

No halting policy, hierarchy, attention stack, language tokenizer, RAG store,
importance label, or textual memory subsystem is present. The module boundary
allows the gated MLP to be replaced by a shared Transformer core after Stage 1.

