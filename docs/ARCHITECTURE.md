# ET-RCM Stage-1 and Stage-1.1 architecture

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

## Baseline matching

B0–B6 are implemented. The Toy-2 quantitative panel evaluates B2–B6 because it
has an associative delayed-retention endpoint; B0/B1 are construction-tested
but require a separately trained sequence-task comparison before behavioral
claims. B3 transfers `gamma/key_dim` of every fast direction per internal tick.
This is the isotropic per-direction budget corresponding to one rank-one query
access; using full `gamma` on every direction would spend `key_dim` times the
transfer budget of one ET-RCM query. B2 uses `rho_slow`, an intentionally strong
single-persistent-memory comparison.

## Deliberate omissions

No halting policy, hierarchy, attention stack, language tokenizer, RAG store,
importance label, or textual memory subsystem is present. The module boundary
allows the gated MLP to be replaced by a shared Transformer core after Stage 1.

## Stage-1.1 learned implementation

`src/etrcm/stage1_1/` adds a batched state and current-event-only contract. The
formal learned model uses four 128-wide H slots and 32x32 F/M matrices. A shared
symbol embedding feeds a learned event encoder. The current H produces a unit
query through RMS normalization, mean pooling and a learned projection. The
same gated MLP transition is reused at every event and NULL tick; no tick owns
separate parameters. Symbol and binary heads read only pooled H.

The fixed memory laws are unchanged: only an external event with
`write_mask=true` executes the delta write; query-dependent transfer conserves
F+M before decay. The event dataclass contains only kind/key/value/aux/current
scalars and has no history or future-utility field. Delayed associative tests
reset H before the final query, and formal capacity evaluation includes a zero
F/M lesion.

Trained baselines are B0 recurrent MLP without matrix memory, B1 GRU, B2 two
persistent heads matching the F+M matrix float count, B3 uniform transfer, B5
no-NULL dynamics and B6 full ET-RCM. Exact parameter counts, persistent-state
bytes and compute budgets are stored per record; B0/B1 size mismatches are not
used to adjudicate the matched-capacity gate.

The frozen Stage-1.1 outcome does not justify adding a Transformer or decoder:
learned idle reasoning and interleaved-time behavioral gates failed. Stage 2
remains a plan, not an implemented architecture.

## Stage-1.2 functional intervention layer

`src/etrcm/stage1_2/` preserves the Stage-1.1 memory laws while adding a
one-step forced-query interface. It can replay an identical state under the
learned query, target-parallel/perpendicular components, sign flip,
norm-matched random query, or selective target/non-target projection removal.
External writes remain impossible on NULL steps, and the intervention itself
does not mutate the source state.

The Stage-1.2 TASK_CUE encodes the operation only. After facts are written and
H is scrubbed, autonomous experiments expose neither key IDs nor values. The
sequential graph task likewise scrubs graph history and permits exactly one
query/read per internal tick; audit fields record query count, H scrub, and
absence of labels in events.

Pre/post lesion quantities are explicit:
`pre_lesion_slow_retention`, `post_lesion_slow_retention`,
`pre_lesion_accuracy`, and `post_lesion_accuracy`. This naming is new to Stage
1.2 and does not alter Stage-1.1 records.

The frozen Stage-1.2 result does not authorize a decoder architecture. Slow M
is demonstrably used, and pre-interference consolidation timing helps in one
matched-compute toy, but target-direction functional addressing, selective
behavioral scaling, and length-generalized sequential recurrence did not pass
their gates. Internal time should therefore remain an optional consolidation
scheduling mechanism, not a continuous-cognition core claim.

## Stage-1.3 continuous-state and expression layer

`src/etrcm/stage1_3/` keeps the persistent state exactly `(H,F,M)` and permits
external events at any internal tick. `EVIDENCE`, `CONTEXT`, `NOISE`, and
`SELF_OUTPUT` are typed current events; the schema contains no history, answer,
target, solved, or halt field. Only an external event whose write mask is true
can execute the delta write. Automatic expression feedback embeds the emitted
symbol into H and leaves F/M and both clocks untouched.

The historical read `r=(F+M)q`, exact M-only and F-only reads, and learned
shared-query arbitration are separate registered architectures. Arbitration
uses one scalar `g=sigmoid(G(H,e))` and returns `g Fq + (1-g) Mq`. The same q
drives conserving fast-to-slow transfer, so arbitration changes what reaches H
without changing the conservation law. The optional separate-query R4 was not
implemented before evidence that R3 helps.

Expression uses a scalar sigmoid head and a structured 32-symbol content head.
A development-selected threshold maps a state to `NO_EMIT` or a symbol action.
Emission never resets state or terminates recurrence. The scalar is expression
value, not a calibrated truth probability or a learned halting policy.

The Stage-1.3 formal comparison includes B0/B1 recurrent controls, B2 single
persistent memory, B3 historical joint read, B4 M-only, B5 F-only, B6 scalar
arbitration, and B7 deliberately non-conserving self-replay. Evaluation-only
cross-time interventions remove M, disable consolidation, or replace q with a
target-independent random direction while replaying identical external events.
