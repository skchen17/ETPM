# Stage 1.3 amendments

No amendments at protocol freeze. Any correction must append the timestamp,
affected run IDs and revisions, whether development or formal outcomes were
visible, and exactly which frozen fields remain unchanged.

## A1 — 2026-09-18, prior-asset byte-hash normalization before training

The first development launcher was rejected by its own integrity check before
any model training or result file was created. The prior-asset SHA list had
been calculated from a Windows checkout whose CRLF working-tree bytes differ
from the canonical LF bytes on the execution server. No historical Git content
had changed. The list is replaced by hashes calculated on the execution server.
Protocol, configs, splits, seeds, metrics, gates, model design and budgets are
unchanged. No development or formal outcome was visible.
