# Stage 1.4 append-only amendments

## A1 — pre-development Linux asset hash correction

The first Stage 1.4 prior-assets manifest (`7ff9a1c`) embedded SHA256 values
computed from a Windows sparse checkout with CRLF conversion. The canonical
Linux project files are unchanged but have LF bytes, so the original values
cannot validate the server. `artifacts/stage1_4_amendment1.freeze.json` records
the SHA256 values computed on the canonical server and supersedes **only** the
two affected hash fields. The parent Git commit and tree hashes in the original
manifest remain correct. This was identified by the first pre-training test;
there were no development or formal outcomes. No historical artifact, gate,
seed, budget or analysis criterion was changed.
