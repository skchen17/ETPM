# Historical Oracle Retrieval Ceiling — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

For each of 8 seeds: 16 paired long-gap episodes at 128/512/2048 distractors. Early four real events build a saved memory-read bank. The observed C→B bridge selects an earlier B source; the oracle injects only that historical read through the same slow-read interface for four ticks. It never reads future target labels. Zero-M, full no-read, norm-matched random, episode-shuffled and learned-read controls share checkpoint/compute.

## Results

G31: **FAIL**. Registered 512/2048, tick-4 comparator minus static-oracle CE margins:

| Comparator | Mean CE advantage | 95% seed CI | Positive seeds |
| --- | --- | --- | --- |
| learned | -0.000479 | [-0.0014552899330738, 0.0006048623006790866] | 2 |
| random | -0.001106 | [-0.0021706083585740998, 5.155867838766412e-05] | 1 |
| shuffled | -0.000862 | [-0.001730836082424503, 0.0001156054524471976] | 1 |
| zero | -0.000913 | [-0.0018975794111611322, 0.00017121041310019734] | 2 |

Tick-4 CE by gap and condition:

| Gap | Condition | CE |
| --- | --- | --- |
| 128 | learned | 0.861837 |
| 128 | no_read | 0.858407 |
| 128 | oracle_closed_loop | 0.862324 |
| 128 | oracle_static | 0.862081 |
| 128 | random | 0.859477 |
| 128 | shuffled | 0.859392 |
| 128 | zero | 0.858883 |
| 512 | learned | 1.617840 |
| 512 | no_read | 1.617116 |
| 512 | oracle_closed_loop | 1.618526 |
| 512 | oracle_static | 1.618518 |
| 512 | random | 1.616718 |
| 512 | shuffled | 1.617134 |
| 512 | zero | 1.617050 |
| 2048 | learned | 2.002758 |
| 2048 | no_read | 2.002675 |
| 2048 | oracle_closed_loop | 2.003015 |
| 2048 | oracle_static | 2.003039 |
| 2048 | random | 2.002627 |
| 2048 | shuffled | 2.002698 |
| 2048 | zero | 2.002680 |

## Scope and limitations

The oracle has privileged access to a saved pre-interference read. Even a positive ceiling would show integration capacity, not that the learned persistent M actually preserved or found that vector.
