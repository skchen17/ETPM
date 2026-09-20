# ET-RCM Stage 2A Natural-Language Prototype Results

Exploratory language-domain prototype. All Stage 1.x frozen artifacts are unchanged; no Stage 1 gate is reopened.

## 1. Architecture

`WordTokenizer → external token event → unmodified Stage 1.5 AnatomicalETRCM (H,F,M) → LM head`. External tokens use existing delta writes; NULL and SELF_OUTPUT do not write. H update, independent F/M reads, query-dependent readout-conserving transfer, and decay are inherited without law changes. The event key/value are the same token id: this is a token-associative proxy, not an explicit entity–value memory encoder.

## 2. Model size

| model | parameters |
|---|---|
| gru | 132871 |
| recurrent | 132871 |
| etrcm | 132871 |

All three instantiate the same parameter scaffold; no-memory/GRU modes leave memory-path parameters unused. Therefore nominal parameter equality overstates effective-parameter matching.

## 3. Tokenizer

Lowercase word/punctuation regex, training-only vocabulary size 90; unseen tokens map to `<unk>`. No pretrained tokenizer or corpus.

## 4. Training data

Generated English templates: {'basic': 800, 'memory': 800, 'reasoning': 800} training and {'basic': 60, 'memory': 60, 'reasoning': 60} validation examples. Validation uses a different PRNG seed but shares templates and lexical inventories; results do not establish open-domain generalization. Basic text includes declaratives, descriptions and dialogue; memory includes meeting-day/ownership questions with 0/8/16/32-token distractors; reasoning includes variable binding, transitivity, light transitions, spatial relation. No importance labels. Random train-uniform token baseline CE is `ln(vocab_size)`.

## 5. Training configuration

```json
{
  "seed": 42,
  "device": "cuda:1",
  "steps_per_phase": 60,
  "batch_size": 6,
  "max_length": 56,
  "train_counts": {
    "basic": 800,
    "memory": 800,
    "reasoning": 800
  },
  "validation_counts": {
    "basic": 60,
    "memory": 60,
    "reasoning": 60
  },
  "model_config": {
    "hidden_dim": 64,
    "latent_slots": 1,
    "symbol_count": 90,
    "key_dim": 32,
    "value_dim": 32,
    "event_type_dim": 8,
    "gamma": 0.12,
    "rho_fast": 0.97,
    "rho_slow": 0.9995,
    "eta_external": 0.6,
    "expression_content_count": 32,
    "read_epsilon": 1e-06
  },
  "optimizer": "AdamW",
  "lr": 0.0008,
  "gradient_clip": 1.0,
  "state_clip": false,
  "training_context": "each document resets; no cross-document carry; full BPTT through max_length"
}
```

Teacher-forced next-token prediction; fresh state per document; full BPTT up to max_length. Gradient clipping is parameter-only (norm 1.0); no hidden-state clipping or stabilization. One seed, one small model size; no variance claim.

## 6. Basic language modeling

| model | phase | CE_basic | CE_memory | CE_reasoning | PPL_basic | PPL_memory | PPL_reasoning |
|---|---|---|---|---|---|---|---|
| gru | basic | 1.932 | 4.487 | 4.424 | 6.9 | 88.82 | 83.42 |
| gru | memory | 1.818 | 1.672 | 3.702 | 6.16 | 5.32 | 40.51 |
| gru | reasoning | 1.661 | 1.226 | 2.066 | 5.26 | 3.41 | 7.89 |
| recurrent | basic | 2.739 | 4.651 | 4.681 | 15.47 | 104.74 | 107.92 |
| recurrent | memory | 2.549 | 3.219 | 4.344 | 12.8 | 25.0 | 77.03 |
| recurrent | reasoning | 2.329 | 2.804 | 2.958 | 10.27 | 16.51 | 19.25 |
| etrcm | basic | 2.585 | 4.693 | 4.698 | 13.27 | 109.14 | 109.76 |
| etrcm | memory | 2.364 | 3.194 | 4.375 | 10.64 | 24.38 | 79.41 |
| etrcm | reasoning | 2.12 | 2.739 | 2.794 | 8.33 | 15.48 | 16.35 |

Training-loss, preclip gradient, and final-batch H/F/M norm trajectories are in `train_log.json`; all checkpoint SHA-256 hashes are in `checkpoint_hashes.json`. Uniform-random CE = 4.500; PPL = 90.

## 7. Generated language samples

20 seeded/unconditional and 20 prompt continuations are shown below, including failures. Mean adjacent-token repetition = 0.045; EOS in 31/40; SELF_OUTPUT external writes = 0.

1. **seeded** Prompt: ``
   Output: `the chair is on is alice. a a talked a in a the a a.. question`; repetition=0.067; EOS=True.
2. **seeded** Prompt: `alice walked to the garden .`
   Output: `a blue bob. talked.: the a a a a question a a a a a a a: a the a`; repetition=0.421; EOS=False.
3. **seeded** Prompt: `user : where is alice ? assistant :`
   Output: `: where in kitchen.:`; repetition=0.000; EOS=True.
4. **seeded** Prompt: `the meeting is on`
   Output: `is monday. question the small question chair when the when the the table..`; repetition=0.083; EOS=True.
5. **seeded** Prompt: `bob has the blue key .`
   Output: `the question pressing they.:. they question a: question a: blue a book a question. a blue they a`; repetition=0.000; EOS=False.
6. **seeded** Prompt: `the light is off .`
   Output: `a pressing waited a the table. the button talked a button... the button the the after. a was after`; repetition=0.056; EOS=False.
7. **seeded** Prompt: ``
   Output: `the north is the. the light a is a question question in was the a question a talked in in the the the`; repetition=0.182; EOS=False.
8. **seeded** Prompt: `alice walked to the garden .`
   Output: `the a green found found a blue a a talked question:`; repetition=0.200; EOS=True.
9. **seeded** Prompt: `user : where is alice ? assistant :`
   Output: `the is alice. in.. in..`; repetition=0.250; EOS=True.
10. **seeded** Prompt: `the meeting is on`
   Output: `.. in the chair the..: the when talked talked. when the the table the table was when. when`; repetition=0.059; EOS=False.
11. **seeded** Prompt: `bob has the blue key .`
   Output: `alice the north question a a was. they question question they a question`; repetition=0.167; EOS=True.
12. **seeded** Prompt: `the light is off .`
   Output: `on button was. bob in. the question the talked. they was a question the a the. a. a question`; repetition=0.000; EOS=False.
13. **seeded** Prompt: ``
   Output: `carol owns in the. a a:.. a is talked they a a.: a in`; repetition=0.000; EOS=True.
14. **seeded** Prompt: `alice walked to the garden .`
   Output: `alice`; repetition=0.000; EOS=True.
15. **seeded** Prompt: `user : where is alice ? assistant :`
   Output: `in: is in the`; repetition=0.000; EOS=True.
16. **seeded** Prompt: `the meeting is on`
   Output: `on was. in the question the a monday they table. monday. talked when they table`; repetition=0.000; EOS=True.
17. **seeded** Prompt: `bob has the blue key .`
   Output: `the red. question a the a question they question question. lunch the question question a question they. a. question question`; repetition=0.105; EOS=False.
18. **seeded** Prompt: `the light is off .`
   Output: `pressing light button the a. and table talked in. question question the the question was the question in a the the table`; repetition=0.143; EOS=False.
19. **seeded** Prompt: ``
   Output: `user and carol the kitchen talked in in question the.. talked question a is question the a chair question:: question`; repetition=0.053; EOS=False.
20. **seeded** Prompt: `alice walked to the garden .`
   Output: ``; repetition=0.000; EOS=True.
21. **prompt_continuation** Prompt: `david is north of bob . bob is north of carol . question : who is south of david ? answer :` Target: `carol`
   Output: ``; repetition=0.000; EOS=True.
22. **prompt_continuation** Prompt: `the meeting is on friday .  question : when is the meeting ? answer :` Target: `friday`
   Output: `.`; repetition=0.000; EOS=True.
23. **prompt_continuation** Prompt: `bob has the red key . alice has the green key . carol has the blue key . question : who has the red key ? answer :` Target: `bob`
   Output: `.`; repetition=0.000; EOS=True.
24. **prompt_continuation** Prompt: `the meeting is on tuesday .  question : when is the meeting ? answer :` Target: `tuesday`
   Output: `.`; repetition=0.000; EOS=True.
25. **prompt_continuation** Prompt: `the light is on . pressing button a changes the light . button a is pressed . question : is the light on or off ? answer :` Target: `off`
   Output: `??`; repetition=0.000; EOS=True.
26. **prompt_continuation** Prompt: `the meeting is on friday .  question : when is the meeting ? answer :` Target: `friday`
   Output: `.`; repetition=0.000; EOS=True.
27. **prompt_continuation** Prompt: `alice has the red key . carol has the blue key . bob has the green key . question : who has the green key ? answer :` Target: `bob`
   Output: `..`; repetition=0.000; EOS=True.
28. **prompt_continuation** Prompt: `carol owns the map .  question : who owns the map ? answer :` Target: `carol`
   Output: ``; repetition=0.000; EOS=True.
29. **prompt_continuation** Prompt: `the light is off . pressing button b changes the light . button b is pressed . question : is the light on or off ? answer :` Target: `on`
   Output: `?`; repetition=0.000; EOS=True.
30. **prompt_continuation** Prompt: `the meeting is on tuesday .  question : when is the meeting ? answer :` Target: `tuesday`
   Output: `.`; repetition=0.000; EOS=True.
31. **prompt_continuation** Prompt: `bob has the blue key . alice has the red key . carol has the green key . question : who has the red key ? answer :` Target: `alice`
   Output: `..`; repetition=0.000; EOS=True.
32. **prompt_continuation** Prompt: `the meeting is on thursday .  question : when is the meeting ? answer :` Target: `thursday`
   Output: `.`; repetition=0.000; EOS=True.
33. **prompt_continuation** Prompt: `every zib is a dax . every dax is a fep . question : is every zib a fep ? answer :` Target: `yes`
   Output: ``; repetition=0.000; EOS=True.
34. **prompt_continuation** Prompt: `alice owns the cup .  question : who owns the cup ? answer :` Target: `alice`
   Output: ``; repetition=0.000; EOS=True.
35. **prompt_continuation** Prompt: `every zib is a fep . every fep is a wug . question : is every zib a wug ? answer :` Target: `yes`
   Output: `..`; repetition=0.000; EOS=True.
36. **prompt_continuation** Prompt: `carol owns the map .  question : who owns the map ? answer :` Target: `carol`
   Output: ``; repetition=0.000; EOS=True.
37. **prompt_continuation** Prompt: `the light is off . pressing button a changes the light . button a is pressed . question : is the light on or off ? answer :` Target: `on`
   Output: `???`; repetition=0.000; EOS=True.
38. **prompt_continuation** Prompt: `the meeting is on tuesday .  question : when is the meeting ? answer :` Target: `tuesday`
   Output: `.`; repetition=0.000; EOS=True.
39. **prompt_continuation** Prompt: `alice has the red key . carol has the blue key . bob has the green key . question : who has the red key ? answer :` Target: `alice`
   Output: `..`; repetition=0.000; EOS=True.
40. **prompt_continuation** Prompt: `the meeting is on tuesday .  question : when is the meeting ? answer :` Target: `tuesday`
   Output: `.`; repetition=0.000; EOS=True.

## 8. Natural-language memory tests

Same trained checkpoint, held-out generated episodes, 12 (configurable) episodes/gap. Gaps 32/128/512/1024 are evaluation-only beyond the training gap range. Candidate accuracy is among four allowed values; exact is unrestricted next-token argmax, so candidate accuracy may overstate generation quality.

| gap | condition | n | candidate_acc | exact | answer_CE | read |
|---|---|---|---|---|---|---|
| 32 | full | 12 | 0.417 | 0.0 | 4.986 | 5.755 |
| 32 | F | 12 | 0.417 | 0.0 | 4.954 | 5.777 |
| 32 | M | 12 | 0.417 | 0.0 | 4.959 | 5.764 |
| 32 | FM | 12 | 0.417 | 0.0 | 4.94 | 5.803 |
| 32 | H | 12 | 0.083 | 0.0 | 4.804 | 5.758 |
| 32 | gamma_zero | 12 | 0.417 | 0.0 | 4.984 | 5.62 |
| 32 | gru | 12 | 0.417 | 0.167 | 3.041 | 0.0 |
| 32 | recurrent | 12 | 0.417 | 0.0 | 4.824 | 0.0 |
| 128 | full | 12 | 0.167 | 0.0 | 5.244 | 5.725 |
| 128 | F | 12 | 0.167 | 0.0 | 5.246 | 5.717 |
| 128 | M | 12 | 0.167 | 0.0 | 5.252 | 5.789 |
| 128 | FM | 12 | 0.167 | 0.0 | 5.254 | 5.806 |
| 128 | H | 12 | 0.25 | 0.0 | 4.569 | 5.73 |
| 128 | gamma_zero | 12 | 0.083 | 0.0 | 5.481 | 5.595 |
| 128 | gru | 12 | 0.25 | 0.083 | 3.243 | 0.0 |
| 128 | recurrent | 12 | 0.25 | 0.0 | 5.601 | 0.0 |
| 512 | full | 12 | 0.167 | 0.0 | 5.143 | 5.537 |
| 512 | F | 12 | 0.167 | 0.0 | 5.141 | 5.547 |
| 512 | M | 12 | 0.167 | 0.0 | 5.152 | 5.814 |
| 512 | FM | 12 | 0.167 | 0.0 | 5.149 | 5.821 |
| 512 | H | 12 | 0.333 | 0.0 | 4.678 | 5.795 |
| 512 | gamma_zero | 12 | 0.083 | 0.0 | 5.52 | 5.579 |
| 512 | gru | 12 | 0.25 | 0.0 | 3.12 | 0.0 |
| 512 | recurrent | 12 | 0.083 | 0.0 | 5.457 | 0.0 |
| 1024 | full | 12 | 0.333 | 0.0 | 4.969 | 5.516 |
| 1024 | F | 12 | 0.333 | 0.0 | 4.968 | 5.524 |
| 1024 | M | 12 | 0.333 | 0.0 | 4.971 | 5.815 |
| 1024 | FM | 12 | 0.333 | 0.0 | 4.969 | 5.819 |
| 1024 | H | 12 | 0.333 | 0.0 | 4.713 | 5.772 |
| 1024 | gamma_zero | 12 | 0.25 | 0.0 | 5.362 | 5.583 |
| 1024 | gru | 12 | 0.333 | 0.0 | 3.203 | 0.0 |
| 1024 | recurrent | 12 | 0.25 | 0.0 | 5.456 | 0.0 |

## 9. H/F/M intervention results

F, M, F+M lesions and H reset are applied after the early statement/distractor but before the question, with the same checkpoint. Gamma-zero reruns the prefix with transfer disabled, same weights; it is not a newly trained ablation. For no-memory baselines F/M are unused. Lesions can move logits without proving a uniquely causal slow-memory role.

- 32/full: target `thursday`, predicted `<eos>` (candidate `monday`), CE 5.536; H/F/M 15.64/2.20/0.54.
- 32/F: target `thursday`, predicted `<eos>` (candidate `friday`), CE 5.667; H/F/M 16.20/1.47/0.49.
- 32/M: target `thursday`, predicted `.` (candidate `monday`), CE 5.406; H/F/M 15.18/2.21/0.24.
- 32/FM: target `thursday`, predicted `<eos>` (candidate `monday`), CE 5.571; H/F/M 15.97/1.48/0.22.
- 32/H: target `thursday`, predicted `.` (candidate `tuesday`), CE 5.416; H/F/M 7.03/2.21/0.48.
- 32/gamma_zero: target `thursday`, predicted `.` (candidate `monday`), CE 4.894; H/F/M 15.52/2.27/0.00.

## 10. Natural-language reasoning tests

| condition_task | n | candidate_acc | exact | answer_CE |
|---|---|---|---|---|
| gru/transitivity | 19 | 0.684 | 0.0 | 4.7 |
| recurrent/transitivity | 19 | 0.684 | 0.0 | 5.773 |
| etrcm/transitivity | 19 | 0.737 | 0.0 | 5.199 |
| gru/transition | 19 | 0.474 | 0.0 | 4.685 |
| recurrent/transition | 19 | 0.474 | 0.0 | 4.04 |
| etrcm/transition | 19 | 0.526 | 0.0 | 3.799 |
| gru/binding | 15 | 0.6 | 0.0 | 2.727 |
| recurrent/binding | 15 | 0.6 | 0.0 | 3.658 |
| etrcm/binding | 15 | 0.6 | 0.0 | 3.46 |
| gru/relation | 19 | 0.211 | 0.0 | 3.19 |
| recurrent/relation | 19 | 0.211 | 0.0 | 4.426 |
| etrcm/relation | 19 | 0.211 | 0.0 | 3.912 |

Reasoning examples (same held-out prompt family):

- `carol has the red key . bob has the blue key . alice has the green key . question : who has the red key ? answer :` Target `carol`; free token `.`; candidate `carol`; CE 2.978.
- `carol has the red key . alice has the blue key . bob has the green key . question : who has the green key ? answer :` Target `bob`; free token `.`; candidate `carol`; CE 4.445.
- `david is north of bob . bob is north of carol . question : who is south of david ? answer :` Target `carol`; free token `<eos>`; candidate `carol`; CE 3.498.
- `david is north of alice . alice is north of carol . question : who is south of david ? answer :` Target `carol`; free token `<eos>`; candidate `alice`; CE 3.588.
- `the light is off . pressing button b changes the light . button b is pressed . question : is the light on or off ? answer :` Target `on`; free token `?`; candidate `off`; CE 4.079.
- `the light is off . pressing button b changes the light . button b is pressed . question : is the light on or off ? answer :` Target `on`; free token `?`; candidate `off`; CE 4.079.
- `every zib is a fep . every fep is a dax . question : is every zib a wug ? answer :` Target `no`; free token `.`; candidate `no`; CE 5.137.
- `every wug is a dax . every dax is a fep . question : is every wug a zib ? answer :` Target `no`; free token `.`; candidate `no`; CE 5.115.

## 11. NULL-tick sweep

| task_K | n | candidate_acc | answer_CE | H | F | M |
|---|---|---|---|---|---|---|
| reasoning/0 | 12 | 0.25 | 4.482 | 12.38 | 2.01 | 0.27 |
| reasoning/1 | 12 | 0.333 | 4.552 | 13.11 | 1.95 | 0.28 |
| reasoning/2 | 12 | 0.333 | 4.653 | 14.02 | 1.89 | 0.28 |
| reasoning/4 | 12 | 0.333 | 4.887 | 16.32 | 1.78 | 0.3 |
| reasoning/8 | 12 | 0.333 | 5.261 | 21.83 | 1.57 | 0.32 |
| reasoning/16 | 12 | 0.25 | 5.562 | 32.99 | 1.23 | 0.34 |
| memory/0 | 12 | 0.333 | 5.274 | 17.38 | 2.19 | 0.5 |
| memory/1 | 12 | 0.25 | 5.315 | 18.49 | 2.12 | 0.51 |
| memory/2 | 12 | 0.25 | 5.365 | 19.75 | 2.05 | 0.52 |
| memory/4 | 12 | 0.25 | 5.466 | 22.59 | 1.93 | 0.54 |
| memory/8 | 12 | 0.25 | 5.605 | 28.76 | 1.7 | 0.56 |
| memory/16 | 12 | 0.417 | 5.69 | 40.2 | 1.33 | 0.58 |

NULL ticks follow the query, contain no new token and invoke no external delta write. The trajectory for each episode, including read and H/F/M norms, is in evaluation.json.

## 12. Continuous conversation

| gap | n | candidate_acc | exact | answer_CE |
|---|---|---|---|---|
| 8 | 12 | 0.167 | 0.0 | 5.879 |
| 32 | 12 | 0.0 | 0.0 | 5.028 |
| 128 | 12 | 0.333 | 0.0 | 5.049 |

The state is carried across each dialogue episode without reset; templates and exact transcripts are saved in evaluation.json. Contradiction and duplicate-response rates cannot be meaningfully estimated from a single-token factual answer protocol; not claimed.

### Repeated-question stream diagnostic

A second, strict continuous-state test asks the same meeting-day question twice, with an intervening 8-token distractor and no state reset. Responses are unconstrained 5-token greedy SELF_OUTPUT generations; duplicate/contradiction rates include empty outputs and are interpreted with their coverage.

| gap | n | first correct | second correct | empty first | empty second | duplicate | wrong-day contradiction | max H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 0.000 | 0.000 | 0.667 | 0.917 | 0.583 | 0.000 | 33.49 |
| 32 | 12 | 0.000 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 35.20 |
| 128 | 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 62.56 |

Examples (all raw episodes are saved in `conversation_diagnostic.json`):

- gap=8, target `friday` → first ``, second ``; τ increased=True; SELF_OUTPUT writes=0.
- gap=8, target `thursday` → first `..`, second ``; τ increased=True; SELF_OUTPUT writes=0.
- gap=32, target `thursday` → first `.....`, second ``; τ increased=True; SELF_OUTPUT writes=0.
- gap=32, target `friday` → first `:::.:`, second `...`; τ increased=True; SELF_OUTPUT writes=0.
- gap=128, target `friday` → first `the the the the the`, second `the. the..`; τ increased=True; SELF_OUTPUT writes=0.
- gap=128, target `tuesday` → first `the the the the the`, second `.....`; τ increased=True; SELF_OUTPUT writes=0.

An empty repeated response counts as a duplicate but not as successful recall; low measured contradiction with high nonanswer rate does not imply consistent factual dialogue.

## 13. Continuous-running stability

| requested | completed | first_nonfinite | first_H_gt_1000 | max_H | max_F | max_M | self_writes |
|---|---|---|---|---|---|---|---|
| 100 | 100 | None | None | 85.23 | 1.89 | 0.72 | 0 |
| 500 | 500 | None | None | 275.79 | 1.91 | 1.15 | 0 |
| 1000 | 1000 | None | None | 465.82 | 1.91 | 1.31 | 0 |
| 5000 | 5000 | None | 2551 | 1866.69 | 1.92 | 1.31 | 0 |

Stage 1.5-like strong norm growth in this mixed-stream test: **True**. All per-tick H/F/M norms, H increments, read norms, source and clocks are saved. A finite 1000-tick stream is not proof of arbitrary-duration stability. No state clipping was applied.

## 14. Spontaneous-output prototype

Exploratory, non-gating interface demo. Inherited expression head was NOT trained on language emission targets; threshold 0.45 is a demonstration threshold, not a calibrated policy. **Initial unforced greedy run:** threshold fired in 10/10 trajectories, but every generated string was empty because EOS was selected immediately. Those original traces remain in `evaluation.json` as `spontaneous_initial_greedy`; this is a negative result. The following is a separate decoder-interface diagnostic, not a replacement of that finding. Decoding samples top-12 at temperature 0.9 and suppresses EOS for the first three tokens. The minimum length is forced to demonstrate the token/state interface and is not evidence of learned spontaneous intent. Generated tokens feed SELF_OUTPUT (no external write), and subsequent external events use the same state.

### Trajectory 1 (fired=True; final τ=80; external t=72)

- time=0; external=`carol walked to the station . carol found a green cup .`; NULL=1; score=0.6768; generated=`.. yellow`; subsequent_state={'H': 13.395261764526367, 'F': 1.4708061218261719, 'M': 0.31358280777931213, 'H_delta': 4.090463638305664, 'finite': True}.
- time=1; external=`the blue cup was on the table . later david took it to the library .`; NULL=1; score=0.6098; generated=``; subsequent_state={'H': 30.504676818847656, 'F': 1.9453611373901367, 'M': 0.49808478355407715, 'H_delta': 18.03289031982422, 'finite': True}.
- time=2; external=`user : where is david ? assistant : david is in the garden .`; NULL=1; score=0.5574; generated=``; subsequent_state={'H': 43.65353012084961, 'F': 2.101205825805664, 'M': 0.5884014964103699, 'H_delta': 15.742748260498047, 'finite': True}.
- time=3; external=`the yellow cup was on the table . later david took it to the kitchen .`; NULL=1; score=0.5270; generated=``; subsequent_state={'H': 52.7907829284668, 'F': 2.1197495460510254, 'M': 0.6775177121162415, 'H_delta': 11.09664535522461, 'finite': True}.
- time=4; external=`user : where is david ? assistant : david is in the library .`; NULL=1; score=0.5057; generated=``; subsequent_state={'H': 63.132110595703125, 'F': 2.146613836288452, 'M': 0.7441625595092773, 'H_delta': 12.665413856506348, 'finite': True}.

### Trajectory 2 (fired=True; final τ=74; external t=66)

- time=0; external=`carol walked to the library . carol found a red cup .`; NULL=1; score=0.6365; generated=`a yellow a`; subsequent_state={'H': 12.026972770690918, 'F': 1.4773868322372437, 'M': 0.2756445109844208, 'H_delta': 3.783829689025879, 'finite': True}.
- time=1; external=`alice walked to the kitchen . alice found a yellow key .`; NULL=1; score=0.6256; generated=``; subsequent_state={'H': 29.062143325805664, 'F': 1.868478775024414, 'M': 0.47222036123275757, 'H_delta': 17.652997970581055, 'finite': True}.
- time=2; external=`alice walked to the garden . alice found a red map .`; NULL=1; score=0.6210; generated=``; subsequent_state={'H': 46.15864944458008, 'F': 1.9605149030685425, 'M': 0.6180856823921204, 'H_delta': 17.452796936035156, 'finite': True}.
- time=3; external=`carol and bob talked in the kitchen . they saw a red key .`; NULL=1; score=0.6108; generated=``; subsequent_state={'H': 61.32317352294922, 'F': 2.084731340408325, 'M': 0.6908657550811768, 'H_delta': 15.836793899536133, 'finite': True}.
- time=4; external=`the yellow book was on the table . later alice took it to the kitchen .`; NULL=1; score=0.5926; generated=``; subsequent_state={'H': 73.08104705810547, 'F': 2.1168789863586426, 'M': 0.7421673536300659, 'H_delta': 14.081032752990723, 'finite': True}.

### Trajectory 3 (fired=True; final τ=80; external t=72)

- time=0; external=`bob and bob talked in the station . they saw a yellow map .`; NULL=1; score=0.6428; generated=`.?.`; subsequent_state={'H': 15.124151229858398, 'F': 1.5770740509033203, 'M': 0.23862209916114807, 'H_delta': 3.906374931335449, 'finite': True}.
- time=1; external=`alice and bob talked in the station . they saw a red book .`; NULL=1; score=0.5444; generated=``; subsequent_state={'H': 33.06106948852539, 'F': 1.9710818529129028, 'M': 0.3925102949142456, 'H_delta': 18.396574020385742, 'finite': True}.
- time=2; external=`the blue cup was on the table . later carol took it to the library .`; NULL=1; score=0.5365; generated=``; subsequent_state={'H': 48.52754592895508, 'F': 2.078784942626953, 'M': 0.4982813596725464, 'H_delta': 16.716289520263672, 'finite': True}.
- time=3; external=`alice and bob talked in the library . they saw a blue cup .`; NULL=1; score=0.5301; generated=``; subsequent_state={'H': 63.76705551147461, 'F': 2.1385059356689453, 'M': 0.5896920561790466, 'H_delta': 16.47624397277832, 'finite': True}.
- time=4; external=`user : where is bob ? assistant : bob is in the library .`; NULL=1; score=0.5187; generated=``; subsequent_state={'H': 76.3862075805664, 'F': 2.1816859245300293, 'M': 0.6373094916343689, 'H_delta': 14.551280975341797, 'finite': True}.

### Trajectory 4 (fired=True; final τ=82; external t=74)

- time=0; external=`user : where is carol ? assistant : carol is in the garden .`; NULL=1; score=0.6164; generated=`garden..`; subsequent_state={'H': 12.651594161987305, 'F': 1.5693687200546265, 'M': 0.14526671171188354, 'H_delta': 3.6940834522247314, 'finite': True}.
- time=1; external=`the blue cup was on the table . later carol took it to the library .`; NULL=1; score=0.5796; generated=``; subsequent_state={'H': 23.812665939331055, 'F': 1.985993504524231, 'M': 0.31426262855529785, 'H_delta': 13.184030532836914, 'finite': True}.
- time=2; external=`the red map was on the table . later david took it to the garden .`; NULL=1; score=0.5498; generated=``; subsequent_state={'H': 39.86784362792969, 'F': 2.0886270999908447, 'M': 0.49700993299484253, 'H_delta': 16.483076095581055, 'finite': True}.
- time=3; external=`user : where is bob ? assistant : bob is in the kitchen .`; NULL=1; score=0.5270; generated=``; subsequent_state={'H': 52.749874114990234, 'F': 2.143733501434326, 'M': 0.5860416889190674, 'H_delta': 14.335371017456055, 'finite': True}.
- time=4; external=`user : where is alice ? assistant : alice is in the library .`; NULL=1; score=0.5074; generated=``; subsequent_state={'H': 65.61595916748047, 'F': 2.116703987121582, 'M': 0.6442180871963501, 'H_delta': 13.78317928314209, 'finite': True}.

### Trajectory 5 (fired=True; final τ=76; external t=68)

- time=0; external=`user : where is david ? assistant : david is in the garden .`; NULL=1; score=0.6173; generated=`. green book`; subsequent_state={'H': 12.705809593200684, 'F': 1.5805236101150513, 'M': 0.1466692090034485, 'H_delta': 3.5508315563201904, 'finite': True}.
- time=1; external=`bob and bob talked in the garden . they saw a blue book .`; NULL=1; score=0.5405; generated=``; subsequent_state={'H': 23.491283416748047, 'F': 2.0105814933776855, 'M': 0.22436730563640594, 'H_delta': 13.33356761932373, 'finite': True}.
- time=2; external=`carol walked to the station . carol found a green book .`; NULL=1; score=0.5601; generated=``; subsequent_state={'H': 36.367156982421875, 'F': 2.054949998855591, 'M': 0.4107188880443573, 'H_delta': 14.047576904296875, 'finite': True}.
- time=3; external=`user : where is david ? assistant : david is in the library .`; NULL=1; score=0.5619; generated=``; subsequent_state={'H': 54.671661376953125, 'F': 2.136807918548584, 'M': 0.5276086926460266, 'H_delta': 19.47768211364746, 'finite': True}.
- time=4; external=`alice and bob talked in the garden . they saw a blue key .`; NULL=1; score=0.5522; generated=``; subsequent_state={'H': 68.6968994140625, 'F': 2.180305004119873, 'M': 0.5910124182701111, 'H_delta': 14.9763822555542, 'finite': True}.

### Trajectory 6 (fired=True; final τ=78; external t=70)

- time=0; external=`carol and bob talked in the library . they saw a green map .`; NULL=1; score=0.5800; generated=`yellow it a`; subsequent_state={'H': 12.171049118041992, 'F': 1.5916316509246826, 'M': 0.1534450650215149, 'H_delta': 3.650364398956299, 'finite': True}.
- time=1; external=`bob and bob talked in the library . they saw a red book .`; NULL=1; score=0.4876; generated=``; subsequent_state={'H': 26.14274024963379, 'F': 1.9825242757797241, 'M': 0.27293068170547485, 'H_delta': 14.607605934143066, 'finite': True}.
- time=2; external=`the red cup was on the table . later alice took it to the station .`; NULL=1; score=0.4996; generated=``; subsequent_state={'H': 39.50020217895508, 'F': 2.0727949142456055, 'M': 0.41446638107299805, 'H_delta': 14.74012565612793, 'finite': True}.
- time=3; external=`carol walked to the station . carol found a green map .`; NULL=1; score=0.5207; generated=``; subsequent_state={'H': 53.217960357666016, 'F': 2.068077325820923, 'M': 0.6064962148666382, 'H_delta': 14.363900184631348, 'finite': True}.
- time=4; external=`bob and bob talked in the kitchen . they saw a blue cup .`; NULL=1; score=0.5363; generated=``; subsequent_state={'H': 69.0303955078125, 'F': 2.1265053749084473, 'M': 0.690363347530365, 'H_delta': 16.335268020629883, 'finite': True}.

### Trajectory 7 (fired=True; final τ=83; external t=72)

- time=0; external=`the green cup was on the table . later bob took it to the station .`; NULL=1; score=0.6554; generated=`alice.. bob book.`; subsequent_state={'H': 14.432311058044434, 'F': 1.4674025774002075, 'M': 0.3285179138183594, 'H_delta': 7.023991107940674, 'finite': True}.
- time=1; external=`carol and bob talked in the station . they saw a red key .`; NULL=1; score=0.5617; generated=``; subsequent_state={'H': 30.364206314086914, 'F': 1.9526129961013794, 'M': 0.49084728956222534, 'H_delta': 16.981950759887695, 'finite': True}.
- time=2; external=`the yellow cup was on the table . later alice took it to the kitchen .`; NULL=1; score=0.5425; generated=``; subsequent_state={'H': 45.484397888183594, 'F': 2.0774242877960205, 'M': 0.6180895566940308, 'H_delta': 16.614177703857422, 'finite': True}.
- time=3; external=`david walked to the station . david found a red book .`; NULL=1; score=0.5502; generated=``; subsequent_state={'H': 60.211181640625, 'F': 2.0668601989746094, 'M': 0.7249974608421326, 'H_delta': 15.726224899291992, 'finite': True}.
- time=4; external=`user : where is alice ? assistant : alice is in the kitchen .`; NULL=1; score=0.5457; generated=``; subsequent_state={'H': 75.83958435058594, 'F': 2.132566452026367, 'M': 0.7676764726638794, 'H_delta': 17.063095092773438, 'finite': True}.

### Trajectory 8 (fired=True; final τ=82; external t=74)

- time=0; external=`alice and bob talked in the kitchen . they saw a green cup .`; NULL=1; score=0.5671; generated=`:. book`; subsequent_state={'H': 10.339919090270996, 'F': 1.5890259742736816, 'M': 0.15928082168102264, 'H_delta': 3.2960753440856934, 'finite': True}.
- time=1; external=`carol walked to the library . carol found a red key .`; NULL=1; score=0.5343; generated=``; subsequent_state={'H': 25.77170753479004, 'F': 1.9233570098876953, 'M': 0.33594825863838196, 'H_delta': 16.375720977783203, 'finite': True}.
- time=2; external=`the red cup was on the table . later david took it to the library .`; NULL=1; score=0.5470; generated=``; subsequent_state={'H': 42.87912368774414, 'F': 2.071399688720703, 'M': 0.5155405402183533, 'H_delta': 18.127052307128906, 'finite': True}.
- time=3; external=`the red book was on the table . later carol took it to the library .`; NULL=1; score=0.5291; generated=``; subsequent_state={'H': 56.63624572753906, 'F': 2.103053092956543, 'M': 0.6446774005889893, 'H_delta': 14.89435863494873, 'finite': True}.
- time=4; external=`the green cup was on the table . later carol took it to the library .`; NULL=1; score=0.5133; generated=``; subsequent_state={'H': 68.36573791503906, 'F': 2.1049020290374756, 'M': 0.7451541423797607, 'H_delta': 12.770986557006836, 'finite': True}.

### Trajectory 9 (fired=True; final τ=74; external t=66)

- time=0; external=`the blue map was on the table . later bob took it to the library .`; NULL=1; score=0.6477; generated=`. book.`; subsequent_state={'H': 12.718597412109375, 'F': 1.6260815858840942, 'M': 0.23005346953868866, 'H_delta': 3.8464934825897217, 'finite': True}.
- time=1; external=`alice walked to the garden . alice found a green key .`; NULL=1; score=0.5820; generated=``; subsequent_state={'H': 26.925888061523438, 'F': 1.9173656702041626, 'M': 0.4162765145301819, 'H_delta': 15.112296104431152, 'finite': True}.
- time=2; external=`carol walked to the station . carol found a blue key .`; NULL=1; score=0.6038; generated=``; subsequent_state={'H': 42.13444900512695, 'F': 2.00191068649292, 'M': 0.6061588525772095, 'H_delta': 15.694774627685547, 'finite': True}.
- time=3; external=`user : where is carol ? assistant : carol is in the library .`; NULL=1; score=0.5980; generated=``; subsequent_state={'H': 58.11116027832031, 'F': 2.1107900142669678, 'M': 0.6860854029655457, 'H_delta': 17.649925231933594, 'finite': True}.
- time=4; external=`carol walked to the library . carol found a red book .`; NULL=1; score=0.5962; generated=``; subsequent_state={'H': 72.36076354980469, 'F': 2.0706026554107666, 'M': 0.7689776420593262, 'H_delta': 14.739797592163086, 'finite': True}.

### Trajectory 10 (fired=True; final τ=76; external t=68)

- time=0; external=`carol walked to the garden . carol found a blue key .`; NULL=1; score=0.6489; generated=`...`; subsequent_state={'H': 11.163890838623047, 'F': 1.4823426008224487, 'M': 0.26006728410720825, 'H_delta': 3.929025650024414, 'finite': True}.
- time=1; external=`david walked to the kitchen . david found a yellow book .`; NULL=1; score=0.6197; generated=``; subsequent_state={'H': 28.826444625854492, 'F': 1.8692326545715332, 'M': 0.4644351005554199, 'H_delta': 18.596206665039062, 'finite': True}.
- time=2; external=`user : where is alice ? assistant : alice is in the kitchen .`; NULL=1; score=0.5898; generated=``; subsequent_state={'H': 48.828277587890625, 'F': 2.0628271102905273, 'M': 0.5603041052818298, 'H_delta': 21.214847564697266, 'finite': True}.
- time=3; external=`david and bob talked in the garden . they saw a red key .`; NULL=1; score=0.5682; generated=``; subsequent_state={'H': 63.180538177490234, 'F': 2.136234760284424, 'M': 0.6057891249656677, 'H_delta': 15.327080726623535, 'finite': True}.
- time=4; external=`the green cup was on the table . later bob took it to the kitchen .`; NULL=1; score=0.5517; generated=``; subsequent_state={'H': 73.2547378540039, 'F': 2.143876552581787, 'M': 0.6655822992324829, 'H_delta': 12.662718772888184, 'finite': True}.

## 15. Baseline comparison

GRU and no-memory gated recurrent LM are trained on exactly the same generated corpus and schedule. Validation CE and reasoning/memory candidate accuracy are tabulated above. A baseline doing as well or better means ET-RCM's added memory is not demonstrated necessary for this language regime.

## 16. Failures and negative results

See all failed exact matches and per-episode negative cases in evaluation.json. High candidate accuracy alone is not counted as unconstrained generation. Long-gap extrapolation, H reset, M lesion and NULL deterioration are interpreted without repairing the frozen Stage 1 gates. The expression score is untrained, so spontaneous language intent is unproven.

## 17. Main bottlenecks

Token-level writes use token=key=value rather than compositional entity-value binding; templates and vocabulary are narrow; training documents reset and are at most 56 tokens; long memory is OOD. No free-form multi-token answer accuracy or robust contradiction metric is established. Nominal parameter matching includes inactive branches. Long NULL stability remains separately unresolved.

## 18. Scientific interpretation

This is prototype evidence for a runnable stateful language loop, not LLM capability, consciousness, autonomous thought, general reasoning, infinite context, or causal memory. P1–P5 must be interpreted separately using metrics above, not collapsed into a single success declaration. A changed output under lesion shows intervention sensitivity, not necessarily a useful persistent memory mechanism.

## 19. Recommendation for next stage

Do not scale model size on these single-seed synthetic-template results alone. First add a lexical held-out split, multiple seeds, stronger matched active-parameter baselines, explicit entity-value language encoding, long-context training, answer-level generation evaluation, and a NULL-stability design study. Preserve Stage 1.x frozen adjudications and pre-register any new gates.

## Reproduction and manifest

Run `PYTHONPATH=src .venv/bin/python experiments/stage2a_train.py --output results/stage2a/raw/stage2a_seed42` then `PYTHONPATH=src .venv/bin/python experiments/stage2a_eval.py --root results/stage2a/raw/stage2a_seed42 --report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md`. Exact CLI overrides are in config.json; evaluation seeds are fixed in source. Stage 2A files only are covered by the SHA-256 manifest. Historical frozen file hashes are checked separately.

## 20. Explicit P1–P5 assessment and requested answers

- **P1 language learning: yes on synthetic templates.** Small ET-RCM final basic CE 2.120 vs random 4.500; final memory CE 2.739; reasoning CE 2.794. Validation CE declines over the curriculum but is not monotone across task shifts.
- **P2 generation: not met for reliable natural-language answers.** Inspect the 40 literal outputs above. EOS appeared in 31/40; most question continuations terminate or emit punctuation. Some seeded fragments resemble English, but repetition and degeneration dominate. SELF_OUTPUT is unseen during teacher-forced training.
- **P3 memory: not established.** Long-gap candidate accuracy (128/512/1024 mean) 0.222 vs chance 0.25; M lesion changes answer CE by >0.01 for at least one gap: True. This is sensitivity, not useful M necessity; free-token exact memory accuracy is zero across all ET-RCM gaps.
- **P4 reasoning: not significant at 0.05.** 37/72 candidate answers correct; chance expected 30.3/72; one-sided Poisson-binomial tail p=0.06834 (single seed, same template family). Unrestricted exact-token accuracy is separately in Section 10.
- **P5 continuous operation: technically runs 1000 ticks, but long-horizon stability unresolved.** 1000 mixed ticks completed=True; max H norm 465.82; first nonfinite=None; SELF_OUTPUT external writes=0. At 5000 ticks H exceeds 1000, so this does not establish safe sustained dynamics.

### Small–medium training comparison

Small hidden dimension 64, 132,871 nominal parameters; medium hidden dimension 128, 401,543 nominal parameters (3.02×). Same data seed, phases and 60 optimizer steps/phase. Medium has validation-only evidence here, not the full long-gap battery. Neither model exceeds 1M parameters, as permitted by this preliminary stage.

| size | model | basic CE | memory CE | reasoning CE |
|---|---|---:|---:|---:|
| small | gru | 1.661 | 1.226 | 2.066 |
| small | recurrent | 2.329 | 2.804 | 2.958 |
| small | etrcm | 2.120 | 2.739 | 2.794 |
| medium | gru | 0.958 | 0.699 | 1.186 |
| medium | recurrent | 1.565 | 2.158 | 2.001 |
| medium | etrcm | 1.379 | 2.070 | 1.931 |

### Direct answers to remaining interpretation questions

- GRU comparison: small GRU basic CE 1.661 vs ET-RCM 2.120; medium GRU 0.958 vs ET-RCM 1.379. Memory pathway is not shown superior.
- H/F/M interventions change answer CE/readouts in Section 8; H reset preserves F/M by construction, but post-reset behavioral benefit must be read from the H rows, not inferred from the architecture.
- NULL ticks: reasoning candidate accuracy 0.250 at K=0 and 0.250 at K=16; answer CE 4.482 → 5.562. The memory sweep is in Section 11; neither monotonicity nor an independent compute advantage is assumed.
- Continuous text without reset: True; no external write from SELF_OUTPUT: True in measured generation/continuous events and unit tests. Spontaneous trajectories emitted in 10/10 cases, with state continuing after output; expression score itself is untrained. The demo forces at least three sampled tokens by temporarily suppressing EOS; this is only an interface check.
- Stage 1.5 instability: judge norm growth from Section 13. Finite values alone do not negate the old failure; a hidden norm above 1000 is separately flagged. No clipping or repair was silently applied.
- Dominant current bottlenecks: weak memory binding and out-of-range retention, baseline competitiveness, long-run state growth, and untrained expression. Basic template modeling works; open-domain generation does not.
- Scaling recommendation: further larger sequence-model work is **not yet justified as a mechanism claim**. A controlled follow-up may improve token/entity encoding and stability, then repeat across seeds and lexical OOD splits.
- All positive statements remain prototype evidence on synthetic English. They do not establish broad natural-language understanding, reliable autobiographical memory or endogenous cognition.

### Development pilot retained

A two-step/phase, batch-2, 40-token pilot was run only to verify training and GPU execution. Its checkpoints/logs remain in `results/stage2a/raw/pilot/` and are not used for P1–P5 claims.
