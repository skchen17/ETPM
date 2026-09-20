"""Language interface invariants plus frozen-history verification."""

import hashlib
from pathlib import Path

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.language import (LanguageETRCM, WordTokenizer, check_finite, generate,
                                    lesion, shift_targets)


def model():
    torch.manual_seed(7)
    return LanguageETRCM(Stage14Config(hidden_dim=32, latent_slots=1, symbol_count=16,
                                       key_dim=8, value_dim=8))


def test_tokenizer_roundtrip():
    tok = WordTokenizer.fit(["Alice walked. Bob waited!"])
    assert tok.decode(tok.encode("Alice walked. Bob waited!")) == "alice walked. bob waited!"
    assert tok.encode("unseen") == [tok.to_id["<unk>"]]


def test_lm_shape_and_next_token_shift():
    net = model()
    state = net.initial_state(3)
    assert net.logits(state).shape == (3, 16)
    inputs, targets = shift_targets(torch.tensor([[2, 4, 5, 3]]))
    assert inputs.tolist() == [[2, 4, 5]]
    assert targets.tolist() == [[4, 5, 3]]


def test_external_writes_but_null_and_self_do_not():
    net = model(); state = net.initial_state(1); token = torch.tensor([4])
    external, _, d = net.step_token(state, token)
    assert bool(d["external_write_flag"][0])
    assert float(d["external_update"].norm()) > 0
    null, _, d = net.null_tick(external)
    assert not bool(d["external_write_flag"][0])
    assert float(d["external_update"].norm()) == 0
    self_state, _, d = net.step_token(null, token, source="self_output")
    assert not bool(d["external_write_flag"][0])
    assert float(d["external_update"].norm()) == 0
    assert self_state.external_time == null.external_time


def test_generated_token_is_self_output_and_state_not_reset():
    net = model()
    tok = WordTokenizer(["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"] + [f"x{i}" for i in range(10)])
    net.lm_head.weight.data.zero_(); net.lm_head.bias.data.zero_(); net.lm_head.bias.data[4] = 10
    start = net.initial_state(1)
    out, end, trace = generate(net, tok, "hello", max_new_tokens=3, greedy=True, state=start)
    assert out == "hello hello hello"
    assert len(trace) == 3 and all(not r["write"] for r in trace)
    assert all(r["external_time_before"] == r["external_time_after"] for r in trace)
    assert end.tau == 5 and end.external_time == 2  # BOS + prompt external; then 3 self tokens
    assert not torch.equal(end.H, start.H)


def test_lesions_exact_and_H_reset_preserves_memory():
    net = model(); state = net.initial_state(1)
    state, _, _ = net.step_token(state, torch.tensor([5]))
    state, _, _ = net.null_tick(state)
    assert torch.equal(lesion(state, "F", net).F, torch.zeros_like(state.F))
    assert torch.equal(lesion(state, "M", net).M, torch.zeros_like(state.M))
    assert torch.equal(lesion(state, "FM", net).F, torch.zeros_like(state.F))
    assert torch.equal(lesion(state, "FM", net).M, torch.zeros_like(state.M))
    reset = lesion(state, "H", net)
    assert torch.equal(reset.F, state.F) and torch.equal(reset.M, state.M)
    assert torch.equal(reset.H, net.initial_state(1).H)


def test_long_rollout_nonfinite_guard():
    net = model(); state = net.initial_state(1)
    assert check_finite(state)
    for _ in range(30):
        state, _, _ = net.null_tick(state)
    assert check_finite(state)
    corrupted = LearnedState(state.H * torch.tensor(float("nan")), state.F, state.M)
    assert not check_finite(corrupted)


def test_greedy_deterministic():
    net = model()
    tok = WordTokenizer(["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"] + [f"x{i}" for i in range(10)])
    a, _, _ = generate(net, tok, "hello", max_new_tokens=5, greedy=True, seed=11)
    b, _, _ = generate(net, tok, "hello", max_new_tokens=5, greedy=True, seed=99)
    assert a == b


def test_minimum_generation_length_is_explicit():
    net = model()
    tok = WordTokenizer(["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"] + [f"x{i}" for i in range(10)])
    net.lm_head.weight.data.zero_(); net.lm_head.bias.data.zero_()
    net.lm_head.bias.data[tok.to_id["<eos>"]] = 10
    net.lm_head.bias.data[tok.to_id["hello"]] = 5
    text, _, trace = generate(net, tok, "", max_new_tokens=4, min_new_tokens=3, greedy=True)
    assert text == "hello hello hello"
    assert len(trace) == 3 and all(not item["write"] for item in trace)


def test_frozen_stage15_manifest_unchanged():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts/stage1_5_all_assets.sha256"
    assert manifest.exists()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
