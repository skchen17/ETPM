"""Minimal token interface around the unmodified Stage 1.5 transition."""

from __future__ import annotations

import re
from dataclasses import asdict

import torch
from torch import nn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import evidence_event, self_output_event
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_5.model import AnatomicalETRCM


TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)
SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")


class WordTokenizer:
    def __init__(self, vocabulary: list[str]):
        self.vocabulary = list(vocabulary)
        self.to_id = {token: i for i, token in enumerate(vocabulary)}

    @classmethod
    def fit(cls, documents: list[str]) -> "WordTokenizer":
        words = sorted({token.lower() for text in documents for token in TOKEN_PATTERN.findall(text)})
        return cls(list(SPECIAL) + [word for word in words if word not in SPECIAL])

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.to_id.get(token.lower(), self.to_id["<unk>"]) for token in TOKEN_PATTERN.findall(text)]
        return ([self.to_id["<bos>"]] if bos else []) + ids + ([self.to_id["<eos>"]] if eos else [])

    def decode(self, ids: list[int]) -> str:
        words = [self.vocabulary[i] for i in ids if self.vocabulary[i] not in {"<bos>", "<eos>", "<pad>"}]
        text = " ".join(words)
        return re.sub(r"\s+([.,!?;:])", r"\1", text)

    def metadata(self) -> dict:
        return {"type": "lowercase_word_punctuation", "vocabulary": self.vocabulary, "oov": "<unk>"}


class LanguageETRCM(AnatomicalETRCM):
    """A token uses existing event encoding; all memory laws come from Stage 1.5."""

    def __init__(self, config: Stage14Config, mode: str = "B5_separate"):
        super().__init__(config, mode=mode)
        self.lm_head = nn.Linear(config.hidden_dim, config.symbol_count)

    def logits(self, state: LearnedState) -> torch.Tensor:
        return self.lm_head(self._pool(state.H))

    def step_token(self, state: LearnedState, token: torch.Tensor, *, source: str = "external"):
        if source == "external":
            event = evidence_event(token, token)
        elif source == "self_output":
            event = self_output_event(token)
        else:
            raise ValueError(f"unknown token source: {source}")
        new_state, diagnostics = self.step(state, event)
        return new_state, self.logits(new_state), diagnostics

    def null_tick(self, state: LearnedState):
        new_state, diagnostics = self.step(state, None)
        return new_state, self.logits(new_state), diagnostics


def shift_targets(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("tokens must be [batch,length>=2]")
    return tokens[:, :-1], tokens[:, 1:]


def lesion(state: LearnedState, component: str, model: LanguageETRCM) -> LearnedState:
    if component == "full":
        return state.clone()
    if component == "F":
        return LearnedState(state.H.clone(), torch.zeros_like(state.F), state.M.clone(), state.tau, state.external_time)
    if component == "M":
        return LearnedState(state.H.clone(), state.F.clone(), torch.zeros_like(state.M), state.tau, state.external_time)
    if component == "FM":
        return LearnedState(state.H.clone(), torch.zeros_like(state.F), torch.zeros_like(state.M), state.tau, state.external_time)
    if component == "H":
        return model.reset_active(state)
    raise ValueError(component)


def observe(model: LanguageETRCM, state: LearnedState, ids: list[int]):
    diagnostics = None
    for token in ids:
        item = torch.tensor([token], device=state.H.device)
        state, _, diagnostics = model.step_token(state, item, source="external")
    return state, diagnostics


@torch.no_grad()
def generate(model: LanguageETRCM, tokenizer: WordTokenizer, prompt: str, *, max_new_tokens: int = 24,
             temperature: float = 1.0, top_k: int = 0, greedy: bool = False, seed: int | None = None,
             state: LearnedState | None = None, min_new_tokens: int = 0) -> tuple[str, LearnedState, list[dict]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be between zero and max_new_tokens")
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    state = model.initial_state(1, device=device) if state is None else state
    ids = tokenizer.encode(prompt, bos=state.external_time == 0)
    state, _ = observe(model, state, ids)
    trace = []
    generated = []
    for position in range(max_new_tokens):
        logits = model.logits(state)[0] / temperature
        if position < min_new_tokens:
            logits[tokenizer.to_id["<eos>"]] = -torch.inf
        if top_k > 0:
            values, indices = logits.topk(min(top_k, logits.numel()))
            scores = torch.full_like(logits, -torch.inf)
            scores[indices] = values
            logits = scores
        token = int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1), 1))
        if token == tokenizer.to_id["<eos>"]:
            break
        generated.append(token)
        before = state
        state, _, diag = model.step_token(state, torch.tensor([token], device=device), source="self_output")
        trace.append({"token": tokenizer.vocabulary[token], "write": bool(diag["external_write_flag"][0]),
                      "external_time_before": before.external_time, "external_time_after": state.external_time,
                      "tau_after": state.tau})
    return tokenizer.decode(generated), state, trace


def config_record(config: Stage14Config) -> dict:
    return asdict(config)


def check_finite(state: LearnedState) -> bool:
    """Explicit rollout guard; callers report the first failing tick, never clip."""
    return bool(all(torch.isfinite(component).all() for component in (state.H, state.F, state.M)))
