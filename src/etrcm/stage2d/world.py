"""Noisy latent world with matched action/outcome marginals.

The hidden support bit is simulator metadata only.  The model receives the
same five observed fields as Stage 2C: surface, nuisance, action and outcome.
One observation identifies the support bit but only supplies Bernoulli(p)
evidence about z, so it cannot identify z.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from etrcm.stage2c.world import ACTION, OUTCOME, Experience, surface


@dataclass(frozen=True)
class NoisyExperience:
    observed: Experience
    support_bit: int
    latent: int
    predictive: bool
    opposed: bool


def infer_support(item: Experience) -> int:
    """Recover the noisy evidence bit from legal observed action/outcome."""
    return item.action if item.outcome == OUTCOME[0] else 1 - item.action


def bayes_p_z0(support_bits: list[int], p_evidence: float, prior: float = 0.5) -> float:
    if not 0.5 < p_evidence < 1.0:
        raise ValueError("p_evidence must lie in (0.5,1)")
    log_odds = math.log(prior / (1.0 - prior))
    weight = math.log(p_evidence / (1.0 - p_evidence))
    log_odds += sum(weight if bit == 0 else -weight for bit in support_bits)
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return e / (1.0 + e)


def _wrong_outcome(seed: int, pair: int, index: int) -> int:
    return OUTCOME[1 + random.Random(seed * 1000003 + pair * 9176 + index * 37 + 11).randrange(3)]


def paired_experiences(
    seed: int,
    replicates: int,
    index: int,
    p_evidence: float,
    *,
    split: str = "train",
    matched_noise: bool = False,
    oppose_fraction: float = 0.0,
) -> list[NoisyExperience]:
    """Return [all z=0 replicas, all z=1 replicas] with paired observables.

    Surfaces, actions and uniforms are paired across latent conditions.  For
    predictive evidence, the same uniform yields complementary support bits.
    For matched noise, both latent conditions get the identical support bit,
    making its mutual information with z exactly zero in the paired sample.
    """
    if replicates < 1 or not 0.5 < p_evidence < 1.0:
        raise ValueError((replicates, p_evidence))
    if not 0.0 <= oppose_fraction <= 1.0:
        raise ValueError(oppose_fraction)
    rows: list[NoisyExperience] = []
    cached = []
    for pair in range(replicates):
        rng = random.Random(seed * 10000019 + pair * 1009 + index * 7919)
        features = surface(rng, split)
        action = (index + pair + seed) % 2
        evidence_u = rng.random()
        oppose = rng.random() < oppose_fraction
        noise_support = int(rng.random() >= 0.5)
        cached.append((features, action, evidence_u, oppose, noise_support))
    for latent in (0, 1):
        for pair, (features, action, evidence_u, oppose, noise_support) in enumerate(cached):
            source_latent = 1 - latent if oppose else latent
            if matched_noise:
                support = noise_support
            else:
                support = source_latent if evidence_u < p_evidence else 1 - source_latent
            outcome = OUTCOME[0] if action == support else _wrong_outcome(seed, pair, index)
            c, s, nuisance = features
            observed = Experience(c, s, nuisance, action, outcome)
            rows.append(NoisyExperience(observed, support, latent, not matched_noise, oppose))
    return rows


def training_experiences(seed: int, batch: int, index: int, p_evidence: float) -> list[NoisyExperience]:
    if batch % 2:
        raise ValueError("training batch must be even")
    return paired_experiences(seed, batch // 2, index, p_evidence, split="train")


def observed_only(rows: list[NoisyExperience]) -> list[Experience]:
    """The sole adapter used at the lifetime-model boundary."""
    return [row.observed for row in rows]


def empirical_support_rate(rows: list[NoisyExperience]) -> float:
    return sum(row.support_bit == row.latent for row in rows) / len(rows)


def outcome_marginal(rows: list[NoisyExperience]) -> list[float]:
    counts = [0, 0, 0, 0]
    for row in rows:
        counts[row.observed.outcome - OUTCOME[0]] += 1
    return [value / len(rows) for value in counts]
