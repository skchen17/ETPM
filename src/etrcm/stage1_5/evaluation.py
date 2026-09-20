"""Paired finite Stage 1.5 architecture and routing evaluations."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.interventions import prediction_js
from etrcm.stage1_4.training import prefix_index, run_prefix
from etrcm.stage1_4.world import FAMILIES, generate_world
from .interventions import CHANNELS, swap_components
from .model import AnatomicalETRCM
from .routing import build_historical_bank, closed_loop_oracle_read, static_oracle_read


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return Fnn.cross_entropy(logits, target, reduction="none")


def _norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.flatten(1).norm(dim=-1)


def _row(model: AnatomicalETRCM, run_id: str, seed: int, experiment: str,
         family: str, episode: int, state: LearnedState, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id, "model": model.mode, "seed": seed,
        "experiment": experiment, "episode": episode, "world_family": family,
        "state_id": f"{seed}:{family}:{episode}",
        "external_step": state.external_time, "internal_tick": state.tau - state.external_time,
        "H_norm": float(state.H[episode].norm()), "F_norm": float(state.F[episode].norm()),
        "M_norm": float(state.M[episode].norm()),
        "parameter_count": model.trainable_parameters(),
        "state_bytes": model.persistent_state_bytes(),
        "precision_mode": "fp32", "target_query_input": False,
    }
    row.update(extra)
    return row


@torch.no_grad()
def evaluate_stability(model: AnatomicalETRCM, *, seed: int, run_id: str,
                       device: torch.device, batch: int = 8,
                       ticks: int = 1024) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_id, family in enumerate(FAMILIES):
        world = generate_world(family, batch=batch, length=16,
                               seed=seed * 100_000 + 15001 + family_id, device=device)
        prefix = prefix_index(world, 5)
        state, diagnostics = run_prefix(model, world, prefix=prefix, null_ticks=0)
        initial_logits = model.predict_logits(state)[1]
        target = world.future(prefix, 1)
        output = diagnostics[-1]
        for tick in range(ticks + 1):
            if tick:
                previous = state
                state, output = model.step(state, None)
                h_delta = _norm(state.H - previous.H)
                f_delta = _norm(state.F - previous.F)
                m_delta = _norm(state.M - previous.M)
            else:
                h_delta = torch.zeros(batch, device=device)
                f_delta = torch.zeros(batch, device=device)
                m_delta = torch.zeros(batch, device=device)
            logits = model.predict_logits(state)[1]
            probs = logits.softmax(-1).clamp_min(1e-9)
            entropy = -(probs * probs.log()).sum(-1)
            js = prediction_js(initial_logits, logits)
            loss = _loss(logits, target)
            q_f = output["q_F"].norm(dim=-1)
            q_m = output["q_M"].norm(dim=-1)
            transfer = _norm(output["transfer"])
            for ep in range(batch):
                rows.append(_row(
                    model, run_id, seed, "A1_stability", family, ep, state,
                    external_step=prefix + 1, internal_tick=tick,
                    H_delta_norm=float(h_delta[ep]), F_delta_norm=float(f_delta[ep]),
                    M_delta_norm=float(m_delta[ep]), prediction_entropy=float(entropy[ep]),
                    prediction_js_from_t0=float(js[ep]), future_CE=float(loss[ep]),
                    q_F_norm=float(q_f[ep]), q_M_norm=float(q_m[ep]),
                    r_F_norm=float(output["r_F_norm"][ep]),
                    r_M_norm=float(output["r_M_norm"][ep]),
                    transfer_norm=float(transfer[ep]),
                    g_F=float(output["gates"][ep, 0]),
                    g_M=float(output["gates"][ep, 1]),
                    external_write=bool(output["external_write_flag"][ep]) if tick else None,
                    compute_budget=prefix + 1 + tick,
                ))
    return rows


@torch.no_grad()
def evaluate_perturbations(model: AnatomicalETRCM, *, seed: int, run_id: str,
                           device: torch.device, batch: int = 8,
                           checkpoints: Iterable[int] = (0, 1, 2, 4, 8, 16, 32, 64, 128)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(seed * 10_000 + 1515)
    for family_id, family in enumerate(FAMILIES):
        world = generate_world(family, batch=batch, length=16,
                               seed=seed * 100_000 + 16001 + family_id, device=device)
        prefix = prefix_index(world, 5)
        base, _ = run_prefix(model, world, prefix=prefix, null_ticks=0)
        checkpoints = tuple(checkpoints)
        base_trajectory = {0: base}
        control = base
        for tick in range(1, max(checkpoints) + 1):
            control, _ = model.step(control, None)
            if tick in checkpoints:
                base_trajectory[tick] = control
        for component in "HFM":
            for epsilon in (0.001, 0.01):
                tensors = {name: getattr(base, name).clone() for name in "HFM"}
                direction = torch.randn(tensors[component].shape, generator=generator, device=device)
                direction = direction / _norm(direction).view(batch, *([1] * (direction.ndim - 1)))
                tensors[component] += epsilon * direction
                state = LearnedState(tensors["H"], tensors["F"], tensors["M"], base.tau, base.external_time)
                initial_full = epsilon
                for tick in range(max(checkpoints) + 1):
                    if tick:
                        state, _ = model.step(state, None)
                    if tick not in checkpoints:
                        continue
                    comparator = base_trajectory[tick]
                    h_diff = _norm(state.H - comparator.H)
                    f_diff = _norm(state.F - comparator.F)
                    m_diff = _norm(state.M - comparator.M)
                    full_diff = (h_diff.square() + f_diff.square() + m_diff.square()).sqrt()
                    js = prediction_js(model.predict_logits(state)[1], model.predict_logits(comparator)[1])
                    for ep in range(batch):
                        rows.append(_row(
                            model, run_id, seed, "A2_perturbation", family, ep, state,
                            external_step=prefix + 1, internal_tick=tick,
                            perturbation_component=component, epsilon=epsilon,
                            H_difference=float(h_diff[ep]), F_difference=float(f_diff[ep]),
                            M_difference=float(m_diff[ep]), full_state_difference=float(full_diff[ep]),
                            H_response_gain=float(h_diff[ep] / initial_full),
                            full_state_gain=float(full_diff[ep] / initial_full),
                            prediction_js=float(js[ep]),
                            compute_budget=prefix + 1 + tick,
                        ))
    return rows


@torch.no_grad()
def evaluate_anatomy(model: AnatomicalETRCM, *, seed: int, run_id: str,
                     device: torch.device, batch: int = 32) -> list[dict[str, object]]:
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 17001, device=device)
    base, _ = run_prefix(model, world, prefix=world.bridge_index, null_ticks=0)
    donor = LearnedState(base.H.roll(1, 0), base.F.roll(1, 0), base.M.roll(1, 0),
                         base.tau, base.external_time)
    tracks: dict[str, LearnedState] = {"intact": base.clone()}
    tracks.update({channel: swap_components(base, donor, channel) for channel in CHANNELS})
    target = world.future(world.bridge_index, 1)
    rows: list[dict[str, object]] = []
    for tick in range(1, 9):
        tracks = {name: model.step(state, None)[0] for name, state in tracks.items()}
        if tick not in (1, 2, 4, 8):
            continue
        intact = tracks["intact"]
        intact_logits = model.predict_logits(intact)[1]
        intact_ce = _loss(intact_logits, target)
        effects = {}
        for condition in CHANNELS:
            state = tracks[condition]
            logits = model.predict_logits(state)[1]
            ce = _loss(logits, target)
            js = prediction_js(intact_logits, logits)
            effects[condition] = ce - intact_ce
            h_diff = _norm(state.H - intact.H)
            for ep in range(batch):
                rows.append(_row(
                    model, run_id, seed, "A5_anatomy", world.family, ep, state,
                    external_step=world.bridge_index + 1, internal_tick=tick,
                    swap_condition=condition, donor_episode=(ep - 1) % batch,
                    exact_swapped_components=condition, future_H_difference=float(h_diff[ep]),
                    prediction_js=float(js[ep]), future_CE=float(ce[ep]),
                    intact_CE=float(intact_ce[ep]), signed_CE_effect=float(effects[condition][ep]),
                    behavioral_accuracy=float(logits[ep].argmax() == target[ep]),
                    compute_budget=world.bridge_index + 1 + tick,
                ))
        interactions = {
            "FM": effects["FM"] - effects["F"] - effects["M"],
            "HF": effects["HF"] - effects["H"] - effects["F"],
            "HM": effects["HM"] - effects["H"] - effects["M"],
            "HFM": effects["HFM"] - effects["HF"] - effects["HM"] - effects["FM"]
                   + effects["H"] + effects["F"] + effects["M"],
        }
        for label, values in interactions.items():
            for ep in range(batch):
                rows.append(_row(
                    model, run_id, seed, "A5_interaction", world.family, ep, intact,
                    external_step=world.bridge_index + 1, internal_tick=tick,
                    interaction=label, signed_CE_interaction=float(values[ep]),
                    compute_budget=world.bridge_index + 1 + tick,
                ))
    return rows


def _effective_read_components(model: AnatomicalETRCM, state: LearnedState) -> tuple[torch.Tensor, torch.Tensor]:
    batch = state.H.shape[0]
    encoded = torch.zeros(batch, model.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
    q_f, q_m = model._queries(state.H)
    reads = model._read_stage14(state.F, state.M, q_f, q_m, state.H, encoded)
    return (reads["gates"][:, :1] * reads["normalized_r_F"],
            reads["gates"][:, 1:] * reads["normalized_r_M"])


@torch.no_grad()
def evaluate_read_mediation(model: AnatomicalETRCM, *, seed: int, run_id: str,
                            device: torch.device, batch: int = 32) -> list[dict[str, object]]:
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 18001, device=device)
    base, _ = run_prefix(model, world, prefix=world.bridge_index, null_ticks=0)
    donor = LearnedState(base.H.roll(1, 0), base.F.roll(1, 0), base.M.roll(1, 0),
                         base.tau, base.external_time)
    target = world.future(world.bridge_index, 1)
    rows: list[dict[str, object]] = []
    for channel in ("F", "M"):
        control, swapped = base.clone(), swap_components(base, donor, channel)
        restored = swapped.clone()
        for tick in range(1, 9):
            control_f, control_m = _effective_read_components(model, control)
            restored_f, restored_m = _effective_read_components(model, restored)
            fused = control_f + restored_m if channel == "F" else restored_f + control_m
            control, _ = model.step(control, None)
            swapped, _ = model.step(swapped, None)
            restored, restored_diag = model.step_with_read(restored, None, fused_read_override=fused)
            if tick not in (1, 2, 4, 8):
                continue
            control_logits = model.predict_logits(control)[1]
            control_ce = _loss(control_logits, target)
            swapped_logits = model.predict_logits(swapped)[1]
            swapped_js = prediction_js(control_logits, swapped_logits)
            restored_logits = model.predict_logits(restored)[1]
            restored_js = prediction_js(control_logits, restored_logits)
            swapped_ce = _loss(swapped_logits, target)
            restored_ce = _loss(restored_logits, target)
            for ep in range(batch):
                for condition, state, ce, js in (
                    ("swap", swapped, swapped_ce, swapped_js),
                    ("swap_read_restored", restored, restored_ce, restored_js),
                ):
                    rows.append(_row(
                        model, run_id, seed, "B2_read_mediation", world.family, ep, state,
                        external_step=world.bridge_index + 1, internal_tick=tick,
                        swap_condition=channel, read_restore_flag=condition == "swap_read_restored",
                        intervention_condition=condition, future_CE=float(ce[ep]),
                        intact_CE=float(control_ce[ep]), prediction_js=float(js[ep]),
                        stored_F_changed=channel == "F", stored_M_changed=channel == "M",
                        fusion_override_norm=float(fused[ep].norm()) if condition == "swap_read_restored" else None,
                        transfer_norm=float(restored_diag["transfer"][ep].norm()) if condition == "swap_read_restored" else None,
                        compute_budget=world.bridge_index + 1 + tick,
                    ))
    return rows


@torch.no_grad()
def evaluate_oracle(model: AnatomicalETRCM, *, seed: int, run_id: str,
                    device: torch.device, gaps: Iterable[int] = (128, 512, 2048),
                    batch: int = 16) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(seed * 100_000 + 19001)
    for gap in gaps:
        world = generate_world("long_gap_relation", batch=batch, length=16,
                               seed=seed * 100_000 + 19001 + gap, device=device, gap=gap)
        early, bank = build_historical_bank(model, world.events, history_length=4)
        static_read, source = static_oracle_read(bank, world.events[world.bridge_index])
        state = early
        for index in range(4, world.bridge_index):
            state, _ = model.step(state, world.events[index])
        random_read = torch.randn(static_read.shape, generator=generator, device=device)
        random_read = random_read / random_read.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        random_read = random_read * static_read.norm(dim=-1, keepdim=True)
        shuffled_read = static_read.roll(1, dims=0)
        conditions = ("learned", "zero", "no_read", "random", "shuffled", "oracle_static", "oracle_closed_loop")
        tracks = {name: state.clone() for name in conditions}
        target = world.future(world.bridge_index, 1)
        for tick in range(1, 9):
            for condition in conditions:
                old = tracks[condition]
                event = world.events[world.bridge_index] if tick == 1 else None
                if condition == "learned":
                    new, output = model.step(old, event)
                    source_index = None
                elif condition == "no_read":
                    new, output = model.step_with_read(old, event, fused_read_override=torch.zeros(batch, model.config.value_dim, device=device))
                    source_index = None
                else:
                    read = {
                        "zero": torch.zeros_like(static_read), "random": random_read,
                        "shuffled": shuffled_read, "oracle_static": static_read,
                    }.get(condition)
                    if condition == "oracle_closed_loop":
                        read, source_index = closed_loop_oracle_read(
                            model, old.H, bank, world.events[world.bridge_index]
                        )
                    elif condition == "oracle_static":
                        source_index = source
                    else:
                        source_index = None
                    assert read is not None
                    new, output = model.step_with_read(old, event, slow_read_override=read)
                tracks[condition] = new
                if tick not in (1, 2, 4, 8):
                    continue
                logits = model.predict_logits(new)[1]
                ce = _loss(logits, target)
                for ep in range(batch):
                    rows.append(_row(
                        model, run_id, seed, "B3_B4_oracle", world.family, ep, new,
                        external_step=world.bridge_index + 1, internal_tick=tick,
                        distractor_count=gap, intervention_condition=condition,
                        oracle_read_flag=condition.startswith("oracle"),
                        oracle_source_item=int(source_index[ep]) if source_index is not None else None,
                        oracle_source_is_history=source_index is not None,
                        future_CE=float(ce[ep]),
                        behavioral_accuracy=float(logits[ep].argmax() == target[ep]),
                        q_F=output["q_F"][ep].tolist(), q_M=output["q_M"][ep].tolist(),
                        r_F_norm=float(output["r_F_norm"][ep]), r_M_norm=float(output["r_M_norm"][ep]),
                        transfer_norm=float(output["transfer"][ep].norm()),
                        exposure_count=int(world.exposure_count[ep]),
                        compute_budget=world.bridge_index + 1 + tick,
                    ))
    return rows
