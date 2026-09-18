"""Frozen Stage-1.3 baseline constructors."""

from __future__ import annotations

from etrcm.stage1_3.model import ContinuousETRCM, Stage13Config


MODEL_NAMES = (
    "B0_no_persistent",
    "B1_gru",
    "B2_single_persistent",
    "B3_joint",
    "B4_m_only",
    "B5_f_only",
    "B6_arbitration",
    "B7_nonconserving_self_replay",
)


def build_model(name: str, config: Stage13Config) -> ContinuousETRCM:
    if name == "B0_no_persistent":
        return ContinuousETRCM(config, read_mode="none")
    if name == "B1_gru":
        return ContinuousETRCM(config, read_mode="none", core_kind="gru")
    if name == "B2_single_persistent":
        return ContinuousETRCM(config, read_mode="single_persistent")
    if name == "B3_joint":
        return ContinuousETRCM(config, read_mode="joint")
    if name == "B4_m_only":
        return ContinuousETRCM(config, read_mode="m_only")
    if name == "B5_f_only":
        return ContinuousETRCM(config, read_mode="f_only")
    if name == "B6_arbitration":
        return ContinuousETRCM(config, read_mode="arbitration")
    if name == "B7_nonconserving_self_replay":
        return ContinuousETRCM(config, read_mode="arbitration", self_output_writes=True)
    raise ValueError(name)
