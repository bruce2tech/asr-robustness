"""Composition of degradation effects into named, reproducible conditions.

A *condition* is an ordered list of stages (see ``configs/degradation.yaml``).
The :class:`DegradationPipeline` applies a condition to a clean signal using a
per-utterance integer seed; the seed drives every random choice (noise offset,
which noise clip, which RIR), so degrading utterance *u* under condition *c*
always yields the identical result.

Every applied stage returns an ``info`` dict; the pipeline collects these into
a metadata record that is written to the evaluation manifest, making each
degraded utterance fully traceable and the results sliceable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from asr_robustness.degrade import effects
from asr_robustness.degrade.banks import NoiseBank, RIRBank

Stage = dict[str, Any]
Condition = list[Stage]


def load_conditions(path: str | Path) -> dict[str, Condition]:
    """Load named conditions from a YAML config.

    Conditions listed under ``conditions:`` are returned as-is. If the config
    also defines ``snr_sweep_db``, one ``noise_<n>db`` condition is synthesized
    per SNR value so the WER-vs-SNR experiment is fully config-driven.
    """
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    conditions: dict[str, Condition] = dict(cfg.get("conditions", {}))
    for snr in cfg.get("snr_sweep_db", []):
        conditions[f"noise_{snr}db"] = [
            {"effect": "add_noise", "snr_db": snr, "noise_type": "noise"}
        ]
    for snr in cfg.get("babble_sweep_db", []):
        conditions[f"babble_{snr}db"] = [
            {"effect": "add_babble", "snr_db": snr, "n_talkers": 6, "noise_type": "speech"}
        ]
    if not conditions:
        raise ValueError(f"no conditions defined in {path}")
    return conditions


class DegradationPipeline:
    """Applies named degradation conditions to clean audio, reproducibly."""

    def __init__(
        self,
        conditions: dict[str, Condition],
        noise_bank: NoiseBank | None = None,
        rir_bank: RIRBank | None = None,
    ):
        self.conditions = conditions
        self.noise_bank = noise_bank
        self.rir_bank = rir_bank

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        noise_bank: NoiseBank | None = None,
        rir_bank: RIRBank | None = None,
    ) -> "DegradationPipeline":
        return cls(load_conditions(path), noise_bank=noise_bank, rir_bank=rir_bank)

    def condition_names(self) -> list[str]:
        return list(self.conditions)

    def _apply_stage(
        self, x: np.ndarray, sr: int, stage: Stage, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        params = {k: v for k, v in stage.items() if k != "effect"}
        effect = stage["effect"]

        if effect == "add_noise":
            if self.noise_bank is None:
                raise ValueError("condition uses 'add_noise' but no noise bank was provided")
            noise_id, noise = self.noise_bank.sample(rng, params.pop("noise_type", None))
            y, info = effects.add_noise(x, noise, params["snr_db"], rng)
            info["noise_id"] = noise_id
            return y, info

        if effect == "add_babble":
            if self.noise_bank is None:
                raise ValueError("condition uses 'add_babble' but no noise bank was provided")
            n_talkers = int(params.get("n_talkers", 6))
            prefix = params.get("noise_type", "speech")
            talkers, talker_ids = [], []
            for _ in range(n_talkers):
                tid, clip = self.noise_bank.sample(rng, prefix)
                talkers.append(clip)
                talker_ids.append(tid)
            y, info = effects.add_babble(x, talkers, params["snr_db"], rng)
            info["talker_ids"] = talker_ids
            return y, info

        if effect == "add_reverb":
            if self.rir_bank is None:
                raise ValueError("condition uses 'add_reverb' but no RIR bank was provided")
            rir_id, rir = self.rir_bank.sample(rng, params.pop("rir_type", None))
            y, info = effects.add_reverb(x, rir)
            info["rir_id"] = rir_id
            return y, info

        if effect == "narrowband":
            return effects.narrowband(x, sr, **params)
        if effect == "mu_law_codec":
            return effects.mu_law_codec(x, **params)
        if effect == "clip":
            return effects.clip_signal(x, **params)
        if effect == "gain":
            return effects.gain(x, **params)
        if effect == "packet_loss":
            return effects.packet_loss(x, sr, rng=rng, **params)
        if effect == "apply_codec":
            return effects.apply_codec(x, sr, **params)

        raise ValueError(f"unknown effect: {effect!r}")

    def apply(
        self, clean: np.ndarray, sr: int, condition: str, seed: int
    ) -> tuple[np.ndarray, dict]:
        """Degrade ``clean`` under ``condition``; returns ``(degraded, metadata)``.

        ``metadata`` records the condition, seed, and the realized ``info`` of
        every stage -- enough to exactly reproduce and to analyze the result.
        """
        if condition not in self.conditions:
            raise KeyError(f"unknown condition: {condition!r}")
        rng = np.random.default_rng(seed)
        x = np.asarray(clean, dtype=np.float32)
        stage_infos: list[dict] = []
        for stage in self.conditions[condition]:
            x, info = self._apply_stage(x, sr, dict(stage), rng)
            stage_infos.append(info)
        metadata = {"condition": condition, "seed": int(seed), "stages": stage_infos}
        return x, metadata
