"""Validated evidence objects used by Struct-XAI reports."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def _require_finite(name: str, value: float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class LayerEvidence:
    """Target and distractor evidence observed at one model layer."""

    layer: int
    target_logit: float
    distractor_logit: float
    gap: float
    top_token: str | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        _require_finite("target_logit", self.target_logit)
        _require_finite("distractor_logit", self.distractor_logit)
        _require_finite("gap", self.gap)

        expected_gap = float(self.target_logit) - float(self.distractor_logit)
        if abs(expected_gap - float(self.gap)) > 1e-9:
            raise ValueError(
                "gap must equal target_logit - distractor_logit "
                f"({expected_gap} != {self.gap})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "target_logit": float(self.target_logit),
            "distractor_logit": float(self.distractor_logit),
            "gap": float(self.gap),
            "top_token": self.top_token,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    """Summary statistics and complete evidence for one prompt trajectory."""

    model_id: str
    example_id: str
    target_token: str
    distractor_token: str
    threshold: float
    min_consecutive: int
    decision_layer: int | None
    stable_from_layer: int | None
    peak_layer: int
    final_gap: float
    peak_gap: float
    sign_flips: int
    layers: tuple[LayerEvidence, ...]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be at least 1")
        if not self.layers:
            raise ValueError("layers cannot be empty")
        if self.sign_flips < 0:
            raise ValueError("sign_flips cannot be negative")

        layer_numbers = [item.layer for item in self.layers]
        if layer_numbers != sorted(layer_numbers) or len(layer_numbers) != len(set(layer_numbers)):
            raise ValueError("layers must be unique and sorted")
        if self.peak_layer not in set(layer_numbers):
            raise ValueError("peak_layer must refer to an observed layer")
        if self.decision_layer is not None and self.decision_layer not in set(layer_numbers):
            raise ValueError("decision_layer must refer to an observed layer")
        if self.stable_from_layer is not None and self.stable_from_layer not in set(layer_numbers):
            raise ValueError("stable_from_layer must refer to an observed layer")

        _require_finite("threshold", self.threshold)
        _require_finite("final_gap", self.final_gap)
        _require_finite("peak_gap", self.peak_gap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "example_id": self.example_id,
            "target_token": self.target_token,
            "distractor_token": self.distractor_token,
            "threshold": float(self.threshold),
            "min_consecutive": int(self.min_consecutive),
            "decision_layer": self.decision_layer,
            "stable_from_layer": self.stable_from_layer,
            "peak_layer": int(self.peak_layer),
            "final_gap": float(self.final_gap),
            "peak_gap": float(self.peak_gap),
            "sign_flips": int(self.sign_flips),
            "layers": [item.to_dict() for item in self.layers],
        }
