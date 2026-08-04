"""Comparisons between a base trajectory and a controlled ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from struct_xai.schema import TrajectorySummary


@dataclass(frozen=True, slots=True)
class LayerEffect:
    """Effect of removing one feature at a single layer."""

    layer: int
    base_gap: float
    ablated_gap: float
    support_effect: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "base_gap": float(self.base_gap),
            "ablated_gap": float(self.ablated_gap),
            "support_effect": float(self.support_effect),
        }


@dataclass(frozen=True, slots=True)
class AblationSummary:
    """Aggregate and layer-level effects for one removed feature."""

    feature: str
    base_decision_layer: int | None
    ablated_decision_layer: int | None
    decision_layer_shift: int | None
    mean_support_effect: float
    final_support_effect: float
    max_absolute_effect: float
    max_absolute_effect_layer: int
    layers: tuple[LayerEffect, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "base_decision_layer": self.base_decision_layer,
            "ablated_decision_layer": self.ablated_decision_layer,
            "decision_layer_shift": self.decision_layer_shift,
            "mean_support_effect": float(self.mean_support_effect),
            "final_support_effect": float(self.final_support_effect),
            "max_absolute_effect": float(self.max_absolute_effect),
            "max_absolute_effect_layer": int(self.max_absolute_effect_layer),
            "layers": [item.to_dict() for item in self.layers],
        }


def compare_ablation(
    base: TrajectorySummary,
    ablated: TrajectorySummary,
    *,
    feature: str,
) -> AblationSummary:
    """Compare aligned trajectories using base gap minus ablated gap."""

    if not feature.strip():
        raise ValueError("feature cannot be empty")
    if base.model_id != ablated.model_id:
        raise ValueError("base and ablated trajectories must use the same model")
    same_target_pair = (
        base.target_token == ablated.target_token
        and base.distractor_token == ablated.distractor_token
    )
    if not same_target_pair:
        raise ValueError("base and ablated trajectories must use the same target pair")

    base_layers = tuple(item.layer for item in base.layers)
    ablated_layers = tuple(item.layer for item in ablated.layers)
    if base_layers != ablated_layers:
        raise ValueError("base and ablated trajectories must contain the same layers")

    base_gaps = np.asarray([item.gap for item in base.layers], dtype=np.float64)
    ablated_gaps = np.asarray([item.gap for item in ablated.layers], dtype=np.float64)
    effects = base_gaps - ablated_gaps
    max_index = int(np.argmax(np.abs(effects)))

    if base.decision_layer is not None and ablated.decision_layer is not None:
        decision_layer_shift: int | None = ablated.decision_layer - base.decision_layer
    else:
        decision_layer_shift = None

    layer_effects = tuple(
        LayerEffect(
            layer=base_layers[index],
            base_gap=float(base_gaps[index]),
            ablated_gap=float(ablated_gaps[index]),
            support_effect=float(effects[index]),
        )
        for index in range(len(base_layers))
    )

    return AblationSummary(
        feature=feature,
        base_decision_layer=base.decision_layer,
        ablated_decision_layer=ablated.decision_layer,
        decision_layer_shift=decision_layer_shift,
        mean_support_effect=float(np.mean(effects)),
        final_support_effect=float(effects[-1]),
        max_absolute_effect=float(abs(effects[max_index])),
        max_absolute_effect_layer=base_layers[max_index],
        layers=layer_effects,
    )
