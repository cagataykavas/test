from __future__ import annotations

import hashlib
import math
from typing import Any


def run_experiment(
    model_name: str,
    prompt: str,
    analysis_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Adapter for Struct-XAI experiments.

    The cloud service intentionally keeps orchestration separate from the
    research scripts. This deterministic fallback produces inspectable demo
    metrics for local/API testing; production integrations can replace this
    function with calls into the corresponding Struct-XAI analysis modules.
    """
    digest = hashlib.sha256(
        f"{model_name}|{prompt}|{analysis_type}|{sorted(config.items())}".encode("utf-8")
    ).hexdigest()
    seed_value = int(digest[:8], 16)

    layer_count = int(config.get("layer_count", 12))
    layer_count = max(2, min(layer_count, 96))

    trajectory = []
    for layer in range(layer_count):
        phase = (seed_value % 997) / 997.0
        margin = math.tanh((layer / max(layer_count - 1, 1) - 0.5) * 3.0 + phase - 0.5)
        trajectory.append({"layer": layer, "decision_margin": round(float(margin), 6)})

    strongest = max(trajectory, key=lambda item: abs(item["decision_margin"]))
    sign_flips = sum(
        1
        for a, b in zip(trajectory, trajectory[1:])
        if (a["decision_margin"] < 0 <= b["decision_margin"])
        or (a["decision_margin"] >= 0 > b["decision_margin"])
    )

    return {
        "analysis_type": analysis_type,
        "model_name": model_name,
        "summary": {
            "layers_analyzed": layer_count,
            "strongest_layer": strongest["layer"],
            "strongest_margin": strongest["decision_margin"],
            "sign_flips": sign_flips,
        },
        "trajectory": trajectory,
        "note": "Deterministic service-layer demo. Replace worker.run_experiment with a direct adapter to the research scripts for full model execution.",
    }
