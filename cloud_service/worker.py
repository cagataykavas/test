from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any


def _run_demo(
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
        f"{model_name}|{prompt}|{analysis_type}|{sorted(config.items())}".encode()
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
        for a, b in itertools.pairwise(trajectory)
        if (a["decision_margin"] < 0 <= b["decision_margin"])
        or (a["decision_margin"] >= 0 > b["decision_margin"])
    )

    return {
        "backend": "demo",
        "is_model_execution": False,
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


SUPPORTED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SUPPORTED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def validate_config(model_name: str, analysis_type: str, config: dict) -> None:
    """Fail before model loading or persistence for unsupported requests."""
    backend = config.get("backend", "demo")
    if backend not in {"demo", "huggingface_local"}:
        raise ValueError("unsupported backend")
    if backend == "demo":
        return
    if model_name != SUPPORTED_MODEL or config.get("revision") != SUPPORTED_REVISION:
        raise ValueError("real backend requires the allowlisted model and immutable revision")
    if analysis_type != "layer_margin":
        raise ValueError("real backend supports layer_margin only")
    if set(config) - {"backend", "revision", "candidate_a", "candidate_b"}:
        raise ValueError("unsupported real-backend config field")
    candidates = [config.get("candidate_a"), config.get("candidate_b")]
    if any(not isinstance(c, str) or not c.strip() or len(c) > 100 for c in candidates):
        raise ValueError("two nonempty candidates of at most 100 characters are required")
    if candidates[0] == candidates[1]:
        raise ValueError("candidates must differ")


def run_experiment(model_name: str, prompt: str, analysis_type: str, config: dict) -> dict:
    validate_config(model_name, analysis_type, config)
    if len(prompt) > 4096:
        raise ValueError("prompt exceeds character budget")
    if config.get("backend", "demo") == "demo":
        return _run_demo(model_name, prompt, analysis_type, config)
    from real_backend import run_layer_margin
    return run_layer_margin(model_name, prompt, config)
