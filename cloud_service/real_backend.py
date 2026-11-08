"""Bounded, offline-only real model inference. No automatic download or remote code."""
from __future__ import annotations

import hashlib
import itertools
import platform
import threading
from functools import lru_cache

_RUN_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load(model_name: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32,
    ).to("cpu").eval()
    if model.config.model_type != "qwen2":
        raise ValueError("unsupported loaded architecture")
    return tokenizer, model


def run_layer_margin(model_name: str, prompt: str, config: dict) -> dict:
    import torch
    import transformers

    # Shared model hook/cache access is serialized. No unbounded waiting queue.
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("model worker busy")
    try:
        tokenizer, model = _load(model_name, config["revision"])
        encoded = tokenizer(prompt, return_tensors="pt")
        if not 1 <= encoded.input_ids.shape[-1] <= 128:
            raise ValueError("real backend accepts 1 to 128 prompt tokens")
        labels = [config["candidate_a"], config["candidate_b"]]
        ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(not row for row in ids) or ids[0][0] == ids[1][0]:
            raise ValueError("first-token metric cannot distinguish candidates: use nonempty, distinct first tokens")
        trajectory = []
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            states = output.hidden_states
            final_logits = model.lm_head(states[-1])[0, -1].float()
            native_logits = output.logits[0, -1].float()
            if not torch.isfinite(native_logits).all():
                raise ValueError("nonfinite model logits")
            torch.testing.assert_close(final_logits, native_logits, atol=0, rtol=0)
            for layer, state in enumerate(states):
                # HF's final state is already normalized; earlier states are not.
                logits = final_logits if layer == len(states) - 1 else model.lm_head(
                    model.model.norm(state[0, -1])
                ).float()
                scores = [float(logits[row[0]]) for row in ids]
                if not all(__import__("math").isfinite(x) for x in scores):
                    raise ValueError("nonfinite candidate scores")
                trajectory.append({"layer": layer, "candidate_scores": dict(zip(labels, scores)),
                                   "decision_margin": scores[0] - scores[1]})
        strongest = max(trajectory, key=lambda row: abs(row["decision_margin"]))
        flips = sum(a["decision_margin"] * b["decision_margin"] < 0
                    for a, b in itertools.pairwise(trajectory))
        return {
            "backend": "huggingface_local", "is_model_execution": True,
            "model_name": model_name, "analysis_type": "layer_margin",
            "metric": "first_token_candidate_logit_margin", "prompt_format": "raw",
            "summary": {"layers_analyzed": len(trajectory), "sign_flips": flips,
                        "strongest_layer": strongest["layer"],
                        "strongest_margin": strongest["decision_margin"]},
            "trajectory": trajectory,
            "candidates": [{"label": label, "token_ids": tokens} for label, tokens in zip(labels, ids)],
            "provenance": {"model_revision": config["revision"],
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "prompt_tokens": int(encoded.input_ids.shape[-1]),
                           "python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "device": "cpu", "dtype": "float32"},
            "native_final_logit_max_error": float((final_logits - native_logits).abs().max()),
            "note": "Real local model execution. First-token margins are not sequence probabilities or causal faithfulness. No cloud deployment claim.",
        }
    finally:
        _RUN_LOCK.release()
