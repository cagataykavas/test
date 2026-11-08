import pytest
from pydantic import ValidationError
from schemas import ExperimentCreate
from worker import SUPPORTED_MODEL, SUPPORTED_REVISION, run_experiment


def payload(**config):
    return {"name": "test", "model_name": SUPPORTED_MODEL, "prompt": "Hello", "config": {
        "backend": "huggingface_local", "revision": SUPPORTED_REVISION,
        "candidate_a": " yes", "candidate_b": " no", **config,
    }}


@pytest.mark.parametrize("change", [
    {"backend": "invalid"}, {"revision": "main"}, {"candidate_a": ""},
    {"candidate_b": " yes"}, {"candidate_a": "x" * 101}, {"device": "cuda"},
])
def test_reject_invalid_config(change):
    with pytest.raises(ValidationError):
        ExperimentCreate(**payload(**change))


def test_accept_pinned_config():
    assert ExperimentCreate(**payload()).config["backend"] == "huggingface_local"


def test_demo_is_explicitly_synthetic():
    result = run_experiment("demo", "hello", "layer_margin", {})
    assert result["backend"] == "demo"
    assert result["is_model_execution"] is False


def test_unsupported_analysis():
    p = payload()
    p["analysis_type"] = "shap"
    with pytest.raises(ValidationError):
        ExperimentCreate(**p)


def test_character_budget():
    p = payload()
    p["prompt"] = "a" * 4097
    with pytest.raises(ValidationError):
        ExperimentCreate(**p)


def test_busy_worker_fails_without_loading(monkeypatch):
    import real_backend
    monkeypatch.setattr(real_backend, "_load", lambda *args: pytest.fail("must not load while busy"))
    assert real_backend._RUN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            real_backend.run_layer_margin(SUPPORTED_MODEL, "hello", payload()["config"])
    finally:
        real_backend._RUN_LOCK.release()


def test_first_token_collision_rejected_before_forward(monkeypatch):
    from types import SimpleNamespace

    import real_backend
    import torch

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

        def encode(self, label, **kwargs):
            # Different complete answers, identical leading-space token.
            return {" 4": [220, 19], " 5": [220, 20]}[label]

    class MustNotRun:
        def __call__(self, **kwargs):
            pytest.fail("ambiguous candidates must be rejected before model forward")

    monkeypatch.setattr(real_backend, "_load", lambda *args: (Tokenizer(), MustNotRun()))
    config = payload(candidate_a=" 4", candidate_b=" 5")["config"]
    with pytest.raises(ValueError, match="first-token metric cannot distinguish candidates"):
        real_backend.run_layer_margin(SUPPORTED_MODEL, "2 + 2?", config)
    # Error paths must release the shared inference lock.
    assert real_backend._RUN_LOCK.acquire(blocking=False)
    real_backend._RUN_LOCK.release()
