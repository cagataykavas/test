from worker import run_experiment


def test_runner_is_deterministic():
    kwargs = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Ankara hangi ülkenin başkentidir?",
        "analysis_type": "layer_margin",
        "config": {"layer_count": 8},
    }

    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first == second
    assert first["summary"]["layers_analyzed"] == 8
    assert len(first["trajectory"]) == 8
