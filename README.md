# Struct-XAI

[![CI](https://github.com/cagataykavas/test/actions/workflows/ci.yml/badge.svg)](https://github.com/cagataykavas/test/actions/workflows/ci.yml)

Struct-XAI is a small research toolkit for inspecting how a target-vs-distractor preference changes across transformer layers. It turns layer-wise logits into reproducible metrics and an explainable JSON report that can be checked without loading a language model.

The repository currently contains two layers of work:

- `src/struct_xai/`: the tested, model-agnostic core.
- numbered root scripts: earlier exploratory experiments for Hugging Face and TransformerLens models. They are retained as research history and will be migrated behind stable adapters.

## What this demonstrates

- Explicit metric definitions rather than conclusions hidden inside plotting code.
- Typed, JSON-serializable evidence for every inspected layer.
- Token-ablation comparisons with the direction and magnitude of each effect.
- Deterministic tests that do not download a model or require a GPU.
- A clean boundary between model extraction and analysis.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
struct-xai examples/sample_input.json --pretty
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run the validation suite:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

## Input format

The CLI accepts a JSON object containing the target and distractor logits observed at each layer:

```json
{
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "example_id": "geo_ankara",
  "target_token": " Ankara",
  "distractor_token": " Istanbul",
  "layer_numbers": [0, 1, 2, 3],
  "target_logits": [0.1, 0.5, 1.4, 2.1],
  "distractor_logits": [0.4, 0.4, 0.8, 1.0],
  "ablations": [
    {
      "feature": "capital cue",
      "target_logits": [0.0, 0.2, 0.6, 0.8],
      "distractor_logits": [0.4, 0.5, 0.8, 1.0]
    }
  ]
}
```

The report includes the raw layer evidence, final and peak gap, first sustained positive decision layer, stable-positive layer, sign-flip count, and ablation effects. Every summary field is traceable to the arrays in the same output.

## Metric definitions

For layer `l`, the preference gap is:

```text
gap(l) = target_logit(l) - distractor_logit(l)
```

- `decision_layer`: first layer where the gap stays above the configured threshold for `N` consecutive layers.
- `stable_from_layer`: earliest layer after which all remaining gaps stay above the threshold.
- `sign_flips`: number of positive-to-negative or negative-to-positive transitions, ignoring values inside the threshold dead zone.
- `support_effect`: base gap minus ablated gap. Positive values mean removing the feature reduced target preference.

See [the methodology guide](docs/methodology.md) for assumptions and limitations.

## Repository layout

```text
src/struct_xai/
  ablation.py       token-ablation comparisons
  cli.py            JSON-in / JSON-out command line interface
  metrics.py        layer-wise metrics
  report.py         explainable report assembly
  schema.py         validated evidence objects
docs/
  methodology.md
examples/
  sample_input.json
tests/
  test_ablation.py
  test_cli.py
  test_metrics.py
```

## Research status

This is a research prototype, not proof that hidden states are literal thoughts or that logit-lens projections fully explain a model. A logit gap is an observable proxy. Strong causal claims require controlled interventions, baselines, uncertainty estimates, and replication across prompts and models.

The next increments will add Hugging Face adapters, activation patching, batch experiment manifests, and plots generated only from saved report files.

## Author

Çağatay Kavas — electrical engineer and Computer Engineering M.S. student working on reinforcement learning, computer vision, anomaly detection, and interpretable language models.
