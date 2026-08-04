# Methodology

Struct-XAI separates evidence extraction from evidence interpretation. Model-specific code is responsible for saving aligned layer-wise values. The core package validates those values and computes metrics without loading the original model.

## Observation unit

An observation is one prompt, one model, one target token, and one distractor token. For every inspected layer, the extractor records:

- the layer number;
- target logit;
- distractor logit;
- optionally, the top projected token.

Comparing raw token strings across tokenizers is unreliable. Cross-model studies should define semantically equivalent token families and record their token IDs alongside the report metadata.

## Preference gap

The primary observable is:

```text
gap(l) = target_logit(l) - distractor_logit(l)
```

A positive value means the target is preferred over the distractor under the chosen projection at layer `l`. It does not mean the model has completed a human-like reasoning step.

## Decision layer

Single-layer crossings are noisy, so the decision layer requires a sustained run:

```text
gap(l), ..., gap(l + N - 1) > threshold
```

The default is two consecutive layers and a zero threshold. Both parameters are stored in the report.

## Stability and sign flips

`stable_from_layer` is the earliest layer after which all remaining gaps stay positive. `sign_flips` counts positive-to-negative and negative-to-positive transitions after values inside a configurable dead zone are discarded. Together they distinguish early commitment from unstable preference oscillation.

## Ablation effect

For a feature removed from the input:

```text
support_effect(l) = base_gap(l) - ablated_gap(l)
```

A positive effect means the removal weakened the target preference at that layer. This is an intervention, but it is not automatically a clean causal estimate: deletion can alter tokenization, syntax, length, or position. Recommended controls include:

1. length-matched neutral replacements;
2. multiple paraphrases;
3. random-token and shuffled-token baselines;
4. confidence intervals across examples;
5. activation patching when architecture access permits it.

## Reproducibility checklist

Each experiment should record:

- exact model revision and tokenizer revision;
- prompt template and chat-template version;
- target and distractor token IDs;
- dtype, device, library versions, and random seeds;
- layers inspected and normalization/projection method;
- all thresholds chosen before looking at final results;
- exclusions and failed examples.

## Claims this package does not make

- Hidden-state projections are not literal thoughts.
- A high logit is not calibrated confidence.
- Early preference is not equivalent to safe early-exit inference.
- A single ablation is not sufficient evidence of general causality.

These boundaries are deliberate. A portfolio-grade research repository should make it easy to audit what was measured and difficult to accidentally overstate what was learned.
