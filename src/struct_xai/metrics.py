"""Model-agnostic metrics for layer-wise target/distractor trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from struct_xai.schema import LayerEvidence, TrajectorySummary


def _as_finite_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resolve_layers(length: int, layer_numbers: Sequence[int] | None) -> tuple[int, ...]:
    if layer_numbers is None:
        return tuple(range(length))

    layers = tuple(int(layer) for layer in layer_numbers)
    if len(layers) != length:
        raise ValueError("layer_numbers must match the trajectory length")
    if any(layer < 0 for layer in layers):
        raise ValueError("layer_numbers must be non-negative")
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ValueError("layer_numbers must be unique and sorted")
    return layers


def target_distractor_gap(
    target_logits: ArrayLike,
    distractor_logits: ArrayLike,
) -> NDArray[np.float64]:
    """Return target logit minus distractor logit at every layer."""

    target = _as_finite_vector("target_logits", target_logits)
    distractor = _as_finite_vector("distractor_logits", distractor_logits)
    if target.shape != distractor.shape:
        raise ValueError("target_logits and distractor_logits must have the same shape")
    return target - distractor


def first_sustained_positive_layer(
    gaps: ArrayLike,
    *,
    threshold: float = 0.0,
    min_consecutive: int = 2,
    layer_numbers: Sequence[int] | None = None,
) -> int | None:
    """Return the first layer starting a sustained positive run."""

    values = _as_finite_vector("gaps", gaps)
    layers = _resolve_layers(len(values), layer_numbers)
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if min_consecutive > len(values):
        return None

    positive = values > threshold
    for start in range(0, len(values) - min_consecutive + 1):
        if bool(np.all(positive[start : start + min_consecutive])):
            return layers[start]
    return None


def stable_positive_from_layer(
    gaps: ArrayLike,
    *,
    threshold: float = 0.0,
    min_remaining: int = 2,
    layer_numbers: Sequence[int] | None = None,
) -> int | None:
    """Return the earliest layer after which all remaining gaps stay positive."""

    values = _as_finite_vector("gaps", gaps)
    layers = _resolve_layers(len(values), layer_numbers)
    if min_remaining < 1:
        raise ValueError("min_remaining must be at least 1")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    final_start = len(values) - min_remaining
    for start in range(max(final_start + 1, 0)):
        if bool(np.all(values[start:] > threshold)):
            return layers[start]
    return None


def count_sign_flips(gaps: ArrayLike, *, threshold: float = 0.0) -> int:
    """Count sign changes while ignoring values in the threshold dead zone."""

    values = _as_finite_vector("gaps", gaps)
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")

    signs = np.zeros(values.shape, dtype=np.int8)
    signs[values > threshold] = 1
    signs[values < -threshold] = -1
    non_zero = signs[signs != 0]
    if non_zero.size < 2:
        return 0
    return int(np.count_nonzero(non_zero[1:] != non_zero[:-1]))


def summarize_trajectory(
    *,
    model_id: str,
    example_id: str,
    target_token: str,
    distractor_token: str,
    target_logits: ArrayLike,
    distractor_logits: ArrayLike,
    layer_numbers: Sequence[int] | None = None,
    top_tokens: Sequence[str | None] | None = None,
    threshold: float = 0.0,
    min_consecutive: int = 2,
) -> TrajectorySummary:
    """Validate a trajectory and calculate all standard Struct-XAI metrics."""

    target = _as_finite_vector("target_logits", target_logits)
    distractor = _as_finite_vector("distractor_logits", distractor_logits)
    if target.shape != distractor.shape:
        raise ValueError("target_logits and distractor_logits must have the same shape")

    layers = _resolve_layers(len(target), layer_numbers)
    if top_tokens is None:
        resolved_top_tokens: tuple[str | None, ...] = (None,) * len(target)
    else:
        resolved_top_tokens = tuple(top_tokens)
        if len(resolved_top_tokens) != len(target):
            raise ValueError("top_tokens must match the trajectory length")

    gaps = target - distractor
    peak_index = int(np.argmax(gaps))
    evidence = tuple(
        LayerEvidence(
            layer=layers[index],
            target_logit=float(target[index]),
            distractor_logit=float(distractor[index]),
            gap=float(gaps[index]),
            top_token=resolved_top_tokens[index],
        )
        for index in range(len(target))
    )

    return TrajectorySummary(
        model_id=model_id,
        example_id=example_id,
        target_token=target_token,
        distractor_token=distractor_token,
        threshold=float(threshold),
        min_consecutive=min_consecutive,
        decision_layer=first_sustained_positive_layer(
            gaps,
            threshold=threshold,
            min_consecutive=min_consecutive,
            layer_numbers=layers,
        ),
        stable_from_layer=stable_positive_from_layer(
            gaps,
            threshold=threshold,
            min_remaining=min_consecutive,
            layer_numbers=layers,
        ),
        peak_layer=layers[peak_index],
        final_gap=float(gaps[-1]),
        peak_gap=float(gaps[peak_index]),
        sign_flips=count_sign_flips(gaps, threshold=abs(float(threshold))),
        layers=evidence,
    )
