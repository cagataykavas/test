"""Explainable, deterministic JSON reporting for Struct-XAI evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from struct_xai.ablation import AblationSummary
from struct_xai.schema import TrajectorySummary

SCHEMA_VERSION = "1.0"


def build_explainable_report(
    trajectory: TrajectorySummary,
    *,
    ablations: Sequence[AblationSummary] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe report with data, definitions, and limitations."""

    metadata_dict = dict(metadata or {})
    try:
        json.dumps(metadata_dict, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON serializable and finite") from exc

    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata_dict,
        "trajectory": trajectory.to_dict(),
        "ablations": [item.to_dict() for item in ablations],
        "definitions": {
            "gap": "target_logit - distractor_logit",
            "decision_layer": (
                "first layer beginning a run of min_consecutive gaps strictly above threshold"
            ),
            "stable_from_layer": (
                "earliest layer after which every remaining gap is strictly above threshold"
            ),
            "sign_flips": (
                "positive/negative transitions after removing values inside the threshold dead zone"
            ),
            "support_effect": (
                "base_gap - ablated_gap; positive means the removed feature supported the target"
            ),
        },
        "limitations": [
            "Logit-lens projections are observable proxies, not literal model thoughts.",
            (
                "Ablation can change tokenization or grammar and should be paired with "
                "controlled baselines."
            ),
            "Single-example effects do not establish population-level causality.",
        ],
    }
    json.dumps(report, allow_nan=False)
    return report


def write_explainable_report(
    path: str | Path,
    trajectory: TrajectorySummary,
    *,
    ablations: Sequence[AblationSummary] = (),
    metadata: Mapping[str, Any] | None = None,
    pretty: bool = True,
) -> Path:
    """Write a report to disk for later analysis or plotting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_explainable_report(
        trajectory,
        ablations=ablations,
        metadata=metadata,
    )
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
