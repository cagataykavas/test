"""Command-line interface for Struct-XAI JSON trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from struct_xai.ablation import AblationSummary, compare_ablation
from struct_xai.metrics import summarize_trajectory
from struct_xai.report import build_explainable_report
from struct_xai.schema import TrajectorySummary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="struct-xai",
        description="Summarize layer-wise target/distractor logits as explainable JSON.",
    )
    parser.add_argument("input", type=Path, help="Input JSON trajectory")
    parser.add_argument("--output", type=Path, help="Write the report to this path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the positive-gap threshold from the input (default: 0.0)",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=None,
        help="Override the sustained-layer count from the input (default: 2)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("input JSON must be an object")
    return data


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"missing required field: {key}")
    return data[key]


def _summarize(
    data: dict[str, Any],
    *,
    threshold: float,
    min_consecutive: int,
    example_id_suffix: str = "",
) -> TrajectorySummary:
    return summarize_trajectory(
        model_id=str(_required(data, "model_id")),
        example_id=f"{_required(data, 'example_id')}{example_id_suffix}",
        target_token=str(_required(data, "target_token")),
        distractor_token=str(_required(data, "distractor_token")),
        target_logits=_required(data, "target_logits"),
        distractor_logits=_required(data, "distractor_logits"),
        layer_numbers=data.get("layer_numbers"),
        top_tokens=data.get("top_tokens"),
        threshold=threshold,
        min_consecutive=min_consecutive,
    )


def _summarize_ablations(
    base_data: dict[str, Any],
    base: TrajectorySummary,
    *,
    threshold: float,
    min_consecutive: int,
) -> list[AblationSummary]:
    raw_ablations = base_data.get("ablations", [])
    if not isinstance(raw_ablations, Sequence) or isinstance(raw_ablations, (str, bytes)):
        raise ValueError("ablations must be an array")

    summaries: list[AblationSummary] = []
    for index, raw in enumerate(raw_ablations):
        if not isinstance(raw, dict):
            raise ValueError(f"ablation {index} must be an object")
        feature = str(_required(raw, "feature"))
        merged = {
            **base_data,
            "target_logits": _required(raw, "target_logits"),
            "distractor_logits": _required(raw, "distractor_logits"),
            "top_tokens": raw.get("top_tokens"),
            "ablations": [],
        }
        ablated = _summarize(
            merged,
            threshold=threshold,
            min_consecutive=min_consecutive,
            example_id_suffix=f"::ablation::{feature}",
        )
        summaries.append(compare_ablation(base, ablated, feature=feature))
    return summaries


def run(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        data = _load_json(args.input)
        threshold = float(data.get("threshold", 0.0) if args.threshold is None else args.threshold)
        min_consecutive = int(
            data.get("min_consecutive", 2)
            if args.min_consecutive is None
            else args.min_consecutive
        )
        base = _summarize(
            data,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )
        ablations = _summarize_ablations(
            data,
            base,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )
        report = build_explainable_report(
            base,
            ablations=ablations,
            metadata=data.get("metadata"),
        )
        rendered = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        ) + "\n"

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"struct-xai: error: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
