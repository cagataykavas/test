"""
Generate a presentation-ready markdown summary from struct_xai_results.json.

Usage:
  python 16_struct_xai_report.py --in-json struct_xai_results.json --out-md mock_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_layers(base_rows: list[dict], top_n: int = 5) -> list[dict]:
    scored = sorted(base_rows, key=lambda r: r.get("target_logprob", -1e9), reverse=True)
    return scored[:top_n]


def biggest_deltas(base_rows: list[dict], variant_rows: list[dict], top_n: int = 5) -> list[tuple[int, float]]:
    deltas: list[tuple[int, float]] = []
    for b, v in zip(base_rows, variant_rows):
        deltas.append((int(b["layer"]), float(v["target_logprob"] - b["target_logprob"])))
    return sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", default="struct_xai_results.json")
    parser.add_argument("--out-md", default="struct_xai_mock_report.md")
    args = parser.parse_args()

    payload = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    base = payload.get("base", [])
    variants = payload.get("variants", [])
    exit_layer = payload.get("router_exit_layer")

    lines: list[str] = []
    lines.append("# Struct-XAI Mock Report (Auto-Generated)")
    lines.append("")
    lines.append("## 1) Experiment Setup")
    lines.append(f"- Model: `{cfg.get('model', 'N/A')}`")
    lines.append(f"- Device: `{cfg.get('device', 'N/A')}`")
    lines.append(f"- Target continuation: `{cfg.get('target', 'N/A')}`")
    lines.append(f"- Delete words: `{cfg.get('delete', [])}`")
    lines.append(f"- Replace map: `{cfg.get('replace', [])}`")
    lines.append("")

    if base:
        lines.append("## 2) Layer Highlights (Base Prompt)")
        for row in best_layers(base, top_n=5):
            lines.append(
                f"- Layer {row['layer']}: top_token={row['top_token']!r}, "
                f"target_logprob={row['target_logprob']:.3f}"
            )
        lines.append("")

    if variants and base and "rows" in variants[0]:
        lines.append("## 3) Largest Ablation Effects")
        for var in variants:
            name = var.get("name", "variant")
            lines.append(f"### {name}")
            for layer, delta in biggest_deltas(base, var.get("rows", []), top_n=5):
                lines.append(f"- Layer {layer}: Δ target_logprob = {delta:+.3f}")
        lines.append("")

    lines.append("## 4) Router Diagnostic")
    if exit_layer is None:
        lines.append("- No stable early-exit layer under current thresholds.")
    else:
        lines.append(f"- Candidate early-exit layer: **{exit_layer}**")
    lines.append(
        "- Note: this is a diagnostic/counterfactual indicator, not a true execution-skipping implementation yet."
    )
    lines.append("")

    lines.append("## 5) Ready-to-Present Narrative")
    lines.append(
        "- We tracked how the next-token belief evolved layer-by-layer and measured how targeted edits changed the same target continuation score."
    )
    lines.append(
        "- This supports a concrete claim: specific tokens/spans shift internal belief trajectories at identifiable layers."
    )
    lines.append(
        "- We also provide a stability-based candidate exit layer as a roadmap toward real Green-AI early-exit engineering."
    )

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[info] wrote report -> {args.out_md}")


if __name__ == "__main__":
    main()
