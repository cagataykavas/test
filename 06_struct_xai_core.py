"""
Struct-XAI core (workable baseline)

What this script gives you, quickly:
1) Layer-wise next-token trajectory (logit-lens style using model norm + lm_head)
2) Token-aware text ablations (delete/replace spans by character offsets)
3) Target-aware scoring (delta log-prob for a task-aligned target continuation)
4) A simple semantic-stability router analysis (diagnostic only)

This is intentionally small and reproducible so you can run it before your mock paper/slides.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Policy = Literal["delete", "replace"]


@dataclass
class SpanEdit:
    start: int
    end: int
    policy: Policy
    replace_text: str = ""


def pick_device() -> str:
    try:
        import torch  # local import so --help / --plan can run without torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_word_spans(text: str, word: str) -> list[tuple[int, int]]:
    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.UNICODE)]


def apply_text_edits(text: str, edits: list[SpanEdit]) -> str:
    if not edits:
        return text
    edits = sorted(edits, key=lambda e: (e.start, e.end))
    out = []
    cursor = 0
    for e in edits:
        out.append(text[cursor:e.start])
        if e.policy == "replace":
            out.append(e.replace_text)
        cursor = e.end
    out.append(text[cursor:])
    return "".join(out)


def get_layer_logits(prompt: str, model, tokenizer, device: str):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: embedding output + each layer output
    hidden_states = outputs.hidden_states
    final_norm = model.model.norm
    lm_head = model.lm_head

    layer_logits: list[torch.Tensor] = []
    for h in hidden_states:
        token_h = h[:, -1, :]  # last token position
        normalized = final_norm(token_h)
        logits = lm_head(normalized)  # [1, vocab]
        layer_logits.append(logits)

    return torch.stack(layer_logits, dim=0).squeeze(1)  # [n_layers+1, vocab]


def get_target_token_ids(tokenizer, target_text: str) -> list[int]:
    ids = tokenizer(target_text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Target text tokenized to empty sequence: {target_text!r}")
    return ids


def summarize_layers(layer_logits, tokenizer, target_first_token_id: int) -> list[dict]:
    import torch

    layer_probs = torch.log_softmax(layer_logits, dim=-1)
    top_vals, top_ids = torch.max(layer_probs, dim=-1)

    rows = []
    for i in range(layer_logits.shape[0]):
        tok_id = int(top_ids[i].item())
        tok = tokenizer.decode([tok_id])
        rows.append(
            {
                "layer": i,
                "top_token": tok,
                "top_logprob": float(top_vals[i].item()),
                "target_logprob": float(layer_probs[i, target_first_token_id].item()),
            }
        )
    return rows


def semantic_stability_exit(rows: list[dict], min_logprob: float, patience: int) -> int | None:
    streak = 0
    prev = None
    for row in rows:
        tok = row["top_token"]
        lp = row["top_logprob"]
        if lp >= min_logprob and tok == prev:
            streak += 1
        elif lp >= min_logprob:
            streak = 1
        else:
            streak = 0

        if streak >= patience:
            return int(row["layer"])
        prev = tok
    return None


def run_case(
    *,
    name: str,
    prompt: str,
    model,
    tokenizer,
    device: str,
    target_first_token_id: int,
) -> list[dict]:
    logits = get_layer_logits(prompt, model, tokenizer, device)
    rows = summarize_layers(logits, tokenizer, target_first_token_id)
    for r in rows:
        r["case"] = name
    return rows


def print_compact_table(base: list[dict], others: list[list[dict]]) -> None:
    by_case = {rows[0]["case"]: rows for rows in others}

    header = ["Layer", "BaseTok", "BaseTargetLP"] + [f"ΔTargetLP({name})" for name in by_case]
    print(" | ".join(f"{h:<18}" for h in header))
    print("-" * (22 * len(header)))

    for i, base_row in enumerate(base):
        row = [
            f"{base_row['layer']:<18}",
            f"{repr(base_row['top_token']):<18}",
            f"{base_row['target_logprob']:<18.3f}",
        ]
        for name, rows in by_case.items():
            delta = rows[i]["target_logprob"] - base_row["target_logprob"]
            row.append(f"{delta:<18.3f}")
        print(" | ".join(row))


def build_edits(prompt: str, delete_words: Iterable[str], replace_map: dict[str, str]) -> dict[str, str]:
    variants: dict[str, str] = {}

    for w in delete_words:
        spans = find_word_spans(prompt, w)
        if not spans:
            continue
        edits = [SpanEdit(start=s, end=e, policy="delete") for s, e in spans]
        variants[f"delete:{w}"] = apply_text_edits(prompt, edits)

    for src, dst in replace_map.items():
        spans = find_word_spans(prompt, src)
        if not spans:
            continue
        edits = [SpanEdit(start=s, end=e, policy="replace", replace_text=dst) for s, e in spans]
        variants[f"replace:{src}->{dst}"] = apply_text_edits(prompt, edits)

    return variants


def parse_kv_replacements(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --replace item {item!r}, expected form src=dst")
        src, dst = item.split("=", 1)
        out[src.strip()] = dst.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Workable Struct-XAI baseline runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model name")
    parser.add_argument("--device", default=pick_device(), help="cpu or cuda")
    parser.add_argument(
        "--prompt",
        default=(
            "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\n"
            "Girdi: Adam sinirle masaya vurdu ve bağırdı.\n"
            "Çıktı:\nSCENE_DIRECTIONS:\n-"
        ),
    )
    parser.add_argument("--target", default=" sahne")
    parser.add_argument("--delete", nargs="*", default=["sinirle", "masaya"])
    parser.add_argument("--replace", nargs="*", default=["Adam=Oyuncu"])
    parser.add_argument("--router-min-logprob", type=float, default=-3.0)
    parser.add_argument("--router-patience", type=int, default=2)
    parser.add_argument("--out-json", default="struct_xai_results.json")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Build ablation variants and save metadata without loading any model.",
    )
    args = parser.parse_args()

    variants = build_edits(args.prompt, args.delete, parse_kv_replacements(args.replace))
    if args.plan_only:
        payload = {
            "config": {
                "model": args.model,
                "device": args.device,
                "target": args.target,
                "delete": args.delete,
                "replace": args.replace,
                "router_min_logprob": args.router_min_logprob,
                "router_patience": args.router_patience,
                "plan_only": True,
            },
            "prompt": args.prompt,
            "variants": [{"name": k, "prompt": v} for k, v in variants.items()],
            "note": "Model execution skipped (--plan-only).",
        }
        out_path = Path(args.out_json)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] plan-only payload written -> {out_path}")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        print(
            "[error] Missing runtime dependency. Please install torch + transformers "
            "or run with --plan-only for non-model prep.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    device = args.device
    print(f"[info] loading model={args.model} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    target_ids = get_target_token_ids(tokenizer, args.target)
    target_first = target_ids[0]

    base = run_case(
        name="base",
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_first_token_id=target_first,
    )

    other_rows = []
    for name, edited_prompt in variants.items():
        rows = run_case(
            name=name,
            prompt=edited_prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_first_token_id=target_first,
        )
        other_rows.append(rows)

    print("\n=== Layer-wise delta table (target = first token of continuation) ===")
    if other_rows:
        print_compact_table(base, other_rows)
    else:
        print("No valid variants were created from --delete/--replace options.")

    exit_layer = semantic_stability_exit(
        base, min_logprob=args.router_min_logprob, patience=args.router_patience
    )
    if exit_layer is None:
        print("\n[router] No stable early-exit layer found under current thresholds.")
    else:
        total_layers = len(base)
        saved = ((total_layers - 1 - exit_layer) / max(1, total_layers - 1)) * 100
        print(
            f"\n[router] Candidate early-exit layer: {exit_layer} "
            f"(counterfactual saved compute ~= {saved:.1f}%)"
        )

    payload = {
        "config": {
            "model": args.model,
            "device": device,
            "target": args.target,
            "delete": args.delete,
            "replace": args.replace,
            "router_min_logprob": args.router_min_logprob,
            "router_patience": args.router_patience,
        },
        "prompt": args.prompt,
        "base": base,
        "variants": [{"name": rows[0]["case"], "rows": rows} for rows in other_rows],
        "router_exit_layer": exit_layer,
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
