"""
Struct-XAI - Basit Semantic Stability Router (MVP)

Not: Bu sürüm gerçek compute skip yapmaz; katman içi kararlılık sinyali üretir.
MVP hedefi: yarınki mock sunum için tekrarlanabilir ölçüm tablosu çıkarmak.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "sshleifer/tiny-gpt2"
THRESHOLD = 0.20
STABILITY_REQ = 2


def _final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def run_struct_xai_router(prompt: str, model, tokenizer, threshold: float, stability_req: int, device: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    previous_thought = None
    stability_count = 0
    total_layers = len(outputs.hidden_states)

    print(f"{'Katman':<8} | {'Düşünce':<20} | {'P(max)':<8} | {'Kararlılık'}")
    print("-" * 62)

    for layer_idx, h_state in enumerate(outputs.hidden_states):
        token_hidden_state = h_state[0, -1, :]
        normalized_state = _final_norm(model, token_hidden_state)
        logits = model.lm_head(normalized_state)
        probs = torch.softmax(logits, dim=-1)

        confidence = float(probs.max().item())
        top_token_id = int(probs.argmax().item())
        current_thought = tokenizer.decode([top_token_id])

        if confidence >= threshold:
            if current_thought == previous_thought:
                stability_count += 1
                status = f"✅ {stability_count}/{stability_req}"
            else:
                stability_count = 1
                status = "⚠️ yeni fikir"
        else:
            stability_count = 0
            status = "❌ güven düşük"

        print(f"Layer {layer_idx:<2} | {repr(current_thought):<20} | {confidence:<8.3f} | {status}")

        if stability_count >= stability_req:
            compute_saved = ((total_layers - (layer_idx + 1)) / total_layers) * 100
            print("=" * 62)
            print(f"🎯 Erken çıkış adayı: layer={layer_idx}, tahmini tasarruf=%{compute_saved:.1f}")
            print("=" * 62)
            return layer_idx, current_thought, compute_saved

        previous_thought = current_thought

    return total_layers - 1, current_thought, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--stability-req", type=int, default=STABILITY_REQ)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    prompt = (
        "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\n"
        "Girdi: Adam sinirle masaya vurdu ve bağırdı.\n"
        "Çıktı:\nSCENE_DIRECTIONS:\n-"
    )
    run_struct_xai_router(prompt, model, tokenizer, args.threshold, args.stability_req, args.device)


if __name__ == "__main__":
    main()
