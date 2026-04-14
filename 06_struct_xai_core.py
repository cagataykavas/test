"""
Struct-XAI - Çekirdek Katman Analizi (MVP)

Bu sürüm demo/makale hazırlığı için "çalışır" olacak şekilde sadeleştirildi:
- Varsayılan model hafif (sshleifer/tiny-gpt2)
- Katman sayısı modelden dinamik okunur
- Güven skoru ham logit değil, olasılık (softmax)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_PROMPT = (
    "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\n"
    "Girdi: Adam sinirle masaya vurdu ve bağırdı.\n"
    "Çıktı:\nSCENE_DIRECTIONS:\n-"
)


@dataclass
class LayerPrediction:
    token: str
    prob: float


def _final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    """Farklı mimarilerde final norm katmanını güvenli şekilde uygular."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def get_layer_predictions(prompt: str, model, tokenizer, device: str) -> list[LayerPrediction]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_preds: list[LayerPrediction] = []
    final_token_idx = -1

    for h_state in outputs.hidden_states:
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = _final_norm(model, token_hidden_state)
        logits = model.lm_head(normalized_state)
        probs = torch.softmax(logits, dim=-1)

        top_token_id = probs.argmax().item()
        top_token_str = tokenizer.decode([top_token_id])
        layer_preds.append(LayerPrediction(token=top_token_str, prob=float(probs[top_token_id].item())))

    return layer_preds


def run_analysis(model_name: str, device: str) -> None:
    print(f"🪄 Struct-XAI Core | model={model_name} | device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    base_prompt = DEFAULT_PROMPT
    test_cases = [
        (
            "sinirle",
            "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\n"
            "Girdi: Adam masaya vurdu ve bağırdı.\n"
            "Çıktı:\nSCENE_DIRECTIONS:\n-",
        ),
        (
            "masaya",
            "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\n"
            "Girdi: Adam sinirle vurdu ve bağırdı.\n"
            "Çıktı:\nSCENE_DIRECTIONS:\n-",
        ),
    ]

    base_preds = get_layer_predictions(base_prompt, model, tokenizer, device)
    ablated_preds = {word: get_layer_predictions(prompt, model, tokenizer, device) for word, prompt in test_cases}

    start_layer = max(0, len(base_preds) // 2)
    print("\n📊 KATMAN BAZLI ETKİ ANALİZİ")
    print("=" * 96)
    print(f"{'Katman':<8} | {'Orijinal':<25} | {'sinirle silinince':<25} | {'masaya silinince':<25}")
    print("-" * 96)

    for i in range(start_layer, len(base_preds)):
        orig = base_preds[i]
        sinir = ablated_preds["sinirle"][i]
        masa = ablated_preds["masaya"][i]
        print(
            f"{i:<8} | {repr(orig.token)} ({orig.prob:.3f})".ljust(35)
            + f"| {repr(sinir.token)} ({sinir.prob:.3f})".ljust(35)
            + f"| {repr(masa.token)} ({masa.prob:.3f})"
        )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_analysis(args.model, args.device)


if __name__ == "__main__":
    main()
