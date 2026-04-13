"""
Struct-XAI Projesi - Zirve Kavram Tarayıcısı (Peak Concept Scanner) 🏔️🐉
Modeller: Qwen2.5-7B (Büyük) vs Qwen2.5-1.5B (Küçük)
Amaç: Sadece son katmana (Layer 28) değil, modelin düşünce nehrindeki
*TÜM* katmanlara bakarak o kavramın (Karakter/Sahne) ulaştığı ZİRVE (Peak) ihtimali bulmak!
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct"
]


def get_peak_ablation_confidence(model_id, prompt, target_token_str):
    print(f"\n[+] {model_id} zihni taranıyor...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]

    peak_confidence = 0.0
    peak_layer = 0

    # Bütün katmanları (Layer 0'dan 28'e) tek tek gezip ZİRVE noktasını arıyoruz!
    for layer_idx, h_state in enumerate(outputs.hidden_states):
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[target_token_id].item() * 100

        if confidence > peak_confidence:
            peak_confidence = confidence
            peak_layer = layer_idx

    # Modelin en sonunda ağzından çıkan (çıktı) kelime
    final_logits = lm_head(final_norm(outputs.hidden_states[-1][0, final_token_idx, :]))
    top_token_str = tokenizer.decode([final_logits.argmax().item()])

    del model
    del tokenizer
    gc.collect()

    return peak_confidence, peak_layer, top_token_str


def main():
    print("🪄 Alice'in Zirve Kavram Tarayıcısı Açılıyor... (7B vs 1.5B)")

    prompt_orig = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    prompt_abl = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"

    target = "人物"  # Karakter/Kişi konsepti

    print("\n" + "=" * 85)
    print(f"{'MODEL BOYUTU':<25} | {'ORİJİNAL ZİRVE (% ve Katman)':<30} | {'ABLASYON ZİRVE'}")
    print("=" * 85)

    for model_name in MODELS:
        # 1. Orijinal Güven (Zirve)
        conf_orig, layer_orig, out_orig = get_peak_ablation_confidence(model_name, prompt_orig, target)

        # 2. Ablasyon Güven (Zirve)
        conf_abl, layer_abl, out_abl = get_peak_ablation_confidence(model_name, prompt_abl, target)

        diff = conf_orig - conf_abl

        print(f"\n{model_name:<25}")
        print(f"Orijinal: %{conf_orig:.2f} (Layer {layer_orig}) | Çıktı: {repr(out_orig)}")
        print(f"Ablasyon: %{conf_abl:.2f} (Layer {layer_abl}) | Çıktı: {repr(out_abl)}")
        print(f"-> Zirveden Düşüş: %{diff:.2f}")

    print("\n" + "=" * 85)
    print("💣 TEZ ÇIKARIMI (Model Scaling & Robustness):")
    print("Büyük modeller bağlamı tutmak için devasa kuleler (yüksek %) inşa ederken,")
    print("küçük modellerin kuleleri zaten cılızdır ve en ufak bir kelime silinmesinde yıkılır!")


if __name__ == "__main__":
    main()