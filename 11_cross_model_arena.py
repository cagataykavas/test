"""
Struct-XAI Projesi - Çapraz Mimari Arenası (Cross-Model Robustness) ⚔️
Modeller: Qwen2.5-7B vs Meta-Llama-3-8B
Amaç: 'Anlamsal Çöküş' (Semantic Collapse) zafiyetinin sadece bir modele mi özgü olduğunu,
yoksa tüm Transformatör mimarilerinin ortak bir zayıflığı mı olduğunu kanıtlamak.
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"

# Test edeceğimiz titanlar!
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2" # Llama'yı kovduk, yerine Özgür Fransız Titanını getirdik!
]


def get_ablation_confidence(model_id, prompt, target_token_str):
    print(f"\n[+] {model_id} laboratuvara indiriliyor/yükleniyor... (Bu biraz sürebilir)")

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

    # Modelin son katmanındaki (karar anındaki) ihtimallere bakıyoruz
    h_state = outputs.hidden_states[-1]
    token_hidden_state = h_state[0, final_token_idx, :]
    normalized_state = final_norm(token_hidden_state)
    logits = lm_head(normalized_state)

    # Hedef kelimenin ihtimalini (Softmax) hesapla
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Llama ve Qwen'in token sözlükleri farklıdır, bu yüzden kelimeyi token'a çevirip arıyoruz
    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    confidence = probs[target_token_id].item() * 100

    # Hafızayı temizle ki diğer modele yer açılsın!
    del model
    del tokenizer
    gc.collect()

    return confidence


def main():
    print("🪄 Alice'in Çapraz Mimari Arenası Açılıyor...")

    prompt_orig = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    prompt_abl = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"

    # Son katmanda modelin çıktı vermeye hazırlandığı "boşluk" veya hedef karakterler
    targets = {
        "Qwen/Qwen2.5-7B-Instruct": " A",
        "mistralai/Mistral-7B-Instruct-v0.2": " ("
    }

    print("\n" + "=" * 70)
    print(f"{'MODEL MİMARİSİ':<40} | {'ORİJİNAL GÜVEN':<15} | {'ABLASYON ÇÖKÜŞÜ'}")
    print("=" * 70)

    for model_name in MODELS:
        target = targets[model_name]

        # 1. Orijinal Güven
        conf_orig = get_ablation_confidence(model_name, prompt_orig, target)

        # 2. Ablasyon (Sinirle kelimesi silinince)
        conf_abl = get_ablation_confidence(model_name, prompt_abl, target)

        diff = conf_orig - conf_abl

        print(f"\n{model_name:<40}")
        print(f"-> Hedef: '{target}' | Orijinal: %{conf_orig:.2f} | Ablasyon: %{conf_abl:.2f} | Çöküş: %{diff:.2f}")

    print("\n" + "=" * 70)
    print("💣 TEZ ÇIKARIMI (Cross-Architecture Validation):")
    print("Eğer Llama-3 de Qwen gibi büyük bir 'Çöküş' yaşıyorsa, Struct-XAI'ın")
    print("keşfettiği bu zafiyet evrenseldir ve tüm yapay zeka dünyasını ilgilendirir!")


if __name__ == "__main__":
    main()