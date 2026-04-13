"""
Struct-XAI Projesi - Hedef Kelime İhtimal Takibi (Target Trajectory) 🎯
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Modelin saçmaladığı (argmax) kelimelere değil, doğrudan bizim istediğimiz
(örneğin bağlamı temsil eden) hedefin zihinsel ihtimal yüzdesine bakmak!
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"


def get_target_probabilities(prompt, target_token_id, model, tokenizer):
    """Modelin her katmanında spesifik bir kelimeye (hedefe) % kaç ihtimal verdiğini ölçer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    layer_probs = []

    for h_state in outputs.hidden_states:
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        # Softmax ile logitleri gerçek Yüzdelik (%) ihtimallere çeviriyoruz!
        probs = F.softmax(logits, dim=-1)

        # Sadece bizim aradığımız o altın hedefin ihtimalini çekiyoruz
        target_prob = probs[target_token_id].item() * 100
        layer_probs.append(target_prob)

    return layer_probs


def main():
    print("🪄 Alice'in Olasılık Mikroskobu Açılıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    prompt_orijinal = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    prompt_ablasyon = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"

    # Hedefimiz: Tiyatro/Sahne konseptini anladığını gösteren o meşhur kelime: '人物' (Kişi/Karakter)
    # Modelin kafasındaki bu fikrin tohumu nasıl büyüyor izleyeceğiz.
    target_word = "人物"
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]

    print(f"\n🎯 Hedef Kavram Belirlendi: '{target_word}' (Token ID: {target_token_id})")
    print("Bu kelimenin katmanlar arası yüzdelik (%) hayatta kalma savaşını izliyoruz...\n")

    probs_orig = get_target_probabilities(prompt_orijinal, target_token_id, model, tokenizer)
    probs_abl = get_target_probabilities(prompt_ablasyon, target_token_id, model, tokenizer)

    print("=" * 60)
    print(f"{'Katman':<8} | {'Orijinal (%)':<15} | {'Ablasyon (%) [-sinirle]':<20}")
    print("=" * 60)

    # Sadece 15'ten 28'e kadar olan kritik zihinsel bölgeye bakalım
    for i in range(15, len(probs_orig)):
        o_prob = probs_orig[i]
        a_prob = probs_abl[i]

        # Farkı vurgulamak için görsel bir bar çizelim
        diff = o_prob - a_prob
        trend = "📉 DÜŞÜŞ!" if diff > 0 else "📈 Yükseliş"
        if abs(diff) < 0.1: trend = "➖ Sabit"

        print(f"Layer {i:<2} | %{o_prob:<13.3f} | %{a_prob:<13.3f} | {trend} (Fark: %{diff:.2f})")

    print("=" * 60)
    print("💣 TEZ ÇIKARIMI:")
    print("Eğer Ablasyon tarafında hedefin ihtimali düşüyorsa, 'sinirle' kelimesi")
    print("modelin doğru cevaba ulaşmasını sağlayan bir 'Köprü' görevi görüyordur!")


if __name__ == "__main__":
    main()