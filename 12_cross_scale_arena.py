"""
Struct-XAI Projesi - Çapraz Ölçek Arenası (Cross-Scale Analysis) 🐉🐣
Modeller: Qwen2.5-7B (Büyük) vs Qwen2.5-1.5B (Küçük)
Amaç: 'Anlamsal Çöküş' ve bağlam anlama kapasitesinin modelin parametre
boyutuyla (zeka seviyesiyle) olan ilişkisini kanıtlamak.
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"

# Kadim Ejderha ve Yavru Ejderha (1.5B saniyeler içinde inecektir!)
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct"
]


def get_ablation_confidence(model_id, prompt, target_token_str):
    print(f"\n[+] {model_id} laboratuvara çağrılıyor...")

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

    # Modelin son karar anındaki (son katman) zihinsel durumunu çekiyoruz
    h_state = outputs.hidden_states[-1]
    token_hidden_state = h_state[0, final_token_idx, :]
    normalized_state = final_norm(token_hidden_state)
    logits = lm_head(normalized_state)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Hedef kelimenin ihtimalini ölçüyoruz
    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    confidence = probs[target_token_id].item() * 100

    # Yavru ejderhanın asıl ne demek istediğini de (en yüksek ihtimal) yakalayalım
    top_token_str = tokenizer.decode([logits.argmax().item()])

    del model
    del tokenizer
    gc.collect()

    return confidence, top_token_str


def main():
    print("🪄 Alice'in Çapraz Ölçek Arenası Açılıyor... (7B vs 1.5B)")

    prompt_orig = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    prompt_abl = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"

    target = "人物"  # Karakter/Kişi konsepti

    print("\n" + "=" * 80)
    print(f"{'MODEL BOYUTU':<30} | {'ORİJİNAL (% ve Karar)':<20} | {'ABLASYON (% ve Karar)'}")
    print("=" * 80)

    for model_name in MODELS:
        # 1. Orijinal Güven
        conf_orig, top_orig = get_ablation_confidence(model_name, prompt_orig, target)

        # 2. Ablasyon
        conf_abl, top_abl = get_ablation_confidence(model_name, prompt_abl, target)

        diff = conf_orig - conf_abl

        print(f"\n{model_name:<30}")
        print(f"Orijinal: %{conf_orig:.2f} (Dediği: {repr(top_orig)})")
        print(f"Ablasyon: %{conf_abl:.2f} (Dediği: {repr(top_abl)})")
        print(f"-> Anlamsal Çöküş Farkı: %{diff:.2f}")

    print("\n" + "=" * 80)
    print("💣 TEZ ÇIKARIMI (Model Scaling Laws):")
    print("Küçük modellerin bağlamı tutma kapasitesi daha zayıf olduğu için,")
    print("cümleden duygu kelimelerini sildiğimizde tamamen farklı (ve saçma) hedeflere savrulabilirler!")


if __name__ == "__main__":
    main()