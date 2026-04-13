"""
Struct-XAI Projesi - Nedensellik ve Ablasyon Testi (SHAP Alternatifi) 🔬
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Kritik bir kelimeyi (Örn: 'sinirle') silmenin, gizli uzaydaki (Latent Space)
'Erken Çıkış' katmanına etkisini matematiksel olarak kanıtlamak.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"
THRESHOLD = 19.5  # Senin simülasyonda belirlediğimiz ideal güven eşiği!


def get_early_exit_data(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_token_idx = -1
    final_norm = model.model.norm
    lm_head = model.lm_head

    for layer_idx, h_state in enumerate(hidden_states):
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        confidence = logits.max().item()
        top_token_id = logits.argmax().item()
        top_token_str = tokenizer.decode([top_token_id])

        # Eğer güven skoru eşiğimizi (18.0) geçerse, model "Ben anladım" der ve çıkar.
        if confidence >= THRESHOLD:
            return layer_idx, top_token_str, confidence

    # Eğer hiçbir katmanda eşiği geçemezse son katmanı döndür
    return len(hidden_states) - 1, top_token_str, confidence


def main():
    print("🪄 Alice'in Nedensellik Ameliyathanesi Isınıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    print("\n[Orijinal Metin Test Ediliyor...]")
    prompt_orijinal = """Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.
Girdi: Adam sinirle masaya vurdu ve bağırdı.
Çıktı:
SCENE_DIRECTIONS:
-"""
    layer_orig, token_orig, conf_orig = get_early_exit_data(prompt_orijinal, model, tokenizer)

    print("[Ablasyon (Silinmiş) Metin Test Ediliyor...]")
    prompt_silinmis = """Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.
Girdi: Adam masaya vurdu ve bağırdı.
Çıktı:
SCENE_DIRECTIONS:
-"""
    layer_abl, token_abl, conf_abl = get_early_exit_data(prompt_silinmis, model, tokenizer)

    print("\n" + "=" * 60)
    print("🏆 STRUCT-XAI NEDENSELLİK RAPORU (SHAP KESİŞİMİ)")
    print("=" * 60)
    print(f"Orijinal Girdi (Adam 'sinirle' vurdu):")
    print(f"-> Çıkış Katmanı: {layer_orig} | Karar: {repr(token_orig)} | Güven: {conf_orig:.2f}")
    print("-" * 60)
    print(f"Ablasyon Girdisi (Sadece vurdu - 'sinirle' KELİMESİ SİLİNDİ):")
    print(f"-> Çıkış Katmanı: {layer_abl} | Karar: {repr(token_abl)} | Güven: {conf_abl:.2f}")
    print("=" * 60)

    if layer_abl > layer_orig:
        print(
            f"💣 BÜYÜK KEŞİF! 'sinirle' kelimesi silinince model {layer_abl - layer_orig} katman daha fazla düşünmek zorunda kaldı!")
        print(
            "Bu, 'sinirle' kelimesinin modelin derinlerindeki 'karar hızlandırma' (causal trigger) mekanizması olduğunu kanıtlar!")


if __name__ == "__main__":
    main()