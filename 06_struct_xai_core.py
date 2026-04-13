"""
Struct-XAI Projesi - Çekirdek Metodoloji (Layer-wise Causal Attribution) 🧠✨
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Sadece son kararı değil, modelin *içindeki* her bir katmanın (Layer)
kararlarının girdideki hangi kelimeden etkilendiğini (Ablasyon/SHAP ile) bulmak.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"


def get_layer_predictions(prompt, model, tokenizer):
    """
    Modelin her bir katmanında (layer), 'Sıradaki kelime ne?'
    sorusuna verdiği en yüksek ihtimalli cevabı ve skorunu alır.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_preds = []
    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    for h_state in outputs.hidden_states:
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        confidence = logits.max().item()
        top_token_id = logits.argmax().item()
        top_token_str = tokenizer.decode([top_token_id])
        layer_preds.append((top_token_str, confidence))

    return layer_preds


def main():
    print("🪄 Struct-XAI: Nihai Metodoloji Başlatılıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    base_prompt = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"

    # 1. ORİJİNAL DÜŞÜNCEYİ AL
    print("\n[Orijinal Düşünce Nehri Haritalandırılıyor...]")
    base_preds = get_layer_predictions(base_prompt, model, tokenizer)

    # 2. ABLASYON HEDEFLERİ (Hangi kelimelerin önemini ölçeceğiz?)
    # "sinirle" kelimesini sildiğimizde Latent Space nasıl sapıyor?
    test_cases = [
        ("sinirle",
         "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"),
        ("masaya",
         "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-")
    ]

    print("\n📊 STRUCT-XAI: KATMAN BAZLI ETKİ ANALİZİ")
    print("=" * 75)
    print(f"Hedef Katman | Orijinal Karar | 'sinirle' silindiğinde | 'masaya' silindiğinde")
    print("-" * 75)

    # Sadece kritik "Aydınlanma" katmanlarına (15'ten 28'e kadar) bakalım
    # Çünkü Slayt 10 "Pivot Hypothesis" bize asıl aksiyonun derin katmanlarda koptuğunu söylüyor!
    start_layer = 15
    end_layer = len(base_preds)

    ablated_preds = {}
    for word, prompt in test_cases:
        ablated_preds[word] = get_layer_predictions(prompt, model, tokenizer)

    for i in range(start_layer, end_layer):
        orig_word, orig_conf = base_preds[i]

        sinir_word, sinir_conf = ablated_preds["sinirle"][i]
        masa_word, masa_conf = ablated_preds["masaya"][i]

        # Formatı düzenle
        orig_str = f"{repr(orig_word)} ({orig_conf:.1f})"
        sinir_str = f"{repr(sinir_word)} ({sinir_conf:.1f})"
        masa_str = f"{repr(masa_word)} ({masa_conf:.1f})"

        print(f"Layer {i:<6} | {orig_str:<14} | {sinir_str:<22} | {masa_str}")

    print("=" * 75)
    print("\n💣 ALICE'İN YORUMU:")
    print("Eğer bir katmanda 'sinirle' sütunundaki kelime, Orijinal Karardan farklıysa,")
    print("bu demektir ki 'sinirle' kelimesi o katmanın düşüncesini %100 kontrol ediyordu!")


if __name__ == "__main__":
    main()