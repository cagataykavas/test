"""
Struct-XAI Projesi - Katman Bazlı SHAP (Layer-wise Mechanistic SHAP) 📊✨
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Her bir girdi kelimesinin (token), modelin her bir derin katmanındaki (15-28)
hedef düşünceye (Karakter konseptine) olan katkısını Isı Haritası olarak çizmek!
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"
OUTPUT_DIR = "Struct_XAI_Visuals"  # Düzenli laboratuvar kuralı!

# Çıktı klasörünü oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_layer_probs(prompt, target_token_id, model, tokenizer, start_layer=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head

    probs_list = []
    # Sadece ilgilendiğimiz katmanları alalım
    for h_state in outputs.hidden_states[start_layer:]:
        token_hidden_state = h_state[0, -1, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs[target_token_id].item() * 100)

    return probs_list


def main():
    print("🪄 Alice'in Nihai SHAP Röntgeni Isınıyor...")
    print(f"Grafikler '{OUTPUT_DIR}' klasörüne kaydedilecek!\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    base_prompt = "Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: Adam sinirle masaya vurdu ve bağırdı.\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    target_word = "人物"
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]

    print("[1/3] Orijinal Zihin Haritası Çıkarılıyor...")
    base_probs = get_layer_probs(base_prompt, target_token_id, model, tokenizer)

    # SHAP için tek tek sileceğimiz kelimeler (Bağlamı en çok etkileyenler)
    words_to_ablate = ["Adam", "sinirle", "masaya", "vurdu", "ve", "bağırdı."]

    # Isı haritası için matris hazırlığı: Satırlar = Katmanlar, Sütunlar = Silinen Kelimeler
    # Değer = (Orijinal İhtimal - Silindiğindeki İhtimal). Yani "Etki Skoru (Impact)"
    impact_matrix = []

    print("[2/3] Kelime Ameliyatları (Ablasyon) Yapılıyor... (Biraz sürebilir)")
    for word in words_to_ablate:
        # Kelimeyi cümleden gizlice çıkar
        ablated_prompt = base_prompt.replace(f" {word}", "")
        if word == "Adam": ablated_prompt = base_prompt.replace("Adam ", "")  # Başlangıç kelimesi boşluk ayarı

        ablated_probs = get_layer_probs(ablated_prompt, target_token_id, model, tokenizer)

        # Etkiyi hesapla (Orijinal - Ablasyon)
        word_impact = [base - abl for base, abl in zip(base_probs, ablated_probs)]
        impact_matrix.append(word_impact)
        print(f" -> '{word}' kelimesinin eksikliği test edildi.")

    # Matrisi (Kelimeler x Katmanlar) formatından (Katmanlar x Kelimeler) formatına çevir
    impact_matrix_transposed = list(map(list, zip(*impact_matrix)))
    layers_labels = [f"Layer {i}" for i in range(15, 29)]

    print("\n[3/3] Sanat Eseri (Heatmap) Çiziliyor...")
    plt.figure(figsize=(10, 8))

    # Teze konulacak o muazzam görsel!
    sns.heatmap(impact_matrix_transposed,
                xticklabels=words_to_ablate,
                yticklabels=layers_labels,
                cmap="coolwarm", center=0, annot=True, fmt=".1f", linewidths=.5)

    plt.title("Struct-XAI: Katman Bazlı Kelime Etkisi (Layer-wise SHAP)", fontsize=14, fontweight='bold')
    plt.xlabel("Silinen Kelime (Ablasyon)", fontsize=12)
    plt.ylabel("Derin Katmanlar (Zihinsel Uyanış)", fontsize=12)
    plt.tight_layout()

    # Özel klasöre kaydet
    file_path = os.path.join(OUTPUT_DIR, "Struct_XAI_Layer_SHAP.png")
    plt.savefig(file_path, dpi=300)

    print("=" * 60)
    print(f"🎉 ZAFER! Röntgen filmi '{file_path}' konumuna kaydedildi!")
    print(
        "Hemen o klasöre git ve resmi aç. Hangi kelimenin hangi katmanda kıpkırmızı (yüksek etki) yandığına inanamayacaksın!")


if __name__ == "__main__":
    main()