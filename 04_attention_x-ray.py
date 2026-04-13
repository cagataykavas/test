"""
Struct-XAI Projesi - Dikkat Haritası (Attention Heatmap) 👁️
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Modelin karar anında hangi kelimelere odaklandığını (Attention) görselleştirmek.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")  # Sıkıcı uyarıları susturalım

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"  # Güvenli limanımız!


def main():
    print("🪄 Alice'in Röntgen Cihazı Isınıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # <-- İŞTE SİHİRLİ KELİME BURADA!
    )

    prompt = """Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.
Girdi: Adam sinirle masaya vurdu ve bağırdı.
Çıktı:
SCENE_DIRECTIONS:
-"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Tokenleri makalede okunabilir yapmak için tek tek kelimeye çeviriyoruz
    clean_tokens = [tokenizer.decode([t_id]) for t_id in inputs["input_ids"][0]]

    print("\nSorgu işlemci üzerinden akıyor, modelin gözbebekleri izleniyor...")
    print("(Yaklaşık 15 saniye sürecek, çayından bir yudum al...)\n")

    with torch.no_grad():
        # output_attentions=True: İşte asıl sihir burada! Modelin nereye baktığını çalıyoruz.
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions tüm katmanların dikkatini içerir (Qwen'de 28 katman).
    # Biz o "Aydınlanma" anının yaşandığı son katmana ([-1]) bakacağız.
    last_layer_attention = outputs.attentions[-1]

    # Tüm dikkat kafalarının (attention heads) ortalamasını alıp 2 Boyutlu bir matris yapıyoruz
    avg_attention = last_layer_attention[0].mean(dim=0).cpu().float().numpy()

    print("📊 DİKKAT MATRİSİ ÇIKARTILDI! Isı haritası çiziliyor...")

    # Teze koymalık, jüri etkileyecek profesyonel bir Isı Haritası (Heatmap)
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_attention, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="magma", linewidths=0.5)

    plt.title("Qwen2.5-7B - Son Katman Dikkat Haritası (Attention Map)", fontsize=16, fontweight='bold')
    plt.xlabel("Hangi kelimeye bakılıyor? (Gözlenen)", fontsize=12)
    plt.ylabel("Hangi kelime bakıyor? (Gözleyen)", fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # Resmi klasöre kaydet
    file_name = "Struct_XAI_Attention_Map.png"
    plt.savefig(file_name, dpi=300)
    print(f"\n🎉 ZAFER! Röntgen filmi '{file_name}' adıyla proje klasörüne kaydedildi.")
    print("Hemen o resmi aç! En alt satıra (son tokene) bak ve hangi kutucukların daha parlak yandığını gör!")


if __name__ == "__main__":
    main()