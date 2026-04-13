"""
Struct-XAI Projesi - Aşama 2: Logit Lens (Zihin Okuma) 🧠
Model: Qwen2.5-7B-Instruct (TransformerLens ile)
Amaç: Modelin "SCENE_DIRECTIONS" üretirken katman katman ne düşündüğünü görmek.
"""

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("🪄 Alice'in Ameliyathanesi Açılıyor...")
    print(f"🚀 Model TransformerLens içine yükleniyor (Bu biraz VRAM yiyebilir, 5090'a güveniyoruz!)...")

    # Modeli HookedTransformer olarak yüklüyoruz. Bu bize her nörona erişim verecek!
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.bfloat16  # 5090 için altın standart
    )

    print("\n✅ Model başarıyla kancalandı! Zihin okuma başlıyor...\n")

    # Basit ama etkili bir prompt seçiyoruz.
    # Modelin tam olarak "Gergin" (veya benzeri bir durum) üretmesini beklediğimiz o sınır anı!
    prompt = """Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.
Girdi: Adam sinirle masaya vurdu ve bağırdı.
Çıktı:
SCENE_DIRECTIONS:
-"""

    print(f"Sorgu (Prompt):\n{prompt}\n")
    print("-" * 50)

    # 1. RUN WITH CACHE: Modelden cevabı üretmesini isterken aynı zamanda
    # tüm katmanlardaki (layer) düşünce nehrini (residual stream) 'cache' içine hapsediyoruz!
    logits, cache = model.run_with_cache(prompt)

    # Promptun son tokeninin (yani "-" işaretinin olduğu anın) indeksini buluyoruz.
    # Biz tam bu anda modelin "Sıradaki kelime ne?" diye düşünmesini izleyeceğiz.
    final_token_index = -1

    print("📊 KATMAN KATMAN DÜŞÜNCE OKUMA (LOGIT LENS)\n")
    print(f"{'Katman':<10} | {'Modelin Tahmini (En yüksek ihtimalli kelime)':<40} | {'Eminlik Skoru'}")
    print("-" * 75)

    # 2. LOGIT LENS UYGULAMASI
    # Modelin her bir katmanında (Qwen'de 28 katman vardır) geziyoruz...
    for layer in range(model.cfg.n_layers):
        # O katmandaki bilgiyi çek
        resid_post = cache[f"blocks.{layer}.hook_resid_post"][0, final_token_index, :]

        # O katmanda ameliyatı erken bitirip, "Hemen şimdi cevap ver!" diyerek unembedding yapıyoruz
        scaled_resid = model.ln_final(resid_post)
        layer_logits = model.unembed(scaled_resid)

        # O katmandaki en yüksek ihtimalli kelimeyi bul
        top_token_id = layer_logits.argmax().item()
        top_token_str = model.to_string(top_token_id)
        confidence = layer_logits.max().item()

        # Sonuçları ekrana bas (Renkli ve havalı görünmesi için biraz hizalama)
        # "\n" gibi kaçış karakterlerini temizle ki ekran bozulmasın
        clean_str = repr(top_token_str)
        print(f"Layer {layer:<4} | {clean_str:<40} | {confidence:.2f}")

    print("-" * 75)
    print("💣 Alice'in Teşhisi: İlk katmanlarda anlamsız kelimeler göreceksin. Orta katmanlarda")
    print("dilbilgisi oturacak, son katmanlarda ise bağlamı anlayıp 'Gergin' veya 'Oda' diyecek!")


if __name__ == "__main__":
    main()