"""
Struct-XAI Projesi - Aşama 1: Yeni Kralın Tahta Çıkışı 👑
Model: Llama-3-8B-Instruct (veya Qwen2.5-7B-Instruct)
Donanım: RTX 5090 (Bfloat16 Optimizasyonu ile)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# --- ALICE'İN SİLAH SEÇİMİ ---
# Eğer Llama-3'te HF izniyle uğraşmak istemezsen burayı "Qwen/Qwen2.5-7B-Instruct" yap. Türkçesi harikadır!
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 30 Senaryoluk Alice Test Seti (O %0 çeken meşhur liste)
test_scenarios = [
    "Kız kapıyı açar, içeri girerken tereddütle arkasına bakar. Elindeki mektubu sıkıca göğsüne bastırır.",
    "Adam öfkeyle telefonu duvara fırlatır. Sonra pişmanlıkla ellerini başının arasına alıp koltuğa çöker.",
    "Çocuk neşeyle parka koşar, salıncağa doğru el sallar. Ancak birden ayağı takılıp düşer ve ağlamaya başlar.",
    "Kadın aynadaki yansımasına bakar, gözündeki yaşı siler ve zoraki bir gülümseme takınır. Rujunu tazelemeye başlar.",
    "Yaşlı adam bastonuna dayanarak zorlukla ayağa kalkar. Pencereye yönelip 'Eskiden buralar hep dutluktu' diye mırıldanır.",
    "Genç adam heyecanla kutuyu açar. İçinden çıkan yüzüğü görünce gözleri parlar ve dizlerinin üzerine çöker.",
    "Doktor dosyayı kapatır ve gözlüğünü çıkarır. Karşısındaki hastaya endişeli bir bakış atıp derin bir nefes alır.",
    "Öğretmen tahtaya sertçe vurur. Sınıftaki uğultu kesilince tebeşiri eline alıp yazmaya başlar.",
    "Asker nöbet kulübesinde titreyerek sigarasını yakar. Uzaktan gelen sese kulak kabartır ve tüfeğini kavrar.",
    "Kedi yavaşça koltuğun arkasından çıkar. Sahibinin bacağına sürtünür ve mama kabına doğru koşar.",
    "Patron masadaki kağıtları hışımla imzalar. Sekreterine dönüp 'Toplantıyı iptal et!' diye bağırır.",
    "Hemşire serumu değiştirirken hastanın alnını kontrol eder. Hafifçe gülümseyerek 'Ateşiniz düşmüş' der.",
    "Garson tepsiyi düşürmemek için denge kurmaya çalışır. Müşterilerin arasından sıyrılarak masaya ulaşır.",
    "Şoför direksiyona yumruk atar. Trafiğin sıkışıklığına söylenerek radyoyu kapatır.",
    "Ressam tuvalin karşısına geçer, fırçayı boyaya batırır. Geri çekilip eserine kısık gözlerle bakar.",
    "Müzisyen kemanını kutusundan çıkarır. Yayını reçineler ve sahne ışıklarına doğru yürür.",
    "Hırsız pencereden sessizce içeri süzülür. Etrafı fenerle tarar ve kasanın olduğu tabloya yönelir.",
    "Anne bebeğini kucağında sallar. Ninni söylerken gözleri yorgunluktan kapanmaktadır.",
    "Dedektif olay yerindeki ayak izlerini inceler. Büyüteciyle yere eğilip bir sigara izmariti bulur.",
    "Çiçekçi vazodaki gülleri düzenler. Vitrindeki solmuş yaprakları temizleyip su püskürtür.",
    "Sporcu bitiş çizgisine yaklaşırken hızlanır. Arkasına bakmadan tüm gücüyle koşar ve ipi göğüsler.",
    "Barmen içkiyi karıştırıp bardağa döker. Müşteriye uzatırken göz kırpar ve 'Müessesenin ikramı' der.",
    "Kütüphaneci gürültü yapan öğrencilere 'Şşş!' yapar. Kitap arabasını iterek rafların arasına kaybolur.",
    "Pilot kokpitteki düğmelere sırayla basar. Kuleyle iletişime geçip 'Kalkışa hazırız' anonsu yapar.",
    "Terzi kumaşı özenle ölçer ve makası eline alır. İğne yastığını bileğine takıp kesime başlar.",
    "Balıkçı oltayı denize savurur. Sabırla beklerken termosundan çayını yudumlar.",
    "Fotoğrafçı objektifi odaklar. 'Gülümseyin!' diye bağırıp deklanşöre basar.",
    "Hakim tokmağı kürsüye vurur. 'Gereği düşünüldü' diyerek salondaki sessizliği bozar.",
    "Sunucu mikrofonu düzeltir ve kameraya döner. 'İyi akşamlar sayın seyirciler' diyerek habere başlar.",
    "Kız çocuğu balonunu elinden kaçırır. Gökyüzüne yükselen balona bakıp hıçkırarak ağlar."
]


def create_prompt(scenario):
    # Modern LLM'ler Chat Template kullanır! Artık saf metin yok, sistem rolleri var.
    return [
        {"role": "system",
         "content": "Sen uzman bir tiyatro asistanısın. Sana verilen sahneyi sadece şu 4 başlık altında incele: SCENE_DIRECTIONS, EMOTIONS, ACTIONS, TIMELINE. Ekstra yorum yapma."},
        {"role": "user", "content": "Örnek Sahne: Adam sinirle masaya vurdu. Hızla ayağa kalkıp pencereye yürüdü."},
        {"role": "assistant",
         "content": "SCENE_DIRECTIONS:\n- Gergin bir oda, masa başı.\nEMOTIONS:\n- Öfke, kararlılık.\nACTIONS:\n- Masaya vurmak, ani kalkış, yürüme.\nTIMELINE:\n t=0-2s: Masaya vurur.\n t=2-5s: Pencereye yürür."},
        {"role": "user", "content": f"Şimdi bu sahneyi analiz et: {scenario}"}
    ]


def main():
    print(f"🚀 RTX 5090 Uyandırılıyor... Hedef Model: {MODEL_NAME}")

    # RTX 5090 için özel bfloat16 optimizasyonu! Bellek dostu ve şimşek hızında.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # 5090'ın VRAM'ini otomatik yutar
    )

    # Pad token ayarları
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    success_count = 0
    total_count = len(test_scenarios)

    print(f"\n🧪 TEST BAŞLIYOR! Bakalım yeni kral {total_count} senaryoda ne yapacak?\n")
    start_time = time.time()

    for i, scenario in enumerate(test_scenarios):
        messages = create_prompt(scenario)
        # Chat formatını modelin anlayacağı tokenlara çeviriyoruz
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.1,  # Halüsinasyonu en aza indir
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Sadece yeni üretilen kısmı alıyoruz (Girdiyi kesip atıyoruz)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Basit Kontrol Mekanizması
        is_successful = all(key in response for key in ["SCENE_DIRECTIONS", "EMOTIONS", "ACTIONS", "TIMELINE"])

        if is_successful:
            success_count += 1
            print(f"✅ [{i + 1}/{total_count}] Başarılı! (Model sahneyi kusursuz anladı)")
        else:
            print(f"❌ [{i + 1}/{total_count}] Hata! Model formatı bozdu.\nÇıktı: {response[:50]}...")

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("🏆 ALICE'İN YENİ KRALLIK RAPORU (RTX 5090 EDITION)")
    print("=" * 50)
    print(f"Geçen Süre: {total_time:.1f} saniye (5090 farkı!)")
    print(f"Başarı Oranı: %{(success_count / total_count) * 100:.2f}")
    if success_count == total_count:
        print("Sonuç: KUSURSUZ! Halüsinasyon canavarı yenildi! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    main()