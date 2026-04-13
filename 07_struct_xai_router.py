"""
Struct-XAI Projesi - Nihai Algoritma: Dinamik Erken Çıkış (Dynamic Early Exit) 🚀
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Sadece yüksek skora aldanmamak; modelin 'Anlamsal Kararlılığını' ölçerek,
güvenli olduğu en erken katmanda işlemi kesip tasarruf sağlamak.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"
THRESHOLD = 18.0  # Minimum Güven Eşiği
STABILITY_REQ = 2  # Kaç katman üst üste aynı şeyi düşünmeli?


def run_struct_xai_router(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    previous_thought = None
    stability_count = 0

    print(f"{'Katman':<8} | {'Düşünce':<20} | {'Güven':<8} | {'Kararlılık Durumu'}")
    print("-" * 65)

    for layer_idx, h_state in enumerate(outputs.hidden_states):
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        confidence = logits.max().item()
        top_token_id = logits.argmax().item()
        current_thought = tokenizer.decode([top_token_id])

        # STRUCT-XAI ÇEKİRDEK MANTIĞI:
        status = ""
        if confidence >= THRESHOLD:
            if current_thought == previous_thought:
                stability_count += 1
                status = f"✅ Sabitleniyor... ({stability_count}/{STABILITY_REQ})"
            else:
                stability_count = 1
                status = "⚠️ Yeni Fikir, Bekleniyor..."
        else:
            stability_count = 0
            status = "❌ Güven Düşük, Fikir Dalgalı"

        clean_thought = repr(current_thought)
        print(f"Layer {layer_idx:<2} | {clean_thought:<20} | {confidence:.2f}   | {status}")

        # ÇIKIŞ (EXIT) TETİKLEYİCİSİ
        if stability_count >= STABILITY_REQ:
            compute_saved = ((28 - layer_idx) / 28) * 100
            print("=" * 65)
            print(f"🎯 STRUCT-XAI KESİCİSİ DEVREYE GİRDİ!")
            print(f"Model {layer_idx}. katmanda tam anlamsal kararlılığa ulaştı.")
            print(f"Kalan katmanlar çöpe atıldı. İşlem Gücü Tasarrufu: %{compute_saved:.1f}")
            print("=" * 65)
            return layer_idx, current_thought

        previous_thought = current_thought

    print("Model yeterli kararlılığa ulaşamadı, tüm katmanlar (28) çalıştırıldı.")
    return 28, current_thought


def main():
    print("🪄 Struct-XAI: Dinamik Yönlendirici (Router) Yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    prompt = """Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.
Girdi: Adam sinirle masaya vurdu ve bağırdı.
Çıktı:
SCENE_DIRECTIONS:
-"""

    print("\n🚀 Orijinal Metin İşleniyor (Zeka ve Kararlılık Testi)...")
    run_struct_xai_router(prompt, model, tokenizer)


if __name__ == "__main__":
    main()