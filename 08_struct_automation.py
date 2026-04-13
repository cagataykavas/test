"""
Struct-XAI Projesi - Otomatize Edilmiş Çıkarım Bandı (Batch Evaluator) 🤖
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Farklı duyguların ve bağlamların (Öfke, Korku, Neşe) ablasyon testlerini
otomatik olarak çalıştırıp, XAI metriklerini karşılaştırmak.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Burası bizim Model Arenamız. İleride buraya "meta-llama/Meta-Llama-3-8B-Instruct" yazıp savaştıracağız!
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"
THRESHOLD = 18.0
STABILITY_REQ = 2

# Test edilecek senaryoların otomasyon listesi
TEST_CASES = [
    {"isim": "Öfke (Orijinal)", "metin": "Adam sinirle masaya vurdu ve bağırdı."},
    {"isim": "Öfke (Ablasyon)", "metin": "Adam masaya vurdu ve bağırdı."},
    {"isim": "Korku (Orijinal)", "metin": "Kadın dehşetle kapıya doğru adımladı."},
    {"isim": "Korku (Ablasyon)", "metin": "Kadın kapıya doğru adımladı."},
    {"isim": "Neşe (Orijinal)", "metin": "Çocuk kahkahalarla hediyesini açtı."},
    {"isim": "Neşe (Ablasyon)", "metin": "Çocuk hediyesini açtı."}
]


def run_struct_evaluator(prompt, model, tokenizer):
    """Router algoritmasının otomatize edilmiş sessiz versiyonu."""
    full_prompt = f"Sen bir tiyatro asistanısın. Sadece istenen başlığı çıkar.\nGirdi: {prompt}\nÇıktı:\nSCENE_DIRECTIONS:\n-"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    previous_thought = None
    stability_count = 0

    for layer_idx, h_state in enumerate(outputs.hidden_states):
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        confidence = logits.max().item()
        top_token_str = tokenizer.decode([logits.argmax().item()])

        if confidence >= THRESHOLD:
            if top_token_str == previous_thought:
                stability_count += 1
            else:
                stability_count = 1
        else:
            stability_count = 0

        if stability_count >= STABILITY_REQ:
            compute_saved = ((28 - layer_idx) / 28) * 100
            return layer_idx, top_token_str, compute_saved

        previous_thought = top_token_str

    return 28, top_token_str, 0.0


def main():
    print(f"🪄 Alice'in Otomasyon Fabrikası Çalışıyor...\nHedef Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    print("\n" + "=" * 85)
    print(f"{'SENARYO TÜRÜ':<20} | {'ÇIKIŞ KATM.':<12} | {'TASARRUF':<12} | {'SON KARAR / HALÜSİNASYON'}")
    print("=" * 85)

    for case in TEST_CASES:
        layer, thought, saved = run_struct_evaluator(case["metin"], model, tokenizer)
        clean_thought = repr(thought)
        print(f"{case['isim']:<20} | Layer {layer:<6} | %{saved:<10.1f} | {clean_thought}")

    print("=" * 85)
    print("💣 Tez Çıkarımı: Orijinal ile Ablasyon arasında Çıkış Katmanı veya Son Karar değişiyorsa,")
    print("o duygu kelimesi modelin zihninde bir 'Causal Trigger' (Nedensel Tetikleyici) görevi görüyordur!")


if __name__ == "__main__":
    main()