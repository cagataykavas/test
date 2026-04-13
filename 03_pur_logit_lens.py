"""
Struct-XAI Projesi - Saf Logit Lens (CPU Modu - Sıfır İndirme)
Model: Qwen2.5-7B-Instruct
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu" # EKRAN KARTINI İPTAL ETTİK, İŞLEMCİ KULLANILACAK

def main():
    print("Saf model CPU'ya yükleniyor (Hiçbir şey indirilmeyecek)...")

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

    print("\nSorgu CPU üzerinden gönderiliyor, katmanlar dinleniyor...")
    print("(İşlemci kullanıldığı için yanıt vermesi 15-20 saniye sürebilir, lütfen bekle)\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_token_index = -1

    print("📊 SAF LOGIT LENS (CPU MODU)\n")
    print(f"{'Katman':<8} | {'Tahmin Edilen Kelime':<25} | {'Eminlik Skoru'}")
    print("-" * 60)

    final_norm = model.model.norm
    lm_head = model.lm_head

    for layer_idx, h_state in enumerate(hidden_states):
        token_hidden_state = h_state[0, final_token_index, :]

        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)

        top_token_id = logits.argmax().item()
        top_token_str = tokenizer.decode([top_token_id])
        confidence = logits.max().item()

        clean_str = repr(top_token_str)
        print(f"Layer {layer_idx:<2} | {clean_str:<25} | {confidence:.2f}")

if __name__ == "__main__":
    main()