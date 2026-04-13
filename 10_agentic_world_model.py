"""
Struct-XAI Projesi - Agentic Interpretability (Dünya Modeli Takibi) 🌍✈️
Model: Qwen2.5-7B-Instruct (CPU Modu)
Amaç: Dinamik bir Çoklu-Ajan (Multi-Agent) uçuş formasyonu senaryosunda,
modelin "Lider", "Hedef" ve "Manevra" kavramlarını gizli uzayda nasıl modellediğini izlemek.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cpu"


def track_world_model_state(prompt, state_tokens, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_token_idx = -1

    state_tracking = {token: [] for token in state_tokens}
    token_ids = {token: tokenizer.encode(token, add_special_tokens=False)[0] for token in state_tokens}

    for h_state in outputs.hidden_states:
        token_hidden_state = h_state[0, final_token_idx, :]
        normalized_state = final_norm(token_hidden_state)
        logits = lm_head(normalized_state)
        probs = F.softmax(logits, dim=-1)

        for token, t_id in token_ids.items():
            state_tracking[token].append(probs[t_id].item() * 100)

    return state_tracking


def main():
    print("🪄 Alice'in Ajan Radarı (World Model Tracker) Açılıyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Çoklu-Ajan (Multi-Agent) Karar Senaryosu
    agentic_prompt = """Durum Raporu:
Ajan 1 (Lider İHA) hedef koordinata kilitlendi ve ilerliyor. 
Ajan 2 (Kanat İHA) formasyonu koruyarak Lideri takip ediyor. 
Acil Durum: Hedef aniden sert bir manevrayla sola saptı.
Görev: Ajan 2'nin (Kanat) formasyonu bozmamak için yapması gereken yeni eylem:
-"""

    # Modelin zihninde aynı anda takip etmesi gereken 3 farklı "Ajan/Çevre" durumu
    state_concepts = [" Lider", " sola", " takip"]

    print("\n✈️ Senaryo: Çoklu İHA Formasyonu ve Hedef Sapması")
    print("Modelin zihnindeki 'Dünya Modeli' değişkenleri taranıyor...\n")

    tracking_data = track_world_model_state(agentic_prompt, state_concepts, model, tokenizer)

    print("=" * 75)
    print(f"{'Katman':<8} | {'Durum 1: LİDER (%)':<20} | {'Durum 2: SOLA (%)':<20} | {'Eylem: TAKİP (%)'}")
    print("=" * 75)

    for i in range(15, len(tracking_data[state_concepts[0]])):
        lider_prob = tracking_data[" Lider"][i]
        sola_prob = tracking_data[" sola"][i]
        takip_prob = tracking_data[" takip"][i]

        print(f"Layer {i:<2} | %{lider_prob:<18.3f} | %{sola_prob:<18.3f} | %{takip_prob:.3f}")

    print("=" * 75)
    print("💣 TEZ ÇIKARIMI (Temporal Credit Assignment):")
    print("Ajan kararı verirken (Takip), geçmişteki olayların (Liderin durumu ve Hedefin sola sapması)")
    print("gizli uzayda nasıl birbiriyle rekabet ettiğini / güncellendiğini kanıtlıyoruz!")


if __name__ == "__main__":
    main()