# Struct-XAI MVP (Mock Paper Ready)

Bu repo artık **yarına kadar gösterilebilir bir MVP akışı** içeriyor.

## Dürüst durum özeti
- Bu çalışma **AI slop değil**, ama önceki haliyle bilimsel iddialar için zayıftı (özellikle sabit 28 katman, ham logit skoru ve ağır model bağımlılığı).
- Bu güncelleme, sistemi "çalışır + tekrarlanabilir" hale getirir: hafif modelle koşan analiz, katman bazlı ablation karşılaştırması ve router çıktısı.
- Henüz tam "causal proof" değil; ama mock bildiri/sunum için sağlam bir "baseline + next steps" sağlar.

## Hızlı başlangıç
```bash
python 06_struct_xai_core.py --model sshleifer/tiny-gpt2 --device cpu
python 07_struct_xai_router.py --model sshleifer/tiny-gpt2 --device cpu --threshold 0.20 --stability-req 2
```

## Neler düzeltildi (MVP scope)
1. **Model bağımsız katman mantığı**
   - Katman sayısı artık modelden dinamik okunuyor.
2. **Güven metriği düzeltmesi**
   - `max(logit)` yerine `max(softmax(logits))` kullanılıyor.
3. **Mimari uyumluluk**
   - Qwen tarzı `model.model.norm` ve GPT2 tarzı `transformer.ln_f` destekleniyor.
4. **Sunum odaklı çıktı**
   - Tablolu katman çıktıları ve erken çıkış adayı yüzdesi üretimi.

## Yarın için önerilen slide akışı (minimum risk)
1. Problem: "Sadece final token açıklaması yetersiz"
2. MVP yöntem: katman bazlı okuma + ablation + semantic stability router
3. Çalışan demo komutları
4. Limitasyonlar: gerçek compute skip henüz yok, tuned lens / patching gelecek iş
5. 2 haftalık plan: patching + task-aligned metric + gerçek early-exit ölçümü

## Tez için net mesaj
- Ana vurucu kısım: **Semantic Collapse bulgusu** (nedensel analize bağlamak şartıyla)
- Mühendislik katkısı: **Green Router** (ama gerçek tasarruf için inference stack seviyesinde skip uygulanmalı)
