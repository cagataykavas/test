# Struct-XAI (End-of-Day Presentable Workflow)

This repo includes a reproducible baseline pipeline and a report generator:

- `06_struct_xai_core.py`: Layer-wise Struct-XAI analysis + token edits + router diagnostic
- `16_struct_xai_report.py`: Auto-generates a mock-paper-ready markdown summary

---

## Step-by-step (today’s practical plan)

### Step 1 — Create experiment payload quickly (works even without torch)
```bash
python 06_struct_xai_core.py \
  --plan-only \
  --target " sahne" \
  --delete sinirle masaya \
  --replace Adam=Oyuncu \
  --out-json struct_xai_plan.json
```

### Step 2 — Run the real model analysis (when torch + transformers are available)
```bash
python 06_struct_xai_core.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cpu \
  --target " sahne" \
  --delete sinirle masaya \
  --replace Adam=Oyuncu \
  --out-json struct_xai_results.json
```

### Step 3 — Generate presentation/report draft
```bash
python 16_struct_xai_report.py \
  --in-json struct_xai_results.json \
  --out-md struct_xai_mock_report.md
```

### Step 4 — Use these three artifacts in your slide deck
1. `struct_xai_results.json`
2. `struct_xai_mock_report.md`
3. Console output table from `06_struct_xai_core.py`

---

## What this baseline proves

1. **Layer trajectory:** top-token belief shifts through layers.
2. **Intervention sensitivity:** deleting/replacing key words changes target log-prob by layer.
3. **Router signal:** a candidate stability-based exit layer can be estimated.

---

## Important caveat

The router output is currently a **counterfactual diagnostic** (it does not yet skip model execution).  
To claim true compute savings, implement real early-exit execution.
