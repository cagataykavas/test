# Struct-XAI — LLM Interpretability Research + Cloud Service

Struct-XAI is an experimental interpretability toolkit for studying how instruction-tuned language models form and revise decisions across layers.

The repository combines **LLM research code** with a lightweight **production-style service layer** so experiments can be submitted through an API, persisted in PostgreSQL, and exported to object storage.

## What is in this repository

The original research scripts cover logit-lens exploration, attention inspection, structured ablation, layer-wise decision-margin analysis, target trajectories, cross-model comparisons and SHAP-style visualization.

The `cloud_service/` layer adds a deployable systems wrapper around those research workflows.

## Cloud architecture

```text
Client
  │
  ▼
FastAPI service
  │
  ├──► PostgreSQL / RDS
  │      experiment metadata + metrics
  │
  ├──► S3
  │      JSON artifacts / large outputs
  │
  └──► Struct-XAI research runner
         model analysis / attribution
```

Local development uses Docker Compose with PostgreSQL. The included Terraform demonstrates an AWS deployment baseline using ECR, ECS/Fargate, RDS PostgreSQL and S3.

## Why a service layer?

Interpretability experiments often begin as isolated scripts. This project separates research from orchestration and persistence so the same analysis can be submitted through an API, tracked by experiment ID, reproduced from stored configuration, persisted in a relational database, exported as artifacts and containerized for repeatable execution.

## Quick start

```bash
cd cloud_service
docker compose up --build
```

Open the API docs at `http://localhost:8000/docs`.

Create an experiment:

```bash
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen-layer-margin-demo",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Ankara hangi ülkenin başkentidir?",
    "analysis_type": "layer_margin",
    "config": {"candidate_a": "Türkiye", "candidate_b": "Yunanistan"}
  }'
```

## Service stack

- FastAPI
- SQLAlchemy
- PostgreSQL
- Pydantic
- boto3 / S3
- Docker / Docker Compose
- Terraform
- AWS ECS Fargate
- AWS RDS PostgreSQL
- AWS ECR
- GitHub Actions

## Repository layout

```text
.
├── 01_llama3_theatrical_benchmark.py
├── 02_logit_lens_explorer.py
├── ...
├── 14_struct_shap_visualizer.py
├── cloud_service/
│   ├── app.py
│   ├── database.py
│   ├── schemas.py
│   ├── storage.py
│   ├── worker.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── infra/terraform/
└── .github/workflows/cloud-service-ci.yml
```

## Experiment model

Each experiment stores an ID, name, target model, prompt, analysis type, JSON configuration, execution status, result summary, optional S3 artifact URI and timestamps.

This is generic enough to support logit-lens, layer-margin, attribution and ablation analyses.

## Local vs AWS

Docker Compose runs the FastAPI service and PostgreSQL locally. Artifacts are stored locally unless an S3 bucket is configured.

Terraform provides a reference AWS architecture with ECR, ECS/Fargate, RDS PostgreSQL, S3, CloudWatch, IAM and networking resources. It is a deployment baseline rather than a claim that this public repository currently operates a paid cloud environment.

## CI

GitHub Actions installs dependencies, compiles the service modules, runs lightweight unit tests and builds the Docker image.

## Research status

Struct-XAI remains a research project. The cloud layer is designed to make the experiments easier to reproduce, inspect and operationalize without hiding the underlying analysis code behind a black box.

## Real local model execution (explicit opt-in)

The default worker is now explicitly labeled `backend: demo` and
`is_model_execution: false`. Its hash-derived values are not model results.
For actual model inference, install CPU PyTorch and `transformers` alongside the
service dependencies, cache the checkpoint beforehand, and submit:

```json
{"name":"qwen-real","model_name":"Qwen/Qwen2.5-0.5B-Instruct",
 "prompt":"Türkiye'nin başkenti hangi şehirdir?","analysis_type":"layer_margin",
 "config":{"backend":"huggingface_local",
 "revision":"7ae557604adf67be50417f59c2c2f167def9a775",
 "candidate_a":" Ankara","candidate_b":" İstanbul"}}
```

The backend is offline-only, CPU float32, with a single allowlisted checkpoint,
no remote code, a 128-token input budget and two distinct first-token candidates.
Concurrent real inference is rejected instead of creating an unbounded queue.
The last Hugging Face hidden state is already normalized; the adapter avoids
normalizing it again and verifies native final-logit parity. Results include
model revision, prompt digest, candidate token IDs, runtime and raw layer margins.
These first-token margins are not full answer probabilities or causal explanations.

`evidence/run_real_api.py` exercises the API with a temporary SQLite database and
local artifact storage. It validates actual model execution, repeated-result
identity, persistence and artifact consistency, and rejection of an unpinned
revision. It does not validate PostgreSQL, S3, Docker or a cloud deployment.
The old CI description above describes intended infrastructure; no currently
active workflow or remote CI pass is claimed by this change.
