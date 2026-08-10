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
