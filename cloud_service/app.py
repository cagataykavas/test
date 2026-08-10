from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from database import Experiment, SessionLocal, init_db
from schemas import ExperimentCreate, ExperimentRead
from storage import save_json_artifact
from worker import run_experiment


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Struct-XAI Experiment Service",
    version="1.0.0",
    description="API for submitting, persisting and exporting LLM interpretability experiments.",
    lifespan=lifespan,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/experiments", response_model=ExperimentRead, status_code=201)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)):
    exp = Experiment(**payload.model_dump(), status="running")
    db.add(exp)
    db.commit()
    db.refresh(exp)

    try:
        result = run_experiment(
            model_name=exp.model_name,
            prompt=exp.prompt,
            analysis_type=exp.analysis_type,
            config=exp.config,
        )
        exp.result = result
        exp.artifact_uri = save_json_artifact(exp.id, result)
        exp.status = "completed"
    except Exception as exc:
        exp.status = "failed"
        exp.result = {"error": str(exc)}

    db.commit()
    db.refresh(exp)
    return exp


@app.get("/experiments", response_model=list[ExperimentRead])
def list_experiments(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    return (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .limit(limit)
        .all()
    )


@app.get("/experiments/{experiment_id}", response_model=ExperimentRead)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp
