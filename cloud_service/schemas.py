from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator
from worker import validate_config


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    model_name: str = Field(min_length=2, max_length=300)
    prompt: str = Field(min_length=1, max_length=4096)
    analysis_type: str = Field(default="layer_margin", max_length=80)
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_backend(self):
        validate_config(self.model_name, self.analysis_type, self.config)
        return self


class ExperimentRead(BaseModel):
    id: str
    name: str
    model_name: str
    prompt: str
    analysis_type: str
    config: dict[str, Any]
    status: str
    result: dict[str, Any] | None = None
    artifact_uri: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
