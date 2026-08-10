import json
import os
from pathlib import Path

import boto3

S3_BUCKET = os.getenv("S3_BUCKET", "")
LOCAL_ARTIFACT_DIR = Path(os.getenv("LOCAL_ARTIFACT_DIR", "artifacts"))


def save_json_artifact(experiment_id: str, payload: dict) -> str:
    key = f"experiments/{experiment_id}/result.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    if S3_BUCKET:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return f"s3://{S3_BUCKET}/{key}"

    path = LOCAL_ARTIFACT_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return str(path)
