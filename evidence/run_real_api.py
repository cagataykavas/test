"""Offline service validation using a private SQLite database and local artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from fastapi.testclient import TestClient

os.environ['S3_BUCKET'] = ''
root = Path(tempfile.mkdtemp(prefix='xai-real-api-'))
os.environ['DATABASE_URL'] = 'sqlite:///' + str(root / 'experiments.db')
os.environ['LOCAL_ARTIFACT_DIR'] = str(root / 'artifacts')
from app import app
from worker import SUPPORTED_MODEL, SUPPORTED_REVISION
from real_backend import _load

torch.set_num_threads(2)
_load.cache_clear()
load_start=time.perf_counter()
_load(SUPPORTED_MODEL, SUPPORTED_REVISION)
model_load_seconds=time.perf_counter()-load_start
config = {'backend':'huggingface_local','revision':SUPPORTED_REVISION,
          'candidate_a':' Ankara','candidate_b':' İstanbul'}
rows=[]
collisions=[]
with TestClient(app) as client:
    assert client.get('/health').status_code == 200
    for i, prompt in enumerate(["Türkiye'nin başkenti hangi şehirdir?", "Türkiye'nin başkenti hangi şehirdir?", "Fransa'nın başkenti hangi şehirdir?"]):
        start=time.perf_counter()
        response=client.post('/experiments',json={'name':f'real-{i}','model_name':SUPPORTED_MODEL,'prompt':prompt,'config':config})
        elapsed=time.perf_counter()-start
        assert response.status_code==201, response.text
        body=response.json()
        assert body['status']=='completed', body
        result=body['result']
        assert result['is_model_execution'] is True
        assert result['native_final_logit_max_error']==0
        artifact=json.loads(Path(body['artifact_uri']).read_text())
        assert artifact==result
        assert client.get('/experiments/'+body['id']).json()['result']==result
        rows.append({'prompt':prompt,'elapsed_seconds':elapsed,'result':result,
                     'artifact_sha256':hashlib.sha256(Path(body['artifact_uri']).read_bytes()).hexdigest()})
    invalid=client.post('/experiments',json={'name':'bad','model_name':SUPPORTED_MODEL,'prompt':'hello','config':{**config,'revision':'main'}})
    assert invalid.status_code==422
    for question, a, b in [("2 + 2?", " 4", " 5"), ("5 - 1?", " 4", " 3"), ("3 + 4?", " 7", " 8")]:
        rejected=client.post('/experiments',json={'name':'collision','model_name':SUPPORTED_MODEL,
            'prompt':question,'config':{**config,'candidate_a':a,'candidate_b':b}}).json()
        assert rejected['status']=='failed', rejected
        assert 'first-token metric cannot distinguish candidates' in rejected['result']['error']
        assert rejected['artifact_uri'] is None
        collisions.append({'prompt':question,'candidates':[a,b],'error':rejected['result']['error']})
    assert len(client.get('/experiments').json())==6
assert rows[0]['result']==rows[1]['result']
Path('evidence/real_api_result.json').write_text(json.dumps({'first_token_collisions_rejected':collisions,'model_load_seconds':model_load_seconds,'request_timing_scope':'all API requests use a preloaded checkpoint; includes inference/persistence/artifact operations','cache_info':str(_load.cache_info()),'rows':rows,'validated':'POST -> real model -> SQL persistence -> local JSON -> GET; repeated prompt identical','limits':'SQLite/local disk only. PostgreSQL, S3, Docker and remote CI not exercised. One checkpoint, three requests, two unique prompts. Latency shared-host descriptive only.'},indent=2,ensure_ascii=False)+'\n')
print('3 real API requests verified; 3 ambiguous Math candidate pairs rejected without artifacts; invalid revision rejected before insert')
