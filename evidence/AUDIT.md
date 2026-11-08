# Real-model worker integration

Baseline worker returned only deterministic hash/tanh demo values. New optional
backend runs the allowlisted Qwen checkpoint offline with revision pinning,
128-token budget, candidate checks, CPUfloat32 and a one-model cache.

Three of three API requests completed through real inference, SQLite persistence,
local artifact creation and GET roundtrip. Two unique prompts were used; the
repeated prompt produced identical results. All artifact/result comparisons were
exact. Unpinned revision rejected before database insertion.

Model load measured separately: 2.539s.
Preloaded end-to-end API durations: [1.442, 1.433, 1.413]s.
Cache: CacheInfo(hits=6, misses=1, maxsize=1, currsize=1).
These shared-host timings are descriptive, not a throughput/SLO guarantee.

Maximum final-logit difference versus native model output:0.0 for all requests.
These are numerical parity and lifecycle results, not answer accuracy.

13 unit/schema tests and Ruff passed. No Docker, PostgreSQL, S3, cloud deployment
or remote CI ran at the pre-publication validation snapshot. No paid services
were used, and no public branch had been pushed at that snapshot.

Reproduce from repository root:
```
PYTHONPATH=cloud_service python -m pytest -q cloud_service
HF_HUB_OFFLINE=1 HF_HOME=<cached-model-dir> PYTHONPATH=cloud_service python evidence/run_real_api.py
```

Model revision:7ae557604adf67be50417f59c2c2f167def9a775.
The model must be cached beforehand. Requirements-model lists optional dependencies;
the tested environment is retained separately.

## First-token ambiguity regression

The three historical Math candidate pairs (space+4 vs space+5, space+4 vs
space+3, space+7 vs space+8) tokenize to the same first token on this pinned
checkpoint. The real API rejected all three with an explicit metric-ambiguity
error, a failed record and no result artifact. No model forward is performed
after detecting a collision. Unit regression also verifies lock release.
These failures are expected safety/correctness outcomes, not successful
experiments or evidence that the model cannot answer arithmetic questions.
