# M4 Runbook

## What M4 adds
- Per-stage retries + `<stage>.dlq` dead-letter topics (MAX_ATTEMPTS, default 3).
- Prometheus metrics (`/metrics` on API; METRICS_PORT on workers) + structured JSON logs.
- `celebvision eval --report r.json --truth t.json --out e.json` (precision/recall/F1).
- Polish: report duration_s/completed_at; Triton metadata name discovery + READY + zero-norm raise; streaming storage; `prepare_triton_models.py --kind`.

## Metrics
curl localhost:8000/metrics            # API (with the stack up)
# workers expose METRICS_PORT (default 9100)

## DLQ
Failed messages retry to <stage>.requested up to MAX_ATTEMPTS, then land on <stage>.dlq
and the job is marked failed. Inspect with any Kafka consumer on <stage>.dlq.

## Eval
celebvision eval --report data/report.json --truth data/truth.json --out data/eval.json

## Deferred to a GPU-host milestone
ASR-on-Triton (Parakeet/Canary), TensorRT/FP16, real KIND_GPU serving validation.
