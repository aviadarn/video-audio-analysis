# docs/M2_RUNBOOK.md

## Local infra
docker compose up -d redpanda postgres minio
.venv/bin/python scripts/init_stack.py   # topics + schema + buckets

## Run tests
.venv/bin/python -m pytest -q                         # unit only (stack tests skipped)
CELEBVISION_STACK=1 .venv/bin/python -m pytest -q      # + integration (stack must be up)

## Full containerized stack (builds app image; heavy first build)
docker compose up --build
# POST a job:
curl -s localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"source":"https://youtu.be/<id>","watchlist_id":"wl1","keywords":["goal"]}'
curl -s localhost:8000/jobs/<job_id>
curl -s localhost:8000/reports/<job_id>

## Enroll a watchlist (pgvector) — requires real embeddings from the local backend image.
## For stub demos, seed via PgVectorWatchlistIndex with STUB_EMBEDDING.
