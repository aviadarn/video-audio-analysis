---
name: run-celebvision
description: Use when running, operating, or testing the celebvision video celebrity-analysis pipeline in this repo — the CLI (analyze/enroll/eval), the async Kafka+FastAPI stack via docker-compose, Triton serving, bringing up the local infra, or running the test suites. Triggers on "run the pipeline", "analyze a video", "start the stack", "submit a job", "run the tests", "bring up infra", "enroll a celebrity", "run eval".
---

# Running celebvision

celebvision detects celebrities in a video by **spoken mention** (Whisper + LLM) and **on-screen
face** (InsightFace → watchlist), emitting a per-scene JSON report. Full architecture + config in
the repo `README.md`; per-milestone details in `docs/M{2,3,4}_RUNBOOK.md`.

**Always use the venv by explicit path** (`.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/celebvision`).
Shell state does not persist between commands here, so do NOT rely on `source .venv/bin/activate`.

## First-time setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev]'                              # fast unit tests only
# for real runs / full stack, also:
.venv/bin/pip install -e '.[dev,service,media,triton,local,obs]'
brew install ffmpeg                                            # needed for real ingest/scene
```

## Choose a backend (env)

- `INFERENCE_BACKEND`: `stub` (default, deterministic, no ML) | `local` (faster-whisper + InsightFace) | `triton` (ArcFace via Triton).
- `LLM_BACKEND`: `stub` (default) | `anthropic` (needs `ANTHROPIC_API_KEY`).
- `WATCHLIST_BACKEND`: `pg` (default, pgvector) | `local` (`.npz`).

## A. Run one video with the CLI (no services)

```bash
export ANTHROPIC_API_KEY=...   # only if LLM_BACKEND=anthropic
.venv/bin/celebvision enroll --watchlist-id demo --canonical-id messi --name "Lionel Messi" \
  --alias messi --images ./refs/messi --out ./data/demo.npz
.venv/bin/celebvision analyze "<youtube-url|hls-url|file.mp4>" \
  --watchlist ./data/demo.npz --keyword goal --out ./data/report.json --workdir ./data/work
```

## B. Run the async stack (Kafka + API + workers)

```bash
docker compose up --build       # redpanda, postgres(pgvector), minio, init, api, coordinator, 6 workers
curl -s localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"source":"<url-or-file>","watchlist_id":"demo","keywords":["goal"]}'   # -> {"job_id": ...}
curl -s localhost:8000/jobs/<job_id>       # status (queued|running|failed|done)
curl -s localhost:8000/reports/<job_id>    # report JSON when done
curl -s localhost:8000/metrics             # Prometheus metrics
```
Default compose backend is `stub`. Compose maps Postgres to host `5432`; if that port is taken,
remap it and set `POSTGRES_DSN` accordingly (see `docs/M4_RUNBOOK.md`).

## C. Triton serving (ArcFace faces)

```bash
.venv/bin/python scripts/prepare_triton_models.py     # writes model_repository/arcface/{1/model.onnx,config.pbtxt}
docker compose --profile triton up triton             # CPU; use --profile gpu on NVIDIA hardware
# route the face worker at Triton: INFERENCE_BACKEND=triton, TRITON_URL=triton:8000
```
Note: on Apple Silicon, Triton runs under linux/amd64 QEMU emulation (CPU only, slow but works).

## D. Evaluate report quality

```bash
.venv/bin/celebvision eval --report ./data/report.json --truth ./data/ground_truth.json --out ./data/eval.json
```

## Tests

```bash
.venv/bin/python -m pytest -q            # fast unit only (no Docker; stack/model tests auto-skip)
.venv/bin/ruff check src tests scripts   # lint
```
Integration + model tests need infra and/or local models. Bring up infra, seed it, then run with
the stack env prefix:
```bash
docker compose up -d redpanda postgres minio
# (Postgres default is host :5432; if you ran it on another port, adjust POSTGRES_DSN below.)
export STACK='CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:5432/celeb'
env $STACK .venv/bin/python scripts/init_stack.py    # topics (incl. *.dlq), schema, buckets
env $STACK .venv/bin/python -m pytest -q             # unit + integration (requires_stack)
.venv/bin/python -m pytest -m slow -v                # model-dependent (ArcFace parity, etc.)
```
- `requires_stack` tests are skipped unless `CELEBVISION_STACK=1` and the infra is reachable.
- `slow` tests need the local ML models (InsightFace `buffalo_l` cache) and are deselected by default.
- A live-Triton test runs only when `TRITON_URL` is reachable (else it skips):
  `TRITON_URL=localhost:8000 .venv/bin/python -m pytest tests/integration/test_triton_e2e.py -v`.

## Reliability / operability notes

- Per-stage failures retry to `<stage>.requested` up to `MAX_ATTEMPTS` (default 3), then dead-letter
  to `<stage>.dlq` and the job is marked `failed`. Inspect a DLQ with any Kafka consumer on `<stage>.dlq`.
- Workers expose Prometheus metrics on `METRICS_PORT` (default 9100); the API on `/metrics`.
