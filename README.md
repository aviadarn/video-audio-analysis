# celebvision

Detect **which celebrities appear in a video** — both **spoken** (mentioned in the audio) and
**on-screen** (recognized by face) — against a predefined watchlist, and produce a per-scene JSON
report with timestamps.

Input: a YouTube URL, an HLS `.m3u8`, or a local video file.
Output: a report where each scene lists spoken mentions, keyword hits, and on-screen faces, plus a
`celebrity_index` mapping each celebrity to the scenes they appear in.

## Pipeline

```
video ─▶ ingest (yt-dlp/ffmpeg) ─▶ scene detection (PySceneDetect) ─┬─▶ transcription (faster-whisper)
                                                                    └─▶ keyframes
                     per scene:  face recognition (InsightFace → watchlist)  +  spoken-mention/keyword extraction (LLM)
                                                              ▼
                                             aggregate ─▶ per-scene JSON report + celebrity_index
```

Models (chosen over the naive defaults): **faster-whisper** (ASR), **InsightFace** SCRFD+ArcFace
(faces), **pgvector** cosine (watchlist match), **Claude** (spoken-mention NER), **NVIDIA Triton**
(ArcFace serving with dynamic batching).

## Architecture (built in four milestones)

| Milestone | What it adds | Spec / Plan |
|---|---|---|
| **M1 Pipeline core** | In-process CPU pipeline + `celebvision analyze`/`enroll` CLI; all external deps behind pluggable `InferenceClient` / `LLMClient` / `WatchlistIndex` protocols | `docs/superpowers/*/2026-07-13-m1-*` |
| **M2 Async infra** | Kafka (Redpanda)-staged workers + Coordinator DAG + async FastAPI + Postgres/pgvector + MinIO + docker-compose | `docs/superpowers/*/2026-07-13-m2-*`, `docs/M2_RUNBOOK.md` |
| **M3 Triton serving** | ArcFace recognition served via Triton (`INFERENCE_BACKEND=triton`), detect+align stay local | `docs/superpowers/*/2026-07-13-m3-*`, `docs/M3_RUNBOOK.md` |
| **M4 Hardening** | Per-stage retries + `<stage>.dlq`, Prometheus metrics + structured logs, precision/recall eval harness, polish | `docs/superpowers/*/2026-07-13-m4-*`, `docs/M4_RUNBOOK.md` |

### Async service topology (M2+)

```
client ──POST /jobs──▶ API (FastAPI) ── produce ingest.requested ──▶ Redpanda (Kafka)
                                                                        │  topic-per-stage + stage.events + *.dlq
   ingest → scene → transcribe → face → mention → aggregate  (workers; consumer group per stage)
        │ read/write refs                                    │ emit stage.events
        ▼                                                    ▼
   Postgres (jobs, scenes, watchlist+pgvector)      MinIO (media, keyframes, reports)
        ▲
   Coordinator: consumes stage.events → emits next-stage requests per the DAG (state in Postgres)
```
Guarantees: at-least-once delivery + idempotent upserts; failed messages retry then dead-letter to
`<stage>.dlq` and the job is marked `failed`.

## Repository layout

```
src/celebvision/
  models.py interfaces.py config.py errors.py     # domain models, pluggable protocols, settings
  media.py                                        # ingest (yt-dlp/ffmpeg), keyframes, ffprobe
  stages/{scenes,faces,mentions,aggregate}.py     # pure pipeline stage functions (M1)
  inference/{local_client,stub_client,triton_client,composite}.py
  llm/{anthropic_client,stub_client}.py
  watchlist/{local_index,pg_index}.py             # numpy .npz (local) / pgvector (pg)
  bus.py db.py storage.py                          # aiokafka / asyncpg / aioboto3 adapters (M2)
  coordinator.py coordinator_main.py              # DAG orchestrator
  workers/{base,ingest,scene,transcribe,face,mention,aggregate}.py + __main__.py
  api/app.py                                       # FastAPI: POST /jobs, GET /jobs, /reports, /metrics, /healthz
  metrics.py logging.py eval.py cli.py            # observability, eval scorer, CLI (M1/M4)
scripts/{init_stack,prepare_triton_models}.py
sql/schema.sql   docker/Dockerfile   docker-compose.yml   model_repository/arcface/config.pbtxt
docs/superpowers/{specs,plans}/   docs/M{2,3,4}_RUNBOOK.md
```

## Prerequisites

- **Python 3.11+** (developed on 3.13). Create a venv: `python3.13 -m venv .venv`.
- **Docker** + Compose (for the async stack / Triton).
- **ffmpeg** on the host for real runs: `brew install ffmpeg` (macOS).
- Optional: **`ANTHROPIC_API_KEY`** (real LLM mention extraction), an **NVIDIA GPU** (production Triton).

## Install

```bash
python3.13 -m venv .venv
.venv/bin/pip install -U pip
# pick the extras you need:
.venv/bin/pip install -e '.[dev]'                          # fast unit tests only (light)
.venv/bin/pip install -e '.[dev,service,media,triton,local,obs]'   # everything
```
Extras: `dev` (pytest/ruff), `service` (fastapi/aiokafka/asyncpg/aioboto3/pgvector), `media`
(scenedetect/yt-dlp), `local` (faster-whisper/insightface/onnxruntime/opencv/anthropic), `triton`
(tritonclient/onnx), `obs` (prometheus-client).

## How to run

### A. In-process CLI (simplest — one video, no services)

```bash
export ANTHROPIC_API_KEY=...            # only if using the real LLM backend
# 1. enroll celebrities from reference photos into a local watchlist
.venv/bin/celebvision enroll --watchlist-id demo --canonical-id messi --name "Lionel Messi" \
  --alias messi --images ./refs/messi --out ./data/demo.npz
# 2. analyze a video → per-scene report JSON
.venv/bin/celebvision analyze "https://www.youtube.com/watch?v=<id>" \
  --watchlist ./data/demo.npz --keyword goal --out ./data/report.json --workdir ./data/work
```

### B. Async service (Kafka stack via docker-compose)

```bash
docker compose up --build                # api + coordinator + 6 workers + redpanda + postgres + minio + init
# submit a job:
curl -s localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"source":"https://youtu.be/<id>","watchlist_id":"demo","keywords":["goal"]}'
curl -s localhost:8000/jobs/<job_id>     # status
curl -s localhost:8000/reports/<job_id>  # report JSON (when done)
curl -s localhost:8000/metrics           # Prometheus metrics
```
Default backend is `stub` (deterministic, no ML). For real models set `INFERENCE_BACKEND=local`
(+`LLM_BACKEND=anthropic`) and build the image with the `local` extra.
Note: compose maps Postgres to host `5432` — remap if occupied (see `docs/M4_RUNBOOK.md`).

### C. Triton serving (M3)

```bash
.venv/bin/python scripts/prepare_triton_models.py            # copy ArcFace ONNX + config into model_repository/
docker compose --profile triton up triton                    # CPU (or --profile gpu on NVIDIA hardware)
# point the face worker at Triton with INFERENCE_BACKEND=triton, TRITON_URL=triton:8000
```

### D. Evaluate quality (M4)

```bash
.venv/bin/celebvision eval --report ./data/report.json --truth ./data/ground_truth.json --out ./data/eval.json
# precision/recall/F1 per modality (audio/face/combined), video + scene level
```

## Testing

```bash
.venv/bin/python -m pytest -q          # fast unit tests only (no Docker; stack/model tests auto-skip)
.venv/bin/ruff check src tests scripts # lint
```
Full integration tests need the infra stack up. Bring it up, then:
```bash
docker compose up -d redpanda postgres minio
CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:5432/celeb \
  .venv/bin/python scripts/init_stack.py     # create topics (incl. *.dlq), schema, buckets
CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:5432/celeb \
  .venv/bin/python -m pytest -q              # unit + integration
.venv/bin/python -m pytest -m slow -v        # model-dependent (ArcFace parity, etc.)
```
Markers: `requires_stack` (needs Docker infra; skipped unless `CELEBVISION_STACK=1`), `slow`
(needs local ML models; deselected by default).

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `KAFKA_BOOTSTRAP` | `localhost:19092` | Redpanda/Kafka brokers |
| `POSTGRES_DSN` | `postgresql://celeb:celeb@localhost:5432/celeb` | Postgres (job/scene state + watchlist) |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `http://localhost:9000` / `minioadmin` / `minioadmin` | object storage |
| `INFERENCE_BACKEND` | `stub` | `stub` \| `local` (faster-whisper+InsightFace) \| `triton` (ArcFace via Triton) |
| `LLM_BACKEND` | `stub` | `stub` \| `anthropic` |
| `WATCHLIST_BACKEND` | `pg` | `pg` (pgvector) \| `local` (`.npz`) |
| `ASR_BACKEND` | `stub` | transcriber used under the `triton` composite backend |
| `TRITON_URL` / `TRITON_MODEL` | `localhost:8000` / `arcface` | Triton endpoint + model |
| `MAX_ATTEMPTS` | `3` | per-stage retries before dead-lettering |
| `METRICS_PORT` | `9100` | worker Prometheus port |

## Roadmap (GPU-host milestone — not verifiable on non-NVIDIA dev machines)

ASR-on-Triton (NVIDIA Parakeet/Canary), TensorRT/FP16 conversion, and real `KIND_GPU` ArcFace
serving on NVIDIA hardware.

## Design docs

Full specs, task-by-task plans, and per-milestone runbooks live under `docs/superpowers/` and
`docs/M{2,3,4}_RUNBOOK.md`. To operate the project inside Claude Code, see the
`run-celebvision` skill in `.claude/skills/`.
