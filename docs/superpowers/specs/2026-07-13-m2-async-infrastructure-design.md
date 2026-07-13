# M2: Async Infrastructure — Design

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Builds on:** M1 Pipeline Core (merged to `main`, commit `cbec508`)

## 1. Purpose

Turn the M1 in-process pipeline into a horizontally-scalable async service: submit a job over
HTTP, process it through Kafka-staged workers that reuse the M1 stage functions unchanged, and
retrieve a per-scene JSON report. State lives in Postgres, media/reports in MinIO, the watchlist
in pgvector.

## 2. Decisions (locked in brainstorming)

| Decision | Choice |
|---|---|
| Base | M1 merged to `main`; M2 on a fresh branch `m2-async-infra` |
| Test strategy | **Real local stack** — `docker compose up` required; tests are integration tests against Redpanda/Postgres/MinIO |
| Broker | **Redpanda** (Kafka-API, ARM-native) for local/test; Apache Kafka swappable in prod |
| Concurrency | **aiokafka + async FastAPI + asyncpg** (one async model) |
| DB layer | **Raw asyncpg + `schema.sql`** applied idempotently on startup; pgvector via `<=>` |
| Error handling | Typed `StageError` → mark job `failed`; robust retries + DLQ deferred to **M4** |
| Inference in tests | Selectable **`stub`** backend (deterministic) so infra tests need no ML deps/GPU |

Out of scope (deferred): retry loops + DLQ topics + eval harness (M4); Triton serving (M3);
auth/multi-tenancy; horizontal autoscaling policy.

## 3. Architecture

```
client ──POST /jobs──▶ API (async FastAPI)
                         │ writes job row (Postgres), produces ingest.requested
                         ▼
   ┌──────────── REDPANDA (Kafka API) ────────────────────────────────┐
   │  *.requested topics (one per stage) + stage.events (completions)  │
   └───┬────────────────────────────────────────────────┬─────────────┘
       │ each worker: consume <stage>.requested          │ emit stage.completed
       ▼                                                  ▼
   workers (1 image, entrypoint=stage; consumer group=stage → scale by partitions)
   ingest │ scene │ transcribe │ face │ mention │ aggregate
       │ read refs from Postgres/MinIO → call M1 stage fn (injected clients) → upsert
       ▼
   ┌── Postgres (jobs, scenes, watchlist+pgvector) ──┐   ┌── MinIO (media, keyframes, reports) ──┐
   └─────────────────────────────────────────────────┘   └───────────────────────────────────────┘
       ▲
   Coordinator: consume stage.events → read job/scene state → emit next *.requested (DAG)
```

The pipeline is a DAG with joins. **Postgres is the authoritative state store.** Workers are
idempotent (upsert keyed by `(job_id, scene_id, stage)`). A **Coordinator** consumes completion
events and emits next-stage events when preconditions are met. Semantics: at-least-once delivery
+ idempotent workers.

## 4. Components

All new code under `src/celebvision/`; every worker calls the existing `celebvision.stages.*`
functions and M1 protocols (`InferenceClient`, `LLMClient`, `WatchlistIndex`) — **no changes to
M1 stage logic** except the single-shot `end_s` fix (§7).

### 4.1 MessageBus (`celebvision/bus.py`)

Thin aiokafka wrapper:
- `async produce(topic: str, key: str, value: dict) -> None`
- `async consume(topic: str, group: str) -> AsyncIterator[Message]` (manual offset commit)
- Message envelope (JSON): `{"job_id": str, "scene_id": int | None, "stage": str, "payload": dict}`.
  Payloads carry references (MinIO keys, ids), never large blobs.

### 4.2 Workers (`celebvision/workers/<stage>.py`, one image)

One Docker image; `CMD`/entrypoint arg selects the stage. Consumer group = stage name (scale by
partitions). Each worker loop:
1. Consume from `<stage>.requested`.
2. Load inputs by reference from Postgres/MinIO.
3. Call the M1 stage function via injected clients.
4. Idempotent-upsert outputs to Postgres/MinIO, keyed by `(job_id, scene_id, stage)`.
5. Emit `stage.completed` to `stage.events`.
6. Commit offset.
7. On `StageError`: set `jobs.status='failed'` + `jobs.error`, commit offset (no poison-loop), log.

Stages: `ingest`, `scene`, `transcribe`, `face`, `mention`, `aggregate`.

### 4.3 Coordinator (`celebvision/coordinator.py`)

Consumes `stage.events`; decisions are pure functions of current Postgres state:
- `ingest.completed` → produce `scenes.requested` + `transcribe.requested`
- (`scenes.completed` ∧ `transcribe.completed`) → for each scene: `faces.requested{scene_id}` + `mentions.requested{scene_id}`
- (all scenes `faces.completed` ∧ all scenes `mentions.completed`) → `aggregate.requested`
- `aggregate.completed` → `jobs.status='done'`

Skips events for jobs already `failed`. Duplicate emits are safe (idempotent workers).

### 4.4 API (`celebvision/api/app.py`, async FastAPI)

- `POST /jobs` — body `{source: str, watchlist_id: str, keywords: list[str]}` → insert `jobs` row (`queued`), produce `ingest.requested`, return `{job_id}`.
- `GET /jobs/{id}` — `{status, error?, stages: {...}, scene_progress: {done, total}}`.
- `GET /reports/{id}` — the aggregated report JSON (from MinIO), 404 if not `done`.
- `GET /healthz` — liveness.

### 4.5 Storage

**Postgres** (`schema.sql`, applied idempotently):
- `jobs(id uuid pk, source_kind text, source_locator text, watchlist_id text, keywords jsonb, status text, error text, expected_scene_count int, created_at timestamptz, completed_at timestamptz)`
- `job_assets(job_id uuid pk, video_key text, audio_key text, transcript_key text)` — `transcript_key` points at the full transcript JSON in MinIO, written by the transcribe worker.
- `scenes(job_id uuid, scene_id int, start_s double precision, end_s double precision, keyframes jsonb, transcript text, faces jsonb, mentions jsonb, faces_status text, mentions_status text, primary key(job_id, scene_id))` — `scenes.transcript` holds the per-scene windowed text, populated by the mention worker (via M1 `scene_transcript_text`); the face worker sets `faces`, the mention worker sets `mentions`.
- `watchlist(canonical_id text, watchlist_id text, name text, aliases jsonb, embedding vector(512), primary key(canonical_id, watchlist_id))` — pgvector extension; cosine index optional.

Status values: `queued | running | failed | done`.

**MinIO** buckets: `media` (video/audio), `keyframes`, `reports`.

### 4.6 PgVectorWatchlistIndex (`celebvision/watchlist/pg_index.py`)

Implements M1's `WatchlistIndex` protocol against Postgres:
- `search(embedding, threshold)` → `SELECT canonical_id, name, 1 - (embedding <=> $1) AS score ... ORDER BY embedding <=> $1 LIMIT 1`; return match if `score >= threshold` else `None`.
- `names()` → `SELECT canonical_id, name, aliases`.
- `watchlist_id` attribute.
- Enrollment writes `(canonical_id, name, aliases, embedding)` rows (one per reference embedding; multiple rows per identity allowed).

M1's `analyze`/`enroll` CLI keeps working; watchlist backend selectable (`WATCHLIST_BACKEND=local|pg`).

### 4.7 Stub backends (`celebvision/inference/stub_client.py`, `celebvision/llm/stub_client.py`)

Deterministic, dependency-free implementations of `InferenceClient`/`LLMClient` selected by
`INFERENCE_BACKEND=stub` / `LLM_BACKEND=stub`. Used by the docker-compose **test profile** so the
end-to-end integration test exercises real Redpanda/Postgres/MinIO + the Coordinator DAG without
faster-whisper/InsightFace/Anthropic. Backends:
- `INFERENCE_BACKEND` ∈ `{local, stub}` (Triton added in M3).
- `LLM_BACKEND` ∈ `{anthropic, stub}`.

## 5. Data flow (happy path)

1. `POST /jobs` → API inserts `jobs` row (`queued`) + produces `ingest.requested`.
2. Ingest worker → download to MinIO `media/`, insert `job_assets`, emit `ingest.completed`.
3. Coordinator → `scenes.requested` + `transcribe.requested`.
4. Scene worker → detect scenes, write N `scenes` rows + keyframes to MinIO, set `expected_scene_count`, emit `scenes.completed`. Transcribe worker → write the full transcript JSON to MinIO, record `job_assets.transcript_key`, emit `transcribe.completed`.
5. Coordinator → per scene: `faces.requested{scene_id}` + `mentions.requested{scene_id}`. (Both are gated on `scenes.completed ∧ transcribe.completed`; faces strictly needs only scenes, but gating both on both keeps the DAG simple and is harmless.)
6. Face worker → read keyframes, match watchlist, write `scenes.faces` + `faces_status`. Mention worker → read the full transcript (`transcript_key`) + scene window via M1 `scene_transcript_text`, write `scenes.transcript` + `scenes.mentions` + `mentions_status`. Both emit completed.
7. Coordinator (all scenes done) → `aggregate.requested`.
8. Aggregate worker → build report (M1 `build_report`), write `reports/{id}.json` to MinIO, `jobs.status='done'`, `completed_at`, emit `aggregate.completed`.
9. Client `GET /reports/{id}` → report JSON.

## 6. Interfaces reused from M1

- `celebvision.stages.scenes/faces/mentions/aggregate` — stage functions, called by workers.
- `celebvision.interfaces` — `InferenceClient`, `LLMClient`, `WatchlistIndex`, `Transcript`, `FaceDetection`, `MentionExtraction`.
- `celebvision.models` — `Report`, `SceneReport`, etc. (report shape unchanged).
- `celebvision.media` — `ingest`, `extract_keyframe` (called by ingest/scene workers).

## 7. Error handling

- Typed `celebvision.errors.StageError(stage: str, reason: str)`. Raised by workers wrapping
  boundary failures (ingest download/ffmpeg, LLM call, DB/MinIO I/O) — resolves M1-handoff #1.
- Worker catch → `jobs.status='failed'`, `jobs.error='<stage>: <reason>'`, commit offset, log.
- Coordinator skips events for `failed` jobs.
- **M1-handoff #2 fixed here:** single-shot scene `end_s` set to `transcript.duration_s` (in the
  scene worker or `build_scene_windows` path) so the persisted report is self-consistent.
- Robust per-stage retries + `<stage>.dlq` topics deferred to M4.

## 8. Deployment (docker-compose)

Services:
- `redpanda` — Kafka-API broker (single node); optional `redpanda-console`.
- `postgres` — pgvector-enabled image.
- `minio` — S3-compatible storage.
- `init` — one-shot: create topics, apply `schema.sql`, create MinIO buckets; exits 0.
- `api` — uvicorn async FastAPI.
- `coordinator`.
- `worker-{ingest,scene,transcribe,face,mention,aggregate}` — one image, stage-selecting CMD, independently scalable.

Config via env: `KAFKA_BOOTSTRAP`, `POSTGRES_DSN`, `MINIO_ENDPOINT`/keys, `INFERENCE_BACKEND`
(default `local`), `LLM_BACKEND` (default `anthropic`), `WATCHLIST_BACKEND=pg`. A `test` compose
profile sets `INFERENCE_BACKEND=stub` + `LLM_BACKEND=stub`.

`docker compose up` boots the full stack.

## 9. Testing (real stack)

Prerequisite: `docker compose up -d` (or a session-scoped fixture that ensures it). Tests marked
`requires_stack` (deselected unless the stack is up).

- **Bus:** produce/consume round-trip through Redpanda.
- **PgVector watchlist:** enroll two identities, `search` returns the correct match above threshold.
- **API:** `POST /jobs` inserts a `jobs` row and produces `ingest.requested` (consume to verify);
  `GET /jobs/{id}` reflects status; `GET /reports/{id}` 404 before done.
- **End-to-end (stub backend):** `docker compose --profile test up`, enroll a stub identity,
  `POST /jobs` with a small file source, poll `GET /jobs/{id}` until `done`, assert the report is
  schema-valid with `celebrity_index` and per-scene rows in Postgres.
- CI: `docker compose up -d` → run suite → tear down.

## 10. File structure (new in M2)

```
src/celebvision/
  bus.py                     # aiokafka MessageBus + envelope
  errors.py                  # StageError
  config.py                  # env-driven settings + client/backend factories
  db.py                      # asyncpg pool + schema.sql loader + queries
  storage.py                 # MinIO blob helpers (put/get object)
  coordinator.py             # DAG orchestrator
  api/app.py                 # FastAPI app + routes
  workers/
    __init__.py              # worker entrypoint dispatch (stage -> loop)
    base.py                  # shared consume/commit/error-handling loop
    ingest.py scene.py transcribe.py face.py mention.py aggregate.py
  inference/stub_client.py   # deterministic InferenceClient
  llm/stub_client.py         # deterministic LLMClient
  watchlist/pg_index.py      # PgVectorWatchlistIndex
sql/schema.sql
docker/Dockerfile
docker-compose.yml
tests/integration/           # requires_stack tests
```

## 11. Handoff to M3

M3 (Triton serving) swaps `INFERENCE_BACKEND=triton` (a `TritonClient` implementing the same
`InferenceClient` protocol) — no worker/DAG changes. M4 adds retries, `<stage>.dlq` topics,
metrics, and the precision/recall eval harness.
