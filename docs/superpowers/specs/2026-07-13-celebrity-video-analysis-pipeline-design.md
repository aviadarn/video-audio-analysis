# Celebrity Video Analysis Pipeline — Design

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation

## 1. Purpose

Process a video (HLS `.m3u8`, plain file, or YouTube URL for testing) and produce a
per-scene JSON report describing which celebrities appear — both **spoken** (mentioned in
the audio) and **on-screen** (recognized by face) — against a **predefined watchlist**,
plus keyword hits.

The system targets high throughput via a **Kafka**-staged pipeline and **NVIDIA Triton**
model serving, while remaining runnable on a CPU-only Mac for local development.

## 2. Scope & key decisions

Decisions locked during brainstorming:

| Decision | Choice |
|---|---|
| Detection modalities | **Audio (spoken names) + visual (faces on scene keyframes)** |
| Celebrity universe | **Predefined watchlist** (fixed set + reference photos) |
| Hardware target | **CPU dev now (Mac), NVIDIA Triton for GPU scale later** |
| Interface | **REST API + message queue** (async jobs) |
| Message bus | **Apache Kafka** (KRaft mode) — Redpanda as ARM-native dev alternative |
| Model serving | **Triton Inference Server** (GPU), with in-process CPU fallback for dev |
| Report | **Per-scene JSON**, stored in DB + downloadable |

Out of scope (YAGNI): HTML report view, burned-in video overlays, open-vocabulary face
recognition, real-time/streaming SLA, multi-tenant auth.

## 3. Recommended models

| Stage | Model | Rationale |
|---|---|---|
| Scene detection | PySceneDetect `AdaptiveDetector` + ffmpeg keyframe extraction | Robust shot-boundary detection; keyframes feed face recognition |
| Transcription (dev) | **faster-whisper** (CTranslate2, `large-v3` / `distil-large-v3`) | ~4× faster than openai-whisper on CPU |
| Transcription (word timestamps) | **WhisperX** (VAD + forced alignment) | Word-level timestamps required to map speech into scene windows; optional diarization |
| Transcription (GPU scale, future) | **NVIDIA Parakeet / Canary (NeMo)** via Triton/Riva | SOTA ASR, high GPU throughput, Triton-native |
| Face detection + embedding | **InsightFace** (SCRFD detector + ArcFace `buffalo_l`), ONNX | Industry standard; runs CPU + GPU; ONNX → Triton/TensorRT clean |
| Watchlist match | **Cosine similarity** over ArcFace embeddings (pgvector) | Simple, fast, evaluable; no extra index infra |
| Celebrity mentions | **Claude Haiku** structured call: transcript → watchlist name linking + keyword hits; optional **GLiNER** local NER pre-filter | Handles nicknames/aliases/pronoun context; local option to cut API cost/latency |
| Serving | **Triton** hosts ASR + face ONNX models with dynamic batching; LLM stays external API | Triton fits vision/ASR GPU workloads; LLM served via Anthropic API |

## 4. Architecture

Kafka-staged pipeline. Each stage is a Kafka consumer group; scale by adding partitions +
consumers. Triton serves the GPU-heavy inference models behind an abstraction.

```
                 ┌─────────┐   POST /jobs (youtube|hls|file, watchlist, keywords)
   client ─────► │  API    │   GET /jobs/{id} , GET /reports/{id}
                 │ FastAPI │
                 └────┬────┘
                      │ produce
        ┌─────────────▼──────────────────── KAFKA (KRaft, topic-per-stage) ───────────────┐
        │  ingest.req → scene.req + transcribe.req → face.req(/scene) + mention.req(/scene)│
        │                          → aggregate.req ;  *.dlq for failures                   │
        └──┬────────┬──────────┬───────────┬────────────┬──────────────┬──────────────────┘
           ▼        ▼          ▼           ▼            ▼              ▼
        ingest   scene     transcribe    face        mention      aggregate
        (yt-dlp/ (PySceneD  (WhisperX)   (InsightF)  (Claude NER)  (build JSON)
         ffmpeg)  +keyframe)     │           │            │            │
           │        │            └──►Triton◄─┘            │            │
           ▼        ▼                (gRPC: whisper, scrfd, arcface)   ▼
        ┌──────────────── Postgres (job+scene state, pgvector watchlist) ─────────┐
        └──────────────── MinIO (media, keyframes, report.json) ──────────────────┘
                      ▲
              Coordinator: reads job state, emits next-stage events when preconditions met
```

### 4.1 Why a Coordinator

The pipeline is a DAG with joins:
- `mention` needs **transcript AND scene windows**.
- `aggregate` needs **all per-scene face results AND all per-scene mention results**.

Kafka-native stream joins are fragile for this. Instead:
- **Postgres holds authoritative job + scene state.**
- Workers are **idempotent**, keyed by `(job_id, scene_id, stage)` via upsert.
- A lightweight **Coordinator** consumes `*.completed` events and emits the next-stage
  event(s) once preconditions are satisfied.

### 4.2 DAG sequencing

```
ingest.completed
   └─► emit scene.req + transcribe.req
(scene.completed AND transcribe.completed)
   └─► emit face.req (per scene) + mention.req (per scene)
(all face.completed AND all mention.completed)
   └─► emit aggregate.req
aggregate.completed
   └─► job status = done
```

## 5. Components

| Component | Responsibility | Key deps |
|---|---|---|
| **API** (FastAPI) | Submit job, poll status, fetch report; produces `ingest.req` | Kafka producer, Postgres |
| **Ingest worker** | yt-dlp (YouTube) / ffmpeg (HLS, file) → 16kHz mono WAV + video → MinIO | yt-dlp, ffmpeg, MinIO |
| **Scene worker** | PySceneDetect `AdaptiveDetector` → scene boundaries; ffmpeg extracts 1–3 keyframes/scene → MinIO | PySceneDetect, ffmpeg |
| **Transcribe worker** | WhisperX / faster-whisper via `InferenceClient` → segments with word timestamps | InferenceClient |
| **Face worker** | InsightFace SCRFD + ArcFace on keyframes → embeddings → cosine match vs watchlist | InferenceClient, pgvector |
| **Mention worker** | Align transcript words into scene windows → Claude structured call → watchlist mentions + keyword hits | LLMClient |
| **Aggregate worker** | Join per-scene face + mention results → report JSON → MinIO + Postgres | Postgres, MinIO |
| **Coordinator** | DAG sequencing from job state | Kafka, Postgres |
| **Watchlist enrollment CLI** | name + aliases + reference photos → ArcFace embeddings → pgvector; aliases feed audio matching | InsightFace, pgvector |

Each unit has a single purpose, communicates through Kafka topics + shared state stores,
and is independently testable with mocked inference/LLM clients.

## 6. Model serving abstraction

`InferenceClient` interface, implementation chosen by env var `INFERENCE_BACKEND`:

- `local` — in-process CPU models (faster-whisper, InsightFace-CPU). Fast iteration on Mac;
  no Triton container required.
- `triton` — gRPC client to Triton hosting models `whisper` (or `parakeet`), `scrfd`,
  `arcface` as ONNX with dynamic batching. GPU production path.

Swapping backends is config-only; no pipeline code changes. The LLM is a separate
`LLMClient` (Anthropic API), never routed through Triton.

Note: Triton official images are Linux/amd64 (CUDA-oriented). On Apple Silicon a Triton CPU
container runs only under emulation (slow) — hence `local` is the default dev backend and
`triton` is the GPU deployment backend.

## 7. Watchlist enrollment

Offline CLI: `enroll --name "<name>" --aliases "<a>,<b>" --images <dir>`

1. Detect + embed each reference photo with InsightFace (SCRFD + ArcFace).
2. Store per-identity: canonical id, name, aliases, one or more ArcFace embeddings in
   Postgres (**pgvector**), plus an alias table used by the audio mention matcher.
3. Face matching = cosine nearest-neighbor over pgvector with a configurable threshold.

## 8. Report schema

```jsonc
{
  "job_id": "...",
  "source": "youtube|hls|file + locator",
  "duration_s": 0,
  "watchlist_id": "...",
  "completed_at": "ISO-8601",
  "scenes": [
    {
      "scene_id": 0,
      "start_s": 0.0,
      "end_s": 0.0,
      "keyframes": ["minio://bucket/key.jpg"],
      "transcript": "...",
      "spoken_mentions": [
        {"name": "...", "canonical_id": "...", "confidence": 0.0, "evidence_span": "..."}
      ],
      "keyword_hits": [
        {"keyword": "...", "count": 0, "spans": ["..."]}
      ],
      "onscreen_faces": [
        {"name": "...", "canonical_id": "...", "confidence": 0.0, "bbox": [0,0,0,0], "keyframe": "minio://..."}
      ]
    }
  ],
  "celebrity_index": [
    {"name": "...", "canonical_id": "...", "scenes": [0], "modalities": ["audio", "face"]}
  ]
}
```

- Per-scene view → "which celebrities appear at this timestamp".
- `celebrity_index` → "which timestamps contain each celebrity".

## 9. Deployment (docker compose)

Services:
- `api` — FastAPI + uvicorn
- `coordinator`
- `worker-*` — one image, entrypoint selects stage; scaled independently (`--scale`)
- `kafka` — KRaft mode, single broker for dev (Redpanda documented as ARM-native drop-in)
- `postgres` — with `pgvector` extension
- `minio` — S3-compatible object storage
- `kafka-ui` — optional, for inspecting topics
- `triton` — GPU model server, behind `--profile gpu`

Run:
- Dev (CPU): `docker compose up` with `INFERENCE_BACKEND=local`
- GPU: `docker compose --profile gpu up` with `INFERENCE_BACKEND=triton`

Kafka messages carry object references (MinIO keys) + ids, never large media blobs.

## 10. Error handling

- **Retries:** Kafka redelivery with max attempts; exhausted messages → `<stage>.dlq`.
- **Failure state:** job marked `failed` with stage + reason in Postgres.
- **Idempotency:** all stage outputs written as upserts keyed by `(job_id, scene_id, stage)`,
  so redelivery is safe.
- **Timeouts:** per-stage processing timeout; timed-out work routed to DLQ.
- **Job status:** `queued | running | failed | done`, with per-stage status tracked.

## 11. Testing strategy

- **Unit (TDD):** each stage tested in isolation with mocked `InferenceClient` / `LLMClient`;
  scene/report builders tested against fixtures.
- **Integration:** `docker compose up` → POST a known YouTube URL → assert report JSON
  conforms to schema and contains expected scene count.
- **Eval harness:** labeled test clip → precision/recall for (a) spoken mentions and
  (b) on-screen face matches vs ground truth; tracked over time.

## 12. Open sub-decisions (defaulted)

- Watchlist face index: **pgvector** (default) vs standalone FAISS.
- Transcription: **WhisperX** for word-level timestamps (default) vs plain faster-whisper.

Both defaults chosen; revisit only if implementation friction arises.
