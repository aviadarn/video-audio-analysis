# M4: Hardening (reliability + polish + eval + observability) — Design

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Builds on:** M3 Triton serving (PR #3). M4 branches off `main` after M3 merges.

## 1. Purpose

Production-harden the celebvision pipeline: per-stage retries + dead-letter queues, Prometheus
metrics + structured logging, a precision/recall evaluation harness, and the small correctness
items deferred from M2/M3. Everything here is **additive** — no change to the Kafka DAG topology,
the stage functions' signatures, or the M1 report schema.

## 2. Scope & decisions

| Area | Decision |
|---|---|
| Retries/DLQ | **Envelope attempt-counter + `<stage>.dlq` topic**; immediate retry (no backoff delay) for M4 |
| Observability | **Prometheus (`prometheus-client`) metrics + structured JSON logging**; `/metrics` on API, metrics port on workers |
| Eval | **Pure scorer**: `score_report(report, ground_truth) -> EvalResult` (precision/recall/F1, per-modality, video + scene level); thin CLI |
| Polish | report `duration_s`/`completed_at`; `TritonFaceClient` metadata name-discovery + READY watchdog; `_embed` zero-norm raise; streaming storage; prep `--kind` + gpu `KIND_GPU` config |
| Branch | `m4-hardening` off `main` after M3 merges |

Deferred to a later **GPU-host milestone** (unverifiable on Apple Silicon): ASR-on-Triton
(Parakeet/Canary), TensorRT/FP16 conversion.

## 3. Reliability — retries + DLQ

- **Envelope** (`bus.py`): `Message` gains `attempts: int = 0`.
- **Settings**: `max_attempts` (`MAX_ATTEMPTS`, default `3`).
- **Worker base loop** (`workers/base.py`), on `StageError`:
  - if `msg.attempts + 1 < max_attempts`: re-produce to `f"{stage}.requested"` with
    `attempts = msg.attempts + 1`; record a retry metric + structured log; commit offset.
  - else: produce the message (unchanged payload, final attempts) to `f"{stage}.dlq"`; call
    `db.set_job_error(job_id, f"{stage}: {reason} (dlq after {attempts} attempts)")`; record a dlq
    metric; commit offset.
  - Success path unchanged (emits `stage.events`, commits). No poison-loop — the offset always
    commits.
- **init_stack**: create the six `<stage>.dlq` topics (`ingest.dlq`, `scenes.dlq`,
  `transcribe.dlq`, `faces.dlq`, `mentions.dlq`, `aggregate.dlq`).
- The Coordinator is unaffected (it never consumes `.dlq`; failed jobs are already skipped).

## 4. Observability

- **`metrics.py`** (`prometheus_client`, using an injectable `CollectorRegistry` for testability):
  - Counters: `celebvision_stage_processed_total{stage}`, `_stage_failed_total{stage}`,
    `_stage_retried_total{stage}`, `_stage_dlq_total{stage}`.
  - Histogram: `celebvision_stage_duration_seconds{stage}`.
  - Helpers: `record_processed(stage)`, `record_failed(stage)`, `record_retried(stage)`,
    `record_dlq(stage)`, `observe_duration(stage, seconds)`, `render() -> bytes`.
- **`logging.py`**: `get_logger()` returning a structured JSON logger; `log_stage(logger, event,
  job_id, stage, status, attempts)` emits one JSON line per stage event.
- **Wiring**: the worker base loop times each handler and records processed/failed/retried/dlq +
  duration, and logs stage events. API adds `GET /metrics` → `render()` with the
  `text/plain; version=0.0.4` content type. Workers call
  `prometheus_client.start_http_server(settings.metrics_port)` on startup (`METRICS_PORT`, default
  `9100`).
- New pip extra `obs = ["prometheus-client>=0.20"]`.
- Tests assert counter/histogram deltas via a fresh `CollectorRegistry`; logging output parsed as
  JSON with the expected keys.

## 5. Eval harness

- **Models** (`eval.py`, pydantic):
  - `GroundTruth`: `job_id: str`, `expected: list[str]` (canonical_ids), optional
    `expected_by_modality: dict[str, list[str]]` (keys `audio`/`face`), optional
    `expected_by_scene: dict[int, list[str]]`.
  - `ModalityScore`: `precision: float`, `recall: float`, `f1: float`, `tp: int`, `fp: int`, `fn: int`.
  - `EvalResult`: `job_id`, `video: dict[str, ModalityScore]` (keys `audio`/`face`/`combined`),
    `scene_mean_f1: float | None`.
- **`score_report(report: Report, gt: GroundTruth) -> EvalResult`** (pure):
  - Detected sets: `audio` = `{m.canonical_id}` over all `spoken_mentions`; `face` = over all
    `onscreen_faces`; `combined` = `{e.canonical_id}` over `celebrity_index`.
  - Expected set per modality: `gt.expected_by_modality[k]` if present else `gt.expected`.
  - `ModalityScore` from set precision/recall/F1 (P=tp/(tp+fp), R=tp/(tp+fn), F1 harmonic; guard
    zero denominators → 0.0).
  - `scene_mean_f1`: if `gt.expected_by_scene`, mean per-scene combined-F1 over scenes present in
    both; else `None`.
- **`score_dataset(reports_dir, truth_file) -> list[EvalResult] + aggregate`**: macro-averaged
  P/R/F1 across jobs.
- **CLI**: `celebvision eval --report <report.json> --truth <truth.json> --out <eval.json>`
  (single) — added to the existing `celebvision` Typer app.
- Pure unit tests on hand-computed sets (e.g. expected `{a,b,c}`, detected `{a,b,d}` → P=2/3,
  R=2/3).

## 6. Deferred polish

- **`aggregate` worker**: load the transcript (`transcript_key`) → `duration_s =
  transcript.duration_s`; `completed_at = datetime.now(timezone.utc).isoformat()` (the aggregate
  worker is a service entry point, so wall-clock is allowed here — unlike M1 `run_pipeline`).
- **`TritonFaceClient`**: on first use, query Triton model metadata to resolve the input/output
  tensor names (replacing hard-coded `input.1`/`683`) and verify the model is `READY`; raise a
  clear `StageError("face", ...)` if not ready/reachable. `_embed` raises
  `StageError("face", "zero-norm embedding")` instead of silently returning an unnormalized vector.
  (Guarded by the live Triton container the M3 setup already runs.)
- **`storage`**: `put_file`/`get_file` stream via aioboto3 `upload_fileobj`/`download_fileobj`
  (no whole-file in-memory buffering). `put_bytes`/`get_bytes` unchanged.
- **`prepare_triton_models.py`**: `render_config_pbtxt` already takes `kind`; add a `--kind`
  CLI flag (default `KIND_CPU`) and generate a `config.gpu.pbtxt` (`KIND_GPU`) alongside. The gpu
  compose profile mounts the `KIND_GPU` config. (Generation + wiring are testable; a real GPU run
  stays GPU-host-only.)

## 7. Interfaces reused / touched

- `bus.Message` (+`attempts`), `workers/base.run_worker` (retry/DLQ + metrics + logging),
  `config.Settings` (+`max_attempts`, `metrics_port`), `scripts/init_stack` (+dlq topics),
  `api/app` (+`/metrics`), `workers/aggregate` (report fields), `inference/triton_client`
  (metadata + zero-norm), `storage` (streaming), `scripts/prepare_triton_models` (`--kind`),
  `cli` (+`eval`). New: `metrics.py`, `logging.py`, `eval.py`.
- **No change** to: coordinator DAG, stage functions in `celebvision.stages.*`, the `Report`/
  `SceneReport` schema (fields only get populated), M1 CLI `analyze`/`enroll`.

## 8. Testing

- **Fast unit** (no infra): envelope `attempts`; worker retry/DLQ branch (fake bus/db — assert
  re-produce vs `.dlq` routing + attempts increment); metrics via `CollectorRegistry`; structured
  log JSON; eval scorer on known sets; aggregate report-field population; storage streaming (fake
  S3 or moto-style — or a `requires_stack` MinIO round-trip); `render_config_pbtxt(kind=...)`.
- **requires_stack**: DLQ integration — feed a stage a message that always fails, assert it lands
  on `<stage>.dlq` after `max_attempts` and the job is `failed`; `/metrics` endpoint returns
  Prometheus text.
- **slow / live-Triton**: TritonFaceClient metadata name-discovery + READY against the running
  container.
- **Regression**: full M1/M2/M3 suites stay green; `ruff check src tests scripts` clean.

## 9. Milestone exit criteria

- Retries + DLQ: a persistently-failing message reaches `<stage>.dlq` after `max_attempts` and the
  job is `failed`; verified by an integration test.
- `/metrics` exposes the stage counters/histogram; worker metrics port serves Prometheus text.
- Eval scorer produces correct P/R/F1 on unit fixtures; `celebvision eval` writes an `EvalResult`.
- Polish items landed: report `duration_s`/`completed_at` populated; Triton metadata discovery +
  READY + zero-norm raise; streaming storage; `--kind`/gpu config.
- Full M1/M2/M3 regression green; ruff clean; DAG/stage/schema unchanged.

## 10. Handoff — remaining GPU-host milestone

A future GPU-host milestone covers ASR-on-Triton (Parakeet/Canary), TensorRT/FP16 conversion, and
validating `KIND_GPU` serving on real NVIDIA hardware.
