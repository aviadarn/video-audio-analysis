# M4: Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production-harden celebvision — per-stage retries + dead-letter queues, Prometheus metrics + structured logging, a precision/recall eval scorer, and the deferred M2/M3 polish — all additive, with no change to the Kafka DAG, stage-function signatures, or the report schema.

**Architecture:** Message envelope gains an `attempts` counter; the worker base loop retries in-place or routes to `<stage>.dlq`, timing/counting/logging each stage. A metrics module (prometheus_client on a private registry) and a structured JSON logger are wired into that loop and exposed via `/metrics`. A pure `eval` scorer computes precision/recall/F1 from a report + ground-truth. Polish items tighten the aggregate report fields, TritonFaceClient robustness, storage streaming, and the Triton GPU config.

**Tech Stack:** Python 3.11+, aiokafka, asyncpg, aioboto3, FastAPI, prometheus-client, tritonclient, pydantic, pytest.

## Global Constraints

- Python **3.11+**, existing `.venv`; `python`/`pip` mean `.venv/bin/python` / `.venv/bin/pip`.
- Builds on merged M3. **Additive only** — do NOT change the coordinator DAG, the `celebvision.stages.*` functions, or the `Report`/`SceneReport` schema (M4 only *populates* existing report fields). The default (profile-less) stack behavior stays unchanged.
- **Retries/DLQ**: envelope `attempts: int = 0`; `Settings.max_attempts` (`MAX_ATTEMPTS`, default `3`); on `StageError`, retry to `<stage>.requested` while `attempts+1 < max_attempts`, else route to `<stage>.dlq` + mark job failed; offset always commits (no poison-loop). Immediate retry (no backoff) for M4.
- **Metrics**: `prometheus_client` on a module-level `CollectorRegistry` (NOT the global default — keeps tests isolated). Metric names: `celebvision_stage_processed_total{stage}`, `_stage_failed_total{stage}`, `_stage_retried_total{stage}`, `_stage_dlq_total{stage}`, histogram `celebvision_stage_duration_seconds{stage}`. New pip extra `obs = ["prometheus-client>=0.20"]`. Workers serve metrics on `METRICS_PORT` (default `9100`); API adds `GET /metrics`.
- **Eval**: pure `score_report(report, ground_truth) -> EvalResult`; P=tp/(tp+fp), R=tp/(tp+fn), F1 harmonic, zero denominators → 0.0.
- **Test markers**: unit tests always run; infra tests `@pytest.mark.requires_stack` (skipped unless `CELEBVISION_STACK=1`); model/Triton tests `@pytest.mark.slow` or `skipif TRITON_URL`. Stack tests use Postgres on host **55432** (`POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb`), Redpanda `localhost:19092`, MinIO `http://localhost:9000` — matching the M2/M3 runbooks.
- Full M1/M2/M3 regression must stay green; `ruff check src tests scripts` clean. Commit after every task (Conventional Commits).

---

### Task 1: Envelope `attempts` + reliability/metrics settings

**Files:**
- Modify: `src/celebvision/bus.py`
- Modify: `src/celebvision/config.py`
- Modify: `tests/test_bus_envelope.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `Message` gains `attempts: int = 0`; `Settings` gains `max_attempts: int = 3` (`MAX_ATTEMPTS`) and `metrics_port: int = 9100` (`METRICS_PORT`).

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_bus_envelope.py
def test_envelope_attempts_default_and_roundtrip():
    from celebvision.bus import Message, encode, decode
    assert Message(job_id="j", stage="faces").attempts == 0
    m = decode(encode(Message(job_id="j", stage="faces", attempts=2)))
    assert m.attempts == 2
```

```python
# append to tests/test_config.py
def test_settings_reliability_defaults():
    from celebvision.config import Settings
    s = Settings.from_env({})
    assert s.max_attempts == 3
    assert s.metrics_port == 9100
    s2 = Settings.from_env({"MAX_ATTEMPTS": "5", "METRICS_PORT": "9200"})
    assert s2.max_attempts == 5
    assert s2.metrics_port == 9200
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bus_envelope.py::test_envelope_attempts_default_and_roundtrip tests/test_config.py::test_settings_reliability_defaults -v`
Expected: FAIL (`attempts`/`max_attempts` missing).

- [ ] **Step 3: Add `attempts` to `Message`**

In `src/celebvision/bus.py`, add to the `Message` model (after `payload`):
```python
    attempts: int = 0
```

- [ ] **Step 4: Add settings** (in `src/celebvision/config.py`)

Dataclass fields (after `triton_model`):
```python
    max_attempts: int = 3
    metrics_port: int = 9100
```
`from_env` mappings (after the `triton_model=` line) — note the int casts:
```python
            max_attempts=int(e.get("MAX_ATTEMPTS", d.max_attempts)),
            metrics_port=int(e.get("METRICS_PORT", d.metrics_port)),
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_bus_envelope.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/celebvision/bus.py src/celebvision/config.py tests/test_bus_envelope.py tests/test_config.py
git commit -m "feat: add message attempts counter and reliability settings"
```

---

### Task 2: Metrics module

**Files:**
- Create: `src/celebvision/metrics.py`
- Modify: `pyproject.toml` (add `obs` extra)
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces (`celebvision.metrics`): module-level `REGISTRY = CollectorRegistry()`; `record_processed(stage)`, `record_failed(stage)`, `record_retried(stage)`, `record_dlq(stage)`, `observe_duration(stage, seconds)`, `render() -> bytes`, `CONTENT_TYPE: str`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
from celebvision import metrics


def test_counters_and_histogram_record():
    metrics.record_processed("faces")
    metrics.record_processed("faces")
    metrics.record_failed("faces")
    metrics.record_retried("faces")
    metrics.record_dlq("faces")
    metrics.observe_duration("faces", 0.5)
    g = metrics.REGISTRY.get_sample_value
    assert g("celebvision_stage_processed_total", {"stage": "faces"}) == 2.0
    assert g("celebvision_stage_failed_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_retried_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_dlq_total", {"stage": "faces"}) == 1.0
    assert g("celebvision_stage_duration_seconds_count", {"stage": "faces"}) == 1.0

def test_render_returns_prometheus_text():
    out = metrics.render()
    assert isinstance(out, bytes)
    assert b"celebvision_stage_processed_total" in out
    assert metrics.CONTENT_TYPE.startswith("text/plain")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.metrics'`

- [ ] **Step 3: Add `obs` extra + install**

In `pyproject.toml` `[project.optional-dependencies]`:
```toml
obs = ["prometheus-client>=0.20"]
```
Run: `.venv/bin/pip install -e '.[dev,service,media,triton,local,obs]'`

- [ ] **Step 4: Write the metrics module**

```python
# src/celebvision/metrics.py
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()
CONTENT_TYPE = CONTENT_TYPE_LATEST

_processed = Counter("celebvision_stage_processed_total",
                     "Stage messages processed", ["stage"], registry=REGISTRY)
_failed = Counter("celebvision_stage_failed_total",
                  "Stage handler failures", ["stage"], registry=REGISTRY)
_retried = Counter("celebvision_stage_retried_total",
                   "Stage messages retried", ["stage"], registry=REGISTRY)
_dlq = Counter("celebvision_stage_dlq_total",
               "Stage messages sent to DLQ", ["stage"], registry=REGISTRY)
_duration = Histogram("celebvision_stage_duration_seconds",
                      "Stage handler duration", ["stage"], registry=REGISTRY)


def record_processed(stage: str) -> None:
    _processed.labels(stage=stage).inc()


def record_failed(stage: str) -> None:
    _failed.labels(stage=stage).inc()


def record_retried(stage: str) -> None:
    _retried.labels(stage=stage).inc()


def record_dlq(stage: str) -> None:
    _dlq.labels(stage=stage).inc()


def observe_duration(stage: str, seconds: float) -> None:
    _duration.labels(stage=stage).observe(seconds)


def render() -> bytes:
    return generate_latest(REGISTRY)
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/celebvision/metrics.py tests/test_metrics.py
git commit -m "feat: add prometheus metrics module"
```

---

### Task 3: Structured logging

**Files:**
- Create: `src/celebvision/logging.py`
- Create: `tests/test_logging.py`

**Interfaces:**
- Produces (`celebvision.logging`): `stage_record(event, job_id, stage, status, attempts) -> dict` (pure); `get_logger(name="celebvision") -> logging.Logger`; `log_stage(logger, event, job_id, stage, status, attempts) -> None` (emits `json.dumps(stage_record(...))` at INFO).

- [ ] **Step 1: Write failing test**

```python
# tests/test_logging.py
import json
import logging
from io import StringIO
from celebvision.logging import stage_record, get_logger, log_stage


def test_stage_record_shape():
    r = stage_record("processed", "j1", "faces", "ok", 0)
    assert r == {"event": "processed", "job_id": "j1", "stage": "faces",
                 "status": "ok", "attempts": 0}

def test_log_stage_emits_json_line():
    logger = logging.getLogger("celebvision.test")
    logger.handlers.clear()
    buf = StringIO()
    h = logging.StreamHandler(buf)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    log_stage(logger, "dlq", "j2", "ingest", "dlq", 3)
    parsed = json.loads(buf.getvalue().strip())
    assert parsed["event"] == "dlq" and parsed["attempts"] == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_logging.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the logging module**

```python
# src/celebvision/logging.py
import json
import logging
import sys


def stage_record(event: str, job_id: str, stage: str, status: str,
                 attempts: int) -> dict:
    return {"event": event, "job_id": job_id, "stage": stage,
            "status": status, "attempts": attempts}


def get_logger(name: str = "celebvision") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_stage(logger: logging.Logger, event: str, job_id: str, stage: str,
              status: str, attempts: int) -> None:
    logger.info(json.dumps(stage_record(event, job_id, stage, status, attempts)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_logging.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/logging.py tests/test_logging.py
git commit -m "feat: add structured stage logging"
```

---

### Task 4: Worker base loop — retries, DLQ, metrics, logging; DLQ topics; metrics port

**Files:**
- Modify: `src/celebvision/workers/base.py`
- Modify: `scripts/init_stack.py`
- Modify: `src/celebvision/workers/__main__.py`
- Modify: `tests/test_worker_base.py`

**Interfaces:**
- Consumes: `Message`/`MessageBus` (with `attempts`), `WorkerContext` (with `.settings.max_attempts`), `celebvision.metrics`, `celebvision.logging`, `StageError`.
- Produces: `run_worker(stage, bus, ctx, handler)` with retry/DLQ + metrics + logging; `init_stack` creates `<stage>.dlq` topics; `workers/__main__.main` starts the metrics HTTP server.

- [ ] **Step 1: Write failing tests** (fake bus/db; assert retry re-produce, DLQ routing, attempts increment)

```python
# replace the body of tests/test_worker_base.py with this (keeps the two prior cases, adds retry/dlq)
import pytest
from celebvision.bus import Message
from celebvision.errors import StageError
from celebvision.workers.base import run_worker, WorkerContext
from celebvision.config import Settings


class FakeBus:
    def __init__(self, messages):
        self._messages = messages
        self.produced = []
    async def produce(self, topic, key, message):
        self.produced.append((topic, message))
    async def stream(self, topics, group):
        for m in self._messages:
            yield m


class FakeDB:
    def __init__(self):
        self.errors = []
    async def set_job_error(self, job_id, error):
        self.errors.append((job_id, error))


def _ctx(db, max_attempts=3):
    return WorkerContext(db=db, storage=None, inference=None, llm=None,
                         settings=Settings.from_env({"MAX_ATTEMPTS": str(max_attempts)}))


async def test_success_emits_completion_event():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1)])
    async def handler(msg, ctx):
        return None
    await run_worker("faces", bus, _ctx(FakeDB()), handler)
    assert bus.produced == [("stage.events",
                             Message(job_id="j", stage="faces", scene_id=1))]


async def test_stage_error_retries_to_requested_with_incremented_attempts():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1, attempts=0)])
    async def handler(msg, ctx):
        raise StageError("faces", "boom")
    await run_worker("faces", bus, _ctx(FakeDB(), max_attempts=3), handler)
    assert len(bus.produced) == 1
    topic, msg = bus.produced[0]
    assert topic == "faces.requested"
    assert msg.attempts == 1


async def test_stage_error_routes_to_dlq_when_attempts_exhausted():
    bus = FakeBus([Message(job_id="j", stage="ingest", attempts=2)])
    db = FakeDB()
    async def handler(msg, ctx):
        raise StageError("ingest", "boom")
    await run_worker("ingest", bus, _ctx(db, max_attempts=3), handler)
    topic, msg = bus.produced[0]
    assert topic == "ingest.dlq"
    assert msg.attempts == 3
    assert db.errors and "dlq after 3 attempts" in db.errors[0][1]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_worker_base.py -v`
Expected: FAIL (retry/dlq behavior not implemented).

- [ ] **Step 3: Rewrite `run_worker`** in `src/celebvision/workers/base.py`

Keep the `WorkerContext` dataclass. Replace `run_worker` with:
```python
import time
from celebvision import metrics
from celebvision.logging import get_logger, log_stage

_LOG = get_logger()


async def run_worker(stage: str, bus: MessageBus, ctx: WorkerContext,
                     handler: Handler) -> None:
    async for msg in bus.stream([f"{stage}.requested"], group=stage):
        start = time.monotonic()
        try:
            await handler(msg, ctx)
        except StageError as e:
            metrics.record_failed(stage)
            attempts = msg.attempts + 1
            if attempts < ctx.settings.max_attempts:
                metrics.record_retried(stage)
                log_stage(_LOG, "retry", msg.job_id, stage, "retry", attempts)
                await bus.produce(f"{stage}.requested", msg.job_id, Message(
                    job_id=msg.job_id, stage=stage, scene_id=msg.scene_id,
                    payload=msg.payload, attempts=attempts))
            else:
                metrics.record_dlq(stage)
                log_stage(_LOG, "dlq", msg.job_id, stage, "dlq", attempts)
                await bus.produce(f"{stage}.dlq", msg.job_id, Message(
                    job_id=msg.job_id, stage=stage, scene_id=msg.scene_id,
                    payload=msg.payload, attempts=attempts))
                await ctx.db.set_job_error(
                    msg.job_id, f"{stage}: {e.reason} (dlq after {attempts} attempts)")
            continue
        metrics.observe_duration(stage, time.monotonic() - start)
        metrics.record_processed(stage)
        log_stage(_LOG, "processed", msg.job_id, stage, "ok", msg.attempts)
        await bus.produce("stage.events", msg.job_id, Message(
            job_id=msg.job_id, stage=stage, scene_id=msg.scene_id))
```
(Keep the existing imports at the top of the file; add `time`, `metrics`, and the logging imports.)

- [ ] **Step 4: Add DLQ topics to `scripts/init_stack.py`**

Extend the `TOPICS` list to include the six DLQ topics:
```python
TOPICS = ["ingest.requested", "scenes.requested", "transcribe.requested",
          "faces.requested", "mentions.requested", "aggregate.requested",
          "stage.events",
          "ingest.dlq", "scenes.dlq", "transcribe.dlq",
          "faces.dlq", "mentions.dlq", "aggregate.dlq"]
```

- [ ] **Step 5: Start the metrics server in the worker entrypoint** (`src/celebvision/workers/__main__.py`)

Add imports at the top of `workers/__main__.py`:
```python
from prometheus_client import start_http_server
from celebvision.metrics import REGISTRY as METRICS_REGISTRY
```
In `main`, after building `bus`/`db`/`ctx` and before `run_worker`, add:
```python
    start_http_server(settings.metrics_port, registry=METRICS_REGISTRY)
```
(Serves the celebvision metrics registry on `settings.metrics_port`.)

- [ ] **Step 6: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/test_worker_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add src/celebvision/workers/base.py scripts/init_stack.py src/celebvision/workers/__main__.py tests/test_worker_base.py
git commit -m "feat: worker retries, DLQ routing, metrics and logging"
```

---

### Task 5: API `/metrics` endpoint

**Files:**
- Modify: `src/celebvision/api/app.py`
- Create: `tests/integration/test_api_metrics.py`

**Interfaces:**
- Consumes: `celebvision.metrics`.
- Produces: `GET /metrics` → `Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)`.

- [ ] **Step 1: Write failing integration test** (requires_stack — reuses the app lifespan)

```python
# tests/integration/test_api_metrics.py
import pytest
from httpx import AsyncClient, ASGITransport
from celebvision.api.app import create_app

pytestmark = pytest.mark.requires_stack


async def test_metrics_endpoint_returns_prometheus_text():
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/metrics")
            assert r.status_code == 200
            assert "celebvision_stage_processed_total" in r.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest tests/integration/test_api_metrics.py -v`
Expected: FAIL (404 — no `/metrics` route). Without stack: skipped.

- [ ] **Step 3: Add the route** to `src/celebvision/api/app.py`

Add import at top: `from fastapi import Response` (extend the existing fastapi import) and `from celebvision import metrics`. Inside `create_app`, add a route:
```python
    @app.get("/metrics")
    async def get_metrics():
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)
```

- [ ] **Step 4: Run to verify it passes** (stack up)

Run: `CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest tests/integration/test_api_metrics.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/api/app.py tests/integration/test_api_metrics.py
git commit -m "feat: expose prometheus /metrics on the API"
```

---

### Task 6: DLQ integration test

**Files:**
- Create: `tests/integration/test_dlq.py`

**Interfaces:**
- Consumes: `MessageBus`, `run_worker`, `WorkerContext`, `Database`, `StageError`.
- Produces: an integration test proving a persistently-failing handler lands on `<stage>.dlq` after `max_attempts` and the job is `failed`.

- [ ] **Step 1: Write the failing/absent test** (requires_stack)

```python
# tests/integration/test_dlq.py
import uuid
import pytest
from celebvision.bus import MessageBus, Message
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.workers.base import run_worker, WorkerContext
from celebvision.errors import StageError

pytestmark = pytest.mark.requires_stack


async def test_failing_handler_lands_on_dlq_and_marks_job_failed():
    s = Settings.from_env({"MAX_ATTEMPTS": "1"})   # 1 => straight to DLQ on first failure
    db = Database(s.postgres_dsn)
    await db.connect()
    await db.apply_schema()
    bus = MessageBus(s.kafka_bootstrap)
    await bus.start()
    try:
        jid = str(uuid.uuid4())
        await db.create_job(jid, "file", "/x.mp4", "wl1", [])
        # produce one request that the handler will always reject
        await bus.produce("ingest.requested", jid, Message(job_id=jid, stage="ingest"))

        ctx = WorkerContext(db=db, storage=None, inference=None, llm=None, settings=s)

        async def failing(msg, ctx):
            raise StageError("ingest", "always fails")

        # Run the real worker loop as a background task; it consumes the request,
        # fails it, and (MAX_ATTEMPTS=1) dead-letters immediately.
        import asyncio
        task = asyncio.create_task(run_worker("ingest", bus, ctx, failing))
        # wait until the job is marked failed (bounded)
        for _ in range(40):
            row = await db.get_job(jid)
            if row["status"] == "failed":
                break
            await asyncio.sleep(0.25)
        task.cancel()
        row = await db.get_job(jid)
        assert row["status"] == "failed"
        assert "dlq after 1 attempts" in row["error"]

        # confirm the message is on ingest.dlq
        seen = False
        count = 0
        async for m in bus.stream(["ingest.dlq"], group=f"dlq-verify-{uuid.uuid4().hex[:8]}"):
            count += 1
            if m.job_id == jid:
                seen = True
                break
            if count >= 1000:
                break
        assert seen
    finally:
        await bus.stop()
        await db.close()
```

- [ ] **Step 2: Run to verify it fails (before Task 4) / passes (after Task 4)**

Since Task 4 already implemented retry/DLQ, this test validates it end-to-end. Run (stack up, topics created via `init_stack`):
`CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest tests/integration/test_dlq.py -v`
Expected: PASS (1 passed). If `ingest.dlq` doesn't exist, run `scripts/init_stack.py` first (with the same env).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_dlq.py
git commit -m "test: DLQ integration — failing handler dead-letters and fails the job"
```

---

### Task 7: Eval scorer — models + `score_report`

**Files:**
- Create: `src/celebvision/eval.py`
- Create: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Report` (M1 models).
- Produces (`celebvision.eval`):
  - `GroundTruth(BaseModel)`: `job_id: str`, `expected: list[str]`, `expected_by_modality: dict[str, list[str]] = {}`, `expected_by_scene: dict[int, list[str]] = {}`.
  - `ModalityScore(BaseModel)`: `precision: float`, `recall: float`, `f1: float`, `tp: int`, `fp: int`, `fn: int`.
  - `EvalResult(BaseModel)`: `job_id: str`, `video: dict[str, ModalityScore]`, `scene_mean_f1: float | None`.
  - `score_report(report: Report, gt: GroundTruth) -> EvalResult`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval.py
from celebvision.eval import GroundTruth, score_report
from celebvision.models import (
    Report, SceneReport, JobSource, SpokenMention, OnscreenFace, CelebrityIndexEntry,
)


def _report():
    scene = SceneReport(
        scene_id=0, start_s=0, end_s=5, keyframes=[], transcript="",
        spoken_mentions=[SpokenMention(name="Messi", canonical_id="messi",
                                       confidence=0.9, evidence_span="")],
        keyword_hits=[],
        onscreen_faces=[OnscreenFace(name="Ronaldo", canonical_id="ronaldo",
                                     confidence=0.8, bbox=(0, 0, 1, 1), keyframe="")],
    )
    return Report(job_id="j1", source=JobSource(kind="file", locator="a.mp4"),
                  duration_s=5.0, watchlist_id="w1", completed_at="",
                  scenes=[scene],
                  celebrity_index=[
                      CelebrityIndexEntry(name="Messi", canonical_id="messi",
                                          scenes=[0], modalities=["audio"]),
                      CelebrityIndexEntry(name="Ronaldo", canonical_id="ronaldo",
                                          scenes=[0], modalities=["face"])])


def test_score_report_combined_prf():
    # expected {messi, ronaldo, neymar}; detected combined {messi, ronaldo}
    gt = GroundTruth(job_id="j1", expected=["messi", "ronaldo", "neymar"])
    res = score_report(_report(), gt)
    c = res.video["combined"]
    assert c.tp == 2 and c.fp == 0 and c.fn == 1
    assert abs(c.precision - 1.0) < 1e-9
    assert abs(c.recall - 2/3) < 1e-9

def test_score_report_per_modality():
    gt = GroundTruth(job_id="j1", expected=["messi", "ronaldo"],
                     expected_by_modality={"audio": ["messi"], "face": ["ronaldo"]})
    res = score_report(_report(), gt)
    assert res.video["audio"].f1 == 1.0   # detected audio {messi} == expected {messi}
    assert res.video["face"].f1 == 1.0     # detected face {ronaldo} == expected {ronaldo}

def test_empty_expected_and_detected_is_zero_not_crash():
    gt = GroundTruth(job_id="j1", expected=[])
    from celebvision.models import Report as R, JobSource as JS
    empty = R(job_id="j1", source=JS(kind="file", locator="a"), duration_s=0,
              watchlist_id="w", completed_at="", scenes=[], celebrity_index=[])
    res = score_report(empty, gt)
    assert res.video["combined"].f1 == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the eval module**

```python
# src/celebvision/eval.py
from pydantic import BaseModel
from celebvision.models import Report


class GroundTruth(BaseModel):
    job_id: str
    expected: list[str]
    expected_by_modality: dict[str, list[str]] = {}
    expected_by_scene: dict[int, list[str]] = {}


class ModalityScore(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class EvalResult(BaseModel):
    job_id: str
    video: dict[str, ModalityScore]
    scene_mean_f1: float | None


def _prf(expected: set[str], detected: set[str]) -> ModalityScore:
    tp = len(expected & detected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ModalityScore(precision=precision, recall=recall, f1=f1,
                         tp=tp, fp=fp, fn=fn)


def _expected_for(gt: GroundTruth, modality: str) -> set[str]:
    return set(gt.expected_by_modality.get(modality, gt.expected))


def score_report(report: Report, gt: GroundTruth) -> EvalResult:
    audio = {m.canonical_id for s in report.scenes for m in s.spoken_mentions}
    face = {f.canonical_id for s in report.scenes for f in s.onscreen_faces}
    combined = {e.canonical_id for e in report.celebrity_index}

    video = {
        "audio": _prf(_expected_for(gt, "audio"), audio),
        "face": _prf(_expected_for(gt, "face"), face),
        "combined": _prf(set(gt.expected), combined),
    }

    scene_mean_f1 = None
    if gt.expected_by_scene:
        by_scene = {s.scene_id: {e.canonical_id for e in s.spoken_mentions}
                    | {f.canonical_id for f in s.onscreen_faces}
                    for s in report.scenes}
        f1s = []
        for sid, exp in gt.expected_by_scene.items():
            if sid in by_scene:
                f1s.append(_prf(set(exp), by_scene[sid]).f1)
        scene_mean_f1 = sum(f1s) / len(f1s) if f1s else None

    return EvalResult(job_id=report.job_id, video=video, scene_mean_f1=scene_mean_f1)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/eval.py tests/test_eval.py
git commit -m "feat: add precision/recall eval scorer"
```

---

### Task 8: `celebvision eval` CLI command

**Files:**
- Modify: `src/celebvision/cli.py`
- Create: `tests/test_cli_eval.py`

**Interfaces:**
- Consumes: `score_report`, `GroundTruth`, `Report`.
- Produces: `celebvision eval --report <path> --truth <path> --out <path>` → writes `EvalResult` JSON, echoes combined F1.

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli_eval.py
import json
from typer.testing import CliRunner
from celebvision.cli import app

runner = CliRunner()


def test_eval_command_writes_result(tmp_path):
    report = {
        "job_id": "j1", "source": {"kind": "file", "locator": "a.mp4"},
        "duration_s": 5.0, "watchlist_id": "w1", "completed_at": "",
        "scenes": [], "celebrity_index": [
            {"name": "Messi", "canonical_id": "messi", "scenes": [0],
             "modalities": ["audio"]}]}
    truth = {"job_id": "j1", "expected": ["messi", "ronaldo"]}
    rp = tmp_path / "report.json"; rp.write_text(json.dumps(report))
    tp = tmp_path / "truth.json"; tp.write_text(json.dumps(truth))
    out = tmp_path / "eval.json"
    result = runner.invoke(app, ["eval", "--report", str(rp), "--truth", str(tp),
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written["job_id"] == "j1"
    # detected combined {messi}; expected {messi, ronaldo} -> recall 0.5, precision 1.0
    assert abs(written["video"]["combined"]["recall"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_eval.py -v`
Expected: FAIL — no `eval` command (exit code != 0).

- [ ] **Step 3: Add the `eval` command** to `src/celebvision/cli.py`

Add imports near the top:
```python
from celebvision.models import Report
from celebvision.eval import GroundTruth, score_report
```
Add the command:
```python
@app.command()
def eval(
    report: str = typer.Option(..., help="Path to a report.json"),
    truth: str = typer.Option(..., help="Path to a ground-truth.json"),
    out: str = typer.Option("eval.json", help="Output EvalResult path"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    with open(truth) as f:
        gt = GroundTruth.model_validate_json(f.read())
    result = score_report(rep, gt)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))
    typer.echo(f"combined F1={result.video['combined'].f1:.3f} -> {out}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_cli_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/cli.py tests/test_cli_eval.py
git commit -m "feat: add celebvision eval CLI command"
```

---

### Task 9: Aggregate report `duration_s` + `completed_at`

**Files:**
- Modify: `src/celebvision/workers/aggregate.py`
- Modify: `tests/integration/test_downstream_workers.py`

**Interfaces:**
- Consumes: `Transcript` (to read `duration_s`), `job_assets.transcript_key`, `BlobStore`.
- Produces: the aggregate worker populates `Report.duration_s` (from the transcript) and `Report.completed_at` (wall-clock at build).

- [ ] **Step 1: Update the downstream integration test to assert the fields**

In `tests/integration/test_downstream_workers.py`, after the `handle_aggregate` call and report load, add assertions:
```python
        assert report["completed_at"] != ""
        assert report["duration_s"] > 0.0
```
(The stub transcript spans ~8 s, so `duration_s` should be > 0.)

- [ ] **Step 2: Run to verify it fails** (stack up)

Run: `CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest tests/integration/test_downstream_workers.py -v`
Expected: FAIL — `duration_s` is `0.0` / `completed_at` is `""` (current M2 behavior).

- [ ] **Step 3: Populate the fields in `handle_aggregate`** (`src/celebvision/workers/aggregate.py`)

Add imports:
```python
from datetime import datetime, timezone
from celebvision.interfaces import Transcript
```
Before `build_report`, compute duration from the transcript asset (falls back to 0.0):
```python
        duration_s = 0.0
        assets = await ctx.db.get_job_assets(msg.job_id)
        if assets and assets.get("transcript_key"):
            raw = await ctx.storage.get_bytes("media", assets["transcript_key"])
            duration_s = Transcript.model_validate_json(raw).duration_s
        completed_at = datetime.now(timezone.utc).isoformat()
```
Change the `build_report(...)` call from `..., "", 0.0, scene_reports)` to:
```python
        report = build_report(
            msg.job_id, JobSource(kind=job["source_kind"], locator=job["source_locator"]),
            job["watchlist_id"], completed_at, duration_s, scene_reports)
```
(`get_job_assets` returns a dict; use `.get("transcript_key")` defensively.)

- [ ] **Step 4: Run to verify pass** (stack up)

Run: `CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest tests/integration/test_downstream_workers.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/workers/aggregate.py tests/integration/test_downstream_workers.py
git commit -m "feat: populate report duration_s and completed_at in aggregate"
```

---

### Task 10: TritonFaceClient — metadata name discovery, READY check, zero-norm raise

**Files:**
- Modify: `src/celebvision/inference/triton_client.py`
- Modify: `tests/test_triton_client_unit.py`
- Modify: `tests/integration/test_triton_e2e.py`

**Interfaces:**
- Consumes: tritonclient metadata API, `StageError`.
- Produces: `TritonFaceClient` resolves input/output tensor names from Triton model metadata (cached) instead of hard-coding, verifies the model is READY, and `_embed` raises on zero-norm.

- [ ] **Step 1: Write failing unit test for zero-norm raise** (pure; injected infer returns zeros)

```python
# append to tests/test_triton_client_unit.py
import numpy as np
import pytest
from celebvision.inference.triton_client import TritonFaceClient
from celebvision.config import Settings
from celebvision.errors import StageError


def test_embed_raises_on_zero_norm():
    client = TritonFaceClient(Settings.from_env({}),
                              infer_fn=lambda blob: np.zeros((blob.shape[0], 512),
                                                             dtype=np.float32))
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    with pytest.raises(StageError):
        client._embed([crop])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_triton_client_unit.py::test_embed_raises_on_zero_norm -v`
Expected: FAIL (currently returns the unnormalized zero vector).

- [ ] **Step 3: Update `triton_client.py`**

In `_embed`, replace the zero-norm fallback:
```python
        out = []
        for row in feats:
            n = float(np.linalg.norm(row))
            if n <= 0.0:
                raise StageError("face", "zero-norm embedding from Triton")
            out.append((row / n).tolist())
        return out
```
Add metadata-driven name resolution + READY check to the tritonclient path. Add fields in `__init__`: `self._in_name = None; self._out_name = None`. Replace `_triton_infer`:
```python
    def _ensure_ready(self, client):
        if not client.is_model_ready(self._settings.triton_model):
            raise StageError("face", f"triton model {self._settings.triton_model} not ready")
        if self._in_name is None:
            meta = client.get_model_metadata(self._settings.triton_model)
            self._in_name = meta["inputs"][0]["name"]
            self._out_name = meta["outputs"][0]["name"]

    def _triton_infer(self, blob):
        import tritonclient.http as httpclient
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self._settings.triton_url)
        self._ensure_ready(self._client)
        inp = httpclient.InferInput(self._in_name, blob.shape, "FP32")
        inp.set_data_from_numpy(blob)
        out = httpclient.InferRequestedOutput(self._out_name)
        resp = self._client.infer(self._settings.triton_model, inputs=[inp], outputs=[out])
        return resp.as_numpy(self._out_name)
```
(`get_model_metadata` returns a dict for the HTTP client; `inputs[0]["name"]`/`outputs[0]["name"]` are the tensor names. The injected `infer_fn` path is unchanged and does not touch metadata.)

- [ ] **Step 4: Run the unit test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_triton_client_unit.py -v`
Expected: PASS (the mapping test from M3 + the new zero-norm test).

- [ ] **Step 5: Confirm the live-Triton E2E still passes with metadata discovery** (Triton up)

The M3 `tests/integration/test_triton_e2e.py` already exercises `_embed` via real Triton; with metadata discovery it now resolves names dynamically. Run (Triton up):
`TRITON_URL=localhost:8000 .venv/bin/python -m pytest tests/integration/test_triton_e2e.py -v`
Expected: PASS (1 passed). (No test-code change needed; if the E2E currently constructs the client without a reachable model, add nothing — it was already skipif-guarded.)

- [ ] **Step 6: Commit**

```bash
git add src/celebvision/inference/triton_client.py tests/test_triton_client_unit.py tests/integration/test_triton_e2e.py
git commit -m "feat: Triton metadata name discovery, READY check, zero-norm raise"
```

---

### Task 11: Streaming blob storage

**Files:**
- Modify: `src/celebvision/storage.py`
- Create: `tests/integration/test_storage_streaming.py`

**Interfaces:**
- Produces: `BlobStore.put_file`/`get_file` stream via aioboto3 `upload_fileobj`/`download_fileobj` (no whole-file in-memory read). Signatures unchanged.

- [ ] **Step 1: Write failing/added integration test** (requires_stack; a multi-MB file round-trips)

```python
# tests/integration/test_storage_streaming.py
import os
import pytest
from celebvision.storage import BlobStore
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_put_get_file_streams_large_file(tmp_path):
    s = Settings.from_env()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    await store.ensure_bucket("test-bucket")
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(5 * 1024 * 1024))   # 5 MB
    await store.put_file("test-bucket", "big.bin", str(src))
    dst = tmp_path / "out.bin"
    await store.get_file("test-bucket", "big.bin", str(dst))
    assert dst.read_bytes() == src.read_bytes()
```

- [ ] **Step 2: Run — should pass against current impl (buffered) too**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_storage_streaming.py -v`
Expected: PASS with the current buffered implementation (this test guards the round-trip; the change below makes it streaming without breaking it). Confirm it passes now, then refactor to streaming and confirm it still passes.

- [ ] **Step 3: Refactor `put_file`/`get_file` to stream** (`src/celebvision/storage.py`)

```python
    async def put_file(self, bucket: str, key: str, path: str) -> str:
        async with self._client() as s3:
            with open(path, "rb") as f:
                await s3.upload_fileobj(f, bucket, key)
        return f"{bucket}/{key}"

    async def get_file(self, bucket: str, key: str, dest: str) -> str:
        async with self._client() as s3:
            with open(dest, "wb") as f:
                await s3.download_fileobj(bucket, key, f)
        return dest
```
(`put_bytes`/`get_bytes` unchanged.)

- [ ] **Step 4: Run to verify still passes** (stack up)

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_storage_streaming.py tests/integration/test_storage.py -v`
Expected: PASS (streaming round-trip + the original bytes round-trip).

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/storage.py tests/integration/test_storage_streaming.py
git commit -m "perf: stream file put/get in blob storage"
```

---

### Task 12: Prep `--kind` + GPU Triton config

**Files:**
- Modify: `scripts/prepare_triton_models.py`
- Modify: `docker-compose.yml`
- Modify: `tests/test_prepare_triton_models.py`

**Interfaces:**
- Consumes: `render_config_pbtxt` (already has a `kind` param).
- Produces: `prepare(model_repo, cache_dir=None, kind="KIND_CPU")` writes `config.pbtxt` with the given kind; `__main__` accepts `--kind`; the gpu compose profile mounts a `KIND_GPU` config.

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_prepare_triton_models.py
def test_render_config_pbtxt_gpu_kind():
    from scripts.prepare_triton_models import render_config_pbtxt
    txt = render_config_pbtxt("arcface", "input.1", [3, 112, 112], "683", [512],
                              kind="KIND_GPU")
    assert "KIND_GPU" in txt
    assert "KIND_CPU" not in txt
```

- [ ] **Step 2: Run to verify it passes-or-fails**

Run: `.venv/bin/python -m pytest tests/test_prepare_triton_models.py::test_render_config_pbtxt_gpu_kind -v`
Expected: PASS already (the renderer took `kind` since M3). If it passes, this test simply locks the behavior — proceed to wire `--kind` through `prepare`/`__main__` (Steps 3-4) which is the actual deliverable, then re-run the full file.

- [ ] **Step 3: Thread `kind` through `prepare` + add `--kind` CLI** (`scripts/prepare_triton_models.py`)

Change `prepare` signature + config write:
```python
def prepare(model_repo="model_repository", cache_dir=None, kind="KIND_CPU"):
    ...
    config_path.write_text(
        render_config_pbtxt("arcface", in_name, in_dims, out_name, out_dims, kind=kind))
    return str(model_path), str(config_path)
```
Update `__main__` to accept `--kind` (simple argv parse, no new dep):
```python
if __name__ == "__main__":
    import sys
    kind = "KIND_CPU"
    if "--kind" in sys.argv:
        kind = sys.argv[sys.argv.index("--kind") + 1]
    mp, cp = prepare(kind=kind)
    print(f"model: {mp}\nconfig: {cp}")
```

- [ ] **Step 4: Document GPU config in compose** (`docker-compose.yml`)

On the `triton-gpu` service (added in M3), add a comment + an env note that its model repo should be generated with `KIND_GPU`. Since the model repo is a mounted volume, add to `triton-gpu`:
```yaml
    # Generate the GPU model config before `--profile gpu up`:
    #   .venv/bin/python scripts/prepare_triton_models.py --kind KIND_GPU
```
(No structural compose change is required beyond this documentation line; `docker compose config` must still parse.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_triton_models.py -v` (renderer + gpu-kind; the `-m slow` prep tests still pass) and `.venv/bin/python -m pytest tests/test_compose_config.py -v`.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_triton_models.py docker-compose.yml tests/test_prepare_triton_models.py
git commit -m "feat: --kind flag and GPU Triton config generation"
```

---

### Task 13: M4 runbook + milestone regression gate

**Files:**
- Create: `docs/M4_RUNBOOK.md`

**Interfaces:**
- Consumes: everything.
- Produces: the runbook + a verified full regression gate.

- [ ] **Step 1: Write the runbook**

```markdown
# docs/M4_RUNBOOK.md

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
```

- [ ] **Step 2: Run the full regression gate** (infra up: Redpanda:19092, MinIO:9000, Postgres:55432; Triton:8000 for the live test)

Seed topics/schema/buckets (includes the new DLQ topics):
`CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python scripts/init_stack.py`
Then:
1. Fast unit (no stack): `.venv/bin/python -m pytest -q` → green.
2. Ruff: `.venv/bin/ruff check src tests scripts` → clean.
3. Slow model tests: `.venv/bin/python -m pytest -m slow -v` → green.
4. Full stack regression: `CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb .venv/bin/python -m pytest -q` → all green (M1/M2/M3 + M4 integration incl. DLQ, /metrics, streaming, aggregate fields).
5. Live Triton (up): `TRITON_URL=localhost:8000 .venv/bin/python -m pytest tests/integration/test_triton_e2e.py -v` → 1 passed.
Expected: all green; ruff clean.

- [ ] **Step 3: Commit**

```bash
git add docs/M4_RUNBOOK.md
git commit -m "docs: add M4 runbook and record regression gate"
```

---

## Milestone exit criteria

- [ ] Retries + DLQ: `test_dlq.py` proves a persistently-failing message dead-letters after `max_attempts` and the job is `failed`.
- [ ] `/metrics` returns Prometheus text (API); metrics module unit-tested via `CollectorRegistry`.
- [ ] `celebvision eval` writes an `EvalResult`; scorer unit-tested (P/R/F1 on fixtures).
- [ ] Report `duration_s`/`completed_at` populated (downstream integration test asserts it).
- [ ] TritonFaceClient metadata discovery + READY + zero-norm raise (unit + live-Triton green).
- [ ] Streaming storage round-trips a multi-MB file.
- [ ] `prepare_triton_models.py --kind KIND_GPU` renders a GPU config; compose parses.
- [ ] Full M1/M2/M3 regression green; ruff clean; DAG/stage/schema unchanged.

## Handoff — GPU-host milestone

Remaining: ASR-on-Triton (Parakeet/Canary), TensorRT/FP16 conversion, and validating `KIND_GPU`
ArcFace serving on real NVIDIA hardware.
