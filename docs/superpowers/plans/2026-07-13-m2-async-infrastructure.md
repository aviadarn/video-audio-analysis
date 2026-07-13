# M2: Async Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the M1 in-process pipeline into a horizontally-scalable async service — submit a job over HTTP, process it through Kafka(Redpanda)-staged workers that reuse the M1 stage functions, retrieve a per-scene JSON report — with state in Postgres, media/reports in MinIO, watchlist in pgvector.

**Architecture:** Fully async (aiokafka + async FastAPI + asyncpg + aioboto3). Six stage workers each consume a `<stage>.requested` topic, do work via injected M1 clients, persist to Postgres/MinIO, and emit a completion event; a Coordinator consumes completions and emits next-stage requests per a DAG driven by Postgres state. **Infra runs in Docker; app code runs in-process during tests** (with a deterministic `stub` inference backend) against the real infra.

**Tech Stack:** Python 3.11, asyncio, aiokafka, asyncpg, aioboto3 (MinIO/S3), pgvector, FastAPI, uvicorn, httpx, pytest-asyncio, Redpanda, Postgres+pgvector, MinIO, docker-compose.

## Global Constraints

- Python **3.11+**; use the existing `.venv` (`python3.13`). `python`/`pip` mean `.venv/bin/python` / `.venv/bin/pip`.
- Builds on merged M1. **Do not change M1 stage logic** (`celebvision.stages.*`, `celebvision.interfaces`, `celebvision.models`) except adding `media.probe_duration` (Task 9). All workers call M1 stage functions via the M1 protocols.
- **Fully async**: aiokafka, asyncpg, aioboto3, async FastAPI. No blocking I/O in async paths (wrap unavoidable sync calls — ffmpeg/scenedetect — with `asyncio.to_thread`).
- **Broker**: Redpanda locally (Kafka API); config via `KAFKA_BOOTSTRAP` (default `localhost:19092`).
- **DB**: raw asyncpg + `sql/schema.sql` applied idempotently; pgvector via `<=>`; `POSTGRES_DSN` (default `postgresql://celeb:celeb@localhost:5432/celeb`).
- **Storage**: MinIO via aioboto3; `MINIO_ENDPOINT` (default `http://localhost:9000`), `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` (default `minioadmin`/`minioadmin`).
- **Backends** selectable by env: `INFERENCE_BACKEND ∈ {stub, local}` (default `stub` for M2; `triton` added M3), `LLM_BACKEND ∈ {stub, anthropic}` (default `stub` for M2), `WATCHLIST_BACKEND ∈ {pg, local}` (default `pg`).
- **Error handling**: typed `StageError(stage, reason)`; a worker catch marks the job `failed` and commits the offset (no poison-loop). Robust retries + DLQ are deferred to M4.
- **Status values**: `queued | running | failed | done`.
- **Message envelope** (JSON): `{job_id, stage, scene_id?, payload}`. Payloads carry references (ids, MinIO keys), never blobs.
- **Tests touching infra** are marked `@pytest.mark.requires_stack` and are skipped unless env `CELEBVISION_STACK=1`. Pure-logic tests (no infra) always run. Default `pytest` (no stack) must stay green.
- **Host prerequisites for stack tests**: `docker compose -f docker-compose.yml up -d redpanda postgres minio init` and `brew install ffmpeg`, then `CELEBVISION_STACK=1 .venv/bin/python -m pytest -m requires_stack`.
- Lint/format: **ruff**. Commit after every task (Conventional Commits).
- Timestamps float seconds; embeddings 512-dim (ArcFace).

---

### Task 1: M2 dependencies, config, errors, test markers

**Files:**
- Modify: `pyproject.toml`
- Create: `src/celebvision/config.py`
- Create: `src/celebvision/errors.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `celebvision.errors.StageError(stage: str, reason: str)` — `Exception`; `str(e) == f"{stage}: {reason}"`; attrs `.stage`, `.reason`.
  - `celebvision.config.Settings` — frozen dataclass; classmethod `from_env(env: Mapping[str,str] | None = None) -> Settings`. Fields (all str unless noted): `kafka_bootstrap`, `postgres_dsn`, `minio_endpoint`, `minio_access_key`, `minio_secret_key`, `inference_backend`, `llm_backend`, `watchlist_backend`, `whisper_model`, `face_model`, `watchlist_path` (for local backend). Defaults per Global Constraints.
  - pytest marker `requires_stack`; a session fixture that skips those tests unless `CELEBVISION_STACK=1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from celebvision.config import Settings
from celebvision.errors import StageError


def test_settings_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s.kafka_bootstrap == "localhost:19092"
    assert s.postgres_dsn.startswith("postgresql://")
    assert s.inference_backend == "stub"
    assert s.llm_backend == "stub"
    assert s.watchlist_backend == "pg"

def test_settings_reads_env_overrides():
    s = Settings.from_env({"KAFKA_BOOTSTRAP": "broker:9092",
                           "INFERENCE_BACKEND": "local"})
    assert s.kafka_bootstrap == "broker:9092"
    assert s.inference_backend == "local"

def test_stage_error_str():
    e = StageError("ingest", "download failed")
    assert str(e) == "ingest: download failed"
    assert e.stage == "ingest"
    assert e.reason == "download failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.config'`

- [ ] **Step 3: Write errors and config**

```python
# src/celebvision/errors.py
class StageError(Exception):
    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage}: {reason}")
```

```python
# src/celebvision/config.py
from collections.abc import Mapping
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = "localhost:19092"
    postgres_dsn: str = "postgresql://celeb:celeb@localhost:5432/celeb"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    inference_backend: str = "stub"
    llm_backend: str = "stub"
    watchlist_backend: str = "pg"
    whisper_model: str = "small"
    face_model: str = "buffalo_l"
    watchlist_path: str = "./data/watchlist.npz"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        d = cls()
        return cls(
            kafka_bootstrap=e.get("KAFKA_BOOTSTRAP", d.kafka_bootstrap),
            postgres_dsn=e.get("POSTGRES_DSN", d.postgres_dsn),
            minio_endpoint=e.get("MINIO_ENDPOINT", d.minio_endpoint),
            minio_access_key=e.get("MINIO_ACCESS_KEY", d.minio_access_key),
            minio_secret_key=e.get("MINIO_SECRET_KEY", d.minio_secret_key),
            inference_backend=e.get("INFERENCE_BACKEND", d.inference_backend),
            llm_backend=e.get("LLM_BACKEND", d.llm_backend),
            watchlist_backend=e.get("WATCHLIST_BACKEND", d.watchlist_backend),
            whisper_model=e.get("WHISPER_MODEL", d.whisper_model),
            face_model=e.get("FACE_MODEL", d.face_model),
            watchlist_path=e.get("WATCHLIST_PATH", d.watchlist_path),
        )
```

- [ ] **Step 4: Update pyproject.toml (deps + markers + asyncio mode)**

Add to `[project.optional-dependencies]` (keep existing `dev` and `local`; append the two new extras and extend `dev`):

```toml
dev = ["pytest>=8.0", "ruff>=0.4", "pytest-asyncio>=0.23", "httpx>=0.27"]
service = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "aiokafka>=0.11",
    "asyncpg>=0.29",
    "aioboto3>=13.0",
    "pgvector>=0.3",
]
media = ["scenedetect>=0.6.3", "yt-dlp>=2024.4.9"]
```

Replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
markers = [
    "slow: requires real models or network (deselected by default)",
    "requires_stack: requires the docker infra stack (skipped unless CELEBVISION_STACK=1)",
]
addopts = "-m 'not slow'"
```

- [ ] **Step 5: Write conftest (skip requires_stack unless stack env set)**

```python
# tests/conftest.py
import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CELEBVISION_STACK") == "1":
        return
    skip = pytest.mark.skip(reason="needs docker stack; set CELEBVISION_STACK=1")
    for item in items:
        if "requires_stack" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 6: Install and run tests**

Run: `.venv/bin/pip install -e '.[dev,service,media]'` then `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (3 passed). Also `.venv/bin/python -m pytest -q` stays green (requires_stack tests skipped).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/celebvision/config.py src/celebvision/errors.py tests/conftest.py tests/test_config.py
git commit -m "feat: add M2 deps, settings, StageError, stack test marker"
```

---

### Task 2: Message bus (envelope + aiokafka transport)

**Files:**
- Create: `src/celebvision/bus.py`
- Create: `tests/test_bus_envelope.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_bus.py`

**Interfaces:**
- Consumes: nothing.
- Produces (`celebvision.bus`):
  - `Message(BaseModel)`: `job_id: str`, `stage: str`, `scene_id: int | None = None`, `payload: dict = {}`.
  - `encode(msg: Message) -> bytes`; `decode(raw: bytes) -> Message`.
  - `class MessageBus`: `__init__(self, bootstrap: str)`; `async start()`; `async stop()`; `async produce(self, topic: str, key: str, message: Message) -> None`; `async def stream(self, topics: list[str], group: str) -> AsyncIterator[Message]` (manual-commit AIOKafkaConsumer; commits after each message is yielded-and-processed).

- [ ] **Step 1: Write the failing envelope test**

```python
# tests/test_bus_envelope.py
from celebvision.bus import Message, encode, decode


def test_envelope_round_trip():
    m = Message(job_id="j1", stage="faces", scene_id=3, payload={"k": "v"})
    again = decode(encode(m))
    assert again.job_id == "j1"
    assert again.stage == "faces"
    assert again.scene_id == 3
    assert again.payload == {"k": "v"}

def test_envelope_scene_id_optional():
    m = decode(encode(Message(job_id="j1", stage="ingest")))
    assert m.scene_id is None
    assert m.payload == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bus_envelope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.bus'`

- [ ] **Step 3: Write the bus**

```python
# src/celebvision/bus.py
from collections.abc import AsyncIterator
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer


class Message(BaseModel):
    job_id: str
    stage: str
    scene_id: int | None = None
    payload: dict = {}


def encode(msg: Message) -> bytes:
    return msg.model_dump_json().encode("utf-8")


def decode(raw: bytes) -> Message:
    return Message.model_validate_json(raw)


class MessageBus:
    def __init__(self, bootstrap: str) -> None:
        self._bootstrap = bootstrap
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap)
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def produce(self, topic: str, key: str, message: Message) -> None:
        assert self._producer is not None, "call start() first"
        await self._producer.send_and_wait(topic, key=key.encode("utf-8"),
                                            value=encode(message))

    async def stream(self, topics: list[str], group: str) -> AsyncIterator[Message]:
        consumer = AIOKafkaConsumer(
            *topics, bootstrap_servers=self._bootstrap, group_id=group,
            enable_auto_commit=False, auto_offset_reset="earliest",
        )
        await consumer.start()
        try:
            async for record in consumer:
                yield decode(record.value)
                await consumer.commit()
        finally:
            await consumer.stop()
```

- [ ] **Step 4: Run envelope test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bus_envelope.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the integration test (requires_stack)**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_bus.py
import pytest
from celebvision.bus import MessageBus, Message
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_produce_and_consume_round_trip():
    bus = MessageBus(Settings.from_env().kafka_bootstrap)
    await bus.start()
    try:
        await bus.produce("test.bus", "k1",
                          Message(job_id="jb", stage="ingest", payload={"n": 1}))
        got = None
        async for msg in bus.stream(["test.bus"], group="test-bus-grp"):
            got = msg
            break
        assert got is not None
        assert got.job_id == "jb"
        assert got.payload == {"n": 1}
    finally:
        await bus.stop()
```

- [ ] **Step 6: Run integration test (skipped without stack) and commit**

Run (no stack): `.venv/bin/python -m pytest tests/integration/test_bus.py -v` → 1 skipped.
Run (with stack, if available): `docker compose up -d redpanda init && CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_bus.py -v` → 1 passed. (The `docker-compose.yml` + `init` land in Task 12; before then, this test only needs Redpanda reachable at `localhost:19092`. If the stack isn't up yet, the skip is expected and acceptable for this task's gate.)

```bash
git add src/celebvision/bus.py tests/test_bus_envelope.py tests/integration/
git commit -m "feat: add aiokafka message bus and JSON envelope"
```

---

### Task 3: Database layer + schema

**Files:**
- Create: `sql/schema.sql`
- Create: `src/celebvision/db.py`
- Create: `tests/integration/test_db.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces (`celebvision.db`): `class Database`:
  - `__init__(self, dsn: str)`; `async connect()` (creates asyncpg pool, registers pgvector on each connection); `async close()`; `async apply_schema()` (reads `sql/schema.sql` and executes it); property `pool`.
  - `async create_job(job_id, source_kind, source_locator, watchlist_id, keywords: list[str]) -> None` (status `queued`).
  - `async get_job(job_id) -> dict | None`.
  - `async set_job_status(job_id, status) -> None`; `async set_job_error(job_id, error) -> None` (status `failed`); `async set_job_completed(job_id) -> None` (status `done`, `completed_at=now()`).
  - `async set_expected_scene_count(job_id, n) -> None`.
  - `async upsert_job_assets(job_id, *, video_key=None, audio_key=None, transcript_key=None) -> None` (COALESCE keeps existing non-null values).
  - `async get_job_assets(job_id) -> dict | None`.
  - `async insert_scene(job_id, scene_id, start_s, end_s, keyframes: list[str]) -> None`.
  - `async get_scene(job_id, scene_id) -> dict | None`; `async get_scenes(job_id) -> list[dict]`.
  - `async set_scene_faces(job_id, scene_id, faces: list[dict]) -> None` (sets `faces` jsonb, `faces_status='done'`).
  - `async set_scene_mentions(job_id, scene_id, transcript: str, mentions: list[dict], keyword_hits: list[dict]) -> None` (sets `transcript`, `mentions` jsonb, `keyword_hits` jsonb, `mentions_status='done'`).
  - `async count_scenes_stage_done(job_id, stage: str) -> int` (stage `'faces'|'mentions'`).
  - `async mark_stage_done(job_id, stage: str) -> None`; `async is_stage_done(job_id, stage: str) -> bool` (job-level stages via `job_stages`).

- [ ] **Step 1: Write the schema**

```sql
-- sql/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id uuid PRIMARY KEY,
    source_kind text NOT NULL,
    source_locator text NOT NULL,
    watchlist_id text NOT NULL,
    keywords jsonb NOT NULL DEFAULT '[]',
    status text NOT NULL DEFAULT 'queued',
    error text,
    expected_scene_count int,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS job_assets (
    job_id uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    video_key text,
    audio_key text,
    transcript_key text
);

CREATE TABLE IF NOT EXISTS scenes (
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    scene_id int NOT NULL,
    start_s double precision NOT NULL,
    end_s double precision NOT NULL,
    keyframes jsonb NOT NULL DEFAULT '[]',
    transcript text NOT NULL DEFAULT '',
    faces jsonb NOT NULL DEFAULT '[]',
    mentions jsonb NOT NULL DEFAULT '[]',
    keyword_hits jsonb NOT NULL DEFAULT '[]',
    faces_status text NOT NULL DEFAULT 'pending',
    mentions_status text NOT NULL DEFAULT 'pending',
    PRIMARY KEY (job_id, scene_id)
);

CREATE TABLE IF NOT EXISTS job_stages (
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage text NOT NULL,
    PRIMARY KEY (job_id, stage)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id bigserial PRIMARY KEY,
    canonical_id text NOT NULL,
    watchlist_id text NOT NULL,
    name text NOT NULL,
    aliases jsonb NOT NULL DEFAULT '[]',
    embedding vector(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS watchlist_wid_idx ON watchlist (watchlist_id);
-- Note: `vector` has no btree opclass, so it cannot be part of a PRIMARY KEY.
-- Multiple embedding rows per (canonical_id, watchlist_id) are allowed.
```

- [ ] **Step 2: Write the failing integration test**

```python
# tests/integration/test_db.py
import json
import uuid
import pytest
from celebvision.db import Database
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def _fresh_db():
    db = Database(Settings.from_env().postgres_dsn)
    await db.connect()
    await db.apply_schema()
    return db


async def test_create_and_get_job():
    db = await _fresh_db()
    try:
        jid = str(uuid.uuid4())
        await db.create_job(jid, "file", "/x.mp4", "wl1", ["goal"])
        row = await db.get_job(jid)
        assert row["status"] == "queued"
        assert row["source_kind"] == "file"
        assert json.loads(row["keywords"]) == ["goal"] if isinstance(row["keywords"], str) else row["keywords"] == ["goal"]
    finally:
        await db.close()


async def test_scene_stage_tracking():
    db = await _fresh_db()
    try:
        jid = str(uuid.uuid4())
        await db.create_job(jid, "file", "/x.mp4", "wl1", [])
        await db.insert_scene(jid, 0, 0.0, 5.0, ["k0.jpg"])
        await db.insert_scene(jid, 1, 5.0, 9.0, ["k1.jpg"])
        await db.set_scene_faces(jid, 0, [{"canonical_id": "messi"}])
        assert await db.count_scenes_stage_done(jid, "faces") == 1
        await db.mark_stage_done(jid, "ingest")
        assert await db.is_stage_done(jid, "ingest") is True
        assert await db.is_stage_done(jid, "scenes") is False
    finally:
        await db.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_db.py -v` (stack up)
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.db'` (or collection error). Without stack: skipped.

- [ ] **Step 4: Write the database layer**

```python
# src/celebvision/db.py
import json
from pathlib import Path
import asyncpg
from pgvector.asyncpg import register_vector

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        assert self._pool is not None, "call connect() first"
        return self._pool

    async def connect(self) -> None:
        # Ensure the vector extension exists BEFORE the pool is created, so every
        # pooled connection's init hook can register the vector codec successfully.
        boot = await asyncpg.connect(self._dsn)
        try:
            await boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await boot.close()
        self._pool = await asyncpg.create_pool(self._dsn, init=self._init_conn,
                                               min_size=1, max_size=10)

    async def _init_conn(self, conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def apply_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)

    async def create_job(self, job_id, source_kind, source_locator, watchlist_id,
                         keywords) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO jobs (id, source_kind, source_locator, watchlist_id, keywords) "
                "VALUES ($1,$2,$3,$4,$5)",
                job_id, source_kind, source_locator, watchlist_id, json.dumps(keywords),
            )

    async def get_job(self, job_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id=$1", job_id)
        return dict(row) if row else None

    async def set_job_status(self, job_id, status) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status=$2 WHERE id=$1", job_id, status)

    async def set_job_error(self, job_id, error) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status='failed', error=$2 WHERE id=$1",
                               job_id, error)

    async def set_job_completed(self, job_id) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status='done', completed_at=now() WHERE id=$1", job_id)

    async def set_expected_scene_count(self, job_id, n) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET expected_scene_count=$2 WHERE id=$1",
                               job_id, n)

    async def upsert_job_assets(self, job_id, *, video_key=None, audio_key=None,
                               transcript_key=None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO job_assets (job_id, video_key, audio_key, transcript_key) "
                "VALUES ($1,$2,$3,$4) ON CONFLICT (job_id) DO UPDATE SET "
                "video_key=COALESCE(EXCLUDED.video_key, job_assets.video_key), "
                "audio_key=COALESCE(EXCLUDED.audio_key, job_assets.audio_key), "
                "transcript_key=COALESCE(EXCLUDED.transcript_key, job_assets.transcript_key)",
                job_id, video_key, audio_key, transcript_key,
            )

    async def get_job_assets(self, job_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM job_assets WHERE job_id=$1", job_id)
        return dict(row) if row else None

    async def insert_scene(self, job_id, scene_id, start_s, end_s, keyframes) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO scenes (job_id, scene_id, start_s, end_s, keyframes) "
                "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (job_id, scene_id) DO UPDATE SET "
                "start_s=EXCLUDED.start_s, end_s=EXCLUDED.end_s, keyframes=EXCLUDED.keyframes",
                job_id, scene_id, start_s, end_s, json.dumps(keyframes),
            )

    async def get_scene(self, job_id, scene_id) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM scenes WHERE job_id=$1 AND scene_id=$2", job_id, scene_id)
        return dict(row) if row else None

    async def get_scenes(self, job_id) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM scenes WHERE job_id=$1 ORDER BY scene_id", job_id)
        return [dict(r) for r in rows]

    async def set_scene_faces(self, job_id, scene_id, faces) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE scenes SET faces=$3, faces_status='done' "
                "WHERE job_id=$1 AND scene_id=$2", job_id, scene_id, json.dumps(faces))

    async def set_scene_mentions(self, job_id, scene_id, transcript, mentions,
                                 keyword_hits) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE scenes SET transcript=$3, mentions=$4, keyword_hits=$5, "
                "mentions_status='done' WHERE job_id=$1 AND scene_id=$2",
                job_id, scene_id, transcript, json.dumps(mentions),
                json.dumps(keyword_hits))

    async def count_scenes_stage_done(self, job_id, stage) -> int:
        col = {"faces": "faces_status", "mentions": "mentions_status"}[stage]
        async with self.pool.acquire() as conn:
            n = await conn.fetchval(
                f"SELECT count(*) FROM scenes WHERE job_id=$1 AND {col}='done'", job_id)
        return int(n)

    async def mark_stage_done(self, job_id, stage) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO job_stages (job_id, stage) VALUES ($1,$2) "
                "ON CONFLICT DO NOTHING", job_id, stage)

    async def is_stage_done(self, job_id, stage) -> bool:
        async with self.pool.acquire() as conn:
            v = await conn.fetchval(
                "SELECT 1 FROM job_stages WHERE job_id=$1 AND stage=$2", job_id, stage)
        return v is not None
```

- [ ] **Step 5: Run test to verify it passes (stack up)**

Run: `docker compose up -d postgres && CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_db.py -v`
Expected: PASS (2 passed). (Postgres reachable at the default DSN. `docker-compose.yml` lands in Task 12; if not present yet, run a throwaway `docker run` pgvector on 5432 — but prefer implementing Task 12's compose first if convenient. The task gate is: green with stack, skipped without.)

- [ ] **Step 6: Commit**

```bash
git add sql/schema.sql src/celebvision/db.py tests/integration/test_db.py
git commit -m "feat: add asyncpg database layer and schema"
```

---

### Task 4: Blob storage (MinIO via aioboto3)

**Files:**
- Create: `src/celebvision/storage.py`
- Create: `tests/integration/test_storage.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces (`celebvision.storage`): `class BlobStore`:
  - `__init__(self, endpoint: str, access_key: str, secret_key: str)`.
  - `async ensure_bucket(bucket: str) -> None` (idempotent create).
  - `async put_bytes(bucket: str, key: str, data: bytes) -> str` (returns `f"{bucket}/{key}"`).
  - `async get_bytes(bucket: str, key: str) -> bytes`.
  - `async put_file(bucket: str, key: str, path: str) -> str`; `async get_file(bucket: str, key: str, dest: str) -> str`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_storage.py
import pytest
from celebvision.storage import BlobStore
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_put_get_round_trip():
    s = Settings.from_env()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    await store.ensure_bucket("test-bucket")
    ref = await store.put_bytes("test-bucket", "hello.txt", b"hi there")
    assert ref == "test-bucket/hello.txt"
    assert await store.get_bytes("test-bucket", "hello.txt") == b"hi there"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.storage'`. Without stack: skipped.

- [ ] **Step 3: Write the storage module**

```python
# src/celebvision/storage.py
import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError


class BlobStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._session = aioboto3.Session()

    def _client(self):
        return self._session.client(
            "s3", endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=Config(signature_version="s3v4"),
        )

    async def ensure_bucket(self, bucket: str) -> None:
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=bucket)
            except ClientError:
                await s3.create_bucket(Bucket=bucket)

    async def put_bytes(self, bucket: str, key: str, data: bytes) -> str:
        async with self._client() as s3:
            await s3.put_object(Bucket=bucket, Key=key, Body=data)
        return f"{bucket}/{key}"

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        async with self._client() as s3:
            resp = await s3.get_object(Bucket=bucket, Key=key)
            async with resp["Body"] as stream:
                return await stream.read()

    async def put_file(self, bucket: str, key: str, path: str) -> str:
        with open(path, "rb") as f:
            return await self.put_bytes(bucket, key, f.read())

    async def get_file(self, bucket: str, key: str, dest: str) -> str:
        data = await self.get_bytes(bucket, key)
        with open(dest, "wb") as f:
            f.write(data)
        return dest
```

- [ ] **Step 4: Run test to verify it passes (stack up)**

Run: `docker compose up -d minio && CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_storage.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/storage.py tests/integration/test_storage.py
git commit -m "feat: add MinIO blob storage via aioboto3"
```

---

### Task 5: pgvector watchlist index

**Files:**
- Create: `src/celebvision/watchlist/pg_index.py`
- Create: `tests/integration/test_pg_watchlist.py`

**Interfaces:**
- Consumes: `Database` (Task 3), M1 `WatchlistIndex` protocol.
- Produces (`celebvision.watchlist.pg_index`): `class PgVectorWatchlistIndex` implementing M1's `WatchlistIndex`:
  - `__init__(self, db: Database, watchlist_id: str)`; attribute `watchlist_id`.
  - `async add(canonical_id, name, aliases: list[str], embeddings: list[list[float]]) -> None`.
  - `async search(embedding: list[float], threshold: float) -> tuple[str,str,float] | None` (cosine similarity `1 - (embedding <=> $1)`, best row, filtered by threshold, scoped to `watchlist_id`).
  - `async names() -> list[tuple[str,str,list[str]]]` (distinct identities in this `watchlist_id`).

  Note: the M1 protocol is sync, but the pg implementation is async; workers/API call it with `await`. The protocol is structural — the async variant is used only in async contexts. Do not change the M1 protocol.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_pg_watchlist.py
import math
import uuid
import pytest
from celebvision.db import Database
from celebvision.config import Settings
from celebvision.watchlist.pg_index import PgVectorWatchlistIndex

pytestmark = pytest.mark.requires_stack


def _unit(seed: int) -> list[float]:
    v = [0.0] * 512
    v[seed] = 1.0
    return v


async def test_enroll_and_search():
    db = Database(Settings.from_env().postgres_dsn)
    await db.connect()
    await db.apply_schema()
    try:
        wid = "wl-" + uuid.uuid4().hex[:8]
        idx = PgVectorWatchlistIndex(db, wid)
        await idx.add("messi", "Lionel Messi", ["messi"], [_unit(0)])
        await idx.add("ronaldo", "Cristiano Ronaldo", ["cr7"], [_unit(1)])

        hit = await idx.search(_unit(0), threshold=0.5)
        assert hit is not None and hit[0] == "messi" and hit[2] > 0.9

        assert await idx.search(_unit(2), threshold=0.5) is None
        names = await idx.names()
        assert ("messi", "Lionel Messi", ["messi"]) in names
    finally:
        await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_pg_watchlist.py -v`
Expected: FAIL — `ModuleNotFoundError`. Without stack: skipped.

- [ ] **Step 3: Write the pg watchlist**

```python
# src/celebvision/watchlist/pg_index.py
import json
from celebvision.db import Database


class PgVectorWatchlistIndex:
    def __init__(self, db: Database, watchlist_id: str) -> None:
        self.db = db
        self.watchlist_id = watchlist_id

    async def add(self, canonical_id, name, aliases, embeddings) -> None:
        async with self.db.pool.acquire() as conn:
            for emb in embeddings:
                await conn.execute(
                    "INSERT INTO watchlist (canonical_id, watchlist_id, name, aliases, embedding) "
                    "VALUES ($1,$2,$3,$4,$5)",
                    canonical_id, self.watchlist_id, name, json.dumps(list(aliases)),
                    emb,
                )

    async def search(self, embedding, threshold):
        async with self.db.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT canonical_id, name, 1 - (embedding <=> $1) AS score "
                "FROM watchlist WHERE watchlist_id=$2 "
                "ORDER BY embedding <=> $1 LIMIT 1",
                embedding, self.watchlist_id,
            )
        if row is None or float(row["score"]) < threshold:
            return None
        return (row["canonical_id"], row["name"], float(row["score"]))

    async def names(self):
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT ON (canonical_id) canonical_id, name, aliases "
                "FROM watchlist WHERE watchlist_id=$1 ORDER BY canonical_id",
                self.watchlist_id,
            )
        out = []
        for r in rows:
            aliases = r["aliases"]
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            out.append((r["canonical_id"], r["name"], aliases))
        return out
```

- [ ] **Step 4: Run test to verify it passes (stack up)**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_pg_watchlist.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/watchlist/pg_index.py tests/integration/test_pg_watchlist.py
git commit -m "feat: add pgvector watchlist index"
```

---

### Task 6: Stub backends + client factories

**Files:**
- Create: `src/celebvision/inference/stub_client.py`
- Create: `src/celebvision/llm/stub_client.py`
- Create: `src/celebvision/factories.py`
- Create: `tests/test_stub_backends.py`

**Interfaces:**
- Consumes: M1 `InferenceClient`/`LLMClient`, `Transcript`/`TranscriptSegment`/`Word`/`FaceDetection`/`MentionExtraction` (Task M1-3), `SpokenMention`/`KeywordHit` (M1-2), `Settings` (Task 1), `PgVectorWatchlistIndex` (Task 5), `LocalWatchlistIndex` (M1-4), `Database`.
- Produces:
  - `StubInferenceClient` (`inference/stub_client.py`): `transcribe(audio_path)` → deterministic `Transcript` (`"welcome to the match messi scored a goal"`, words spanning 0–10 s); `analyze_faces(image_path)` → `[FaceDetection(bbox=(0,0,10,10), embedding=STUB_EMBEDDING)]` where `STUB_EMBEDDING` is a 512-dim unit vector with index 0 set to 1.0. Also exported: `STUB_EMBEDDING: list[float]`.
  - `StubLLMClient` (`llm/stub_client.py`): `extract_mentions(transcript_text, watchlist, keywords)` → for each `(cid,name,aliases)` in watchlist, emit a `SpokenMention` if `name`'s first token (lowercased) OR any alias appears in `transcript_text.lower()`; `keyword_hits` = one `KeywordHit(count=text.lower().count(kw))` per keyword present.
  - `factories.py`: `build_inference_client(settings) -> InferenceClient`; `build_llm_client(settings) -> LLMClient`; `async build_watchlist(settings, db, watchlist_id) -> WatchlistIndex` (pg → `PgVectorWatchlistIndex`; local → `LocalWatchlistIndex.load(settings.watchlist_path)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stub_backends.py
from celebvision.inference.stub_client import StubInferenceClient, STUB_EMBEDDING
from celebvision.llm.stub_client import StubLLMClient


def test_stub_transcribe_is_deterministic():
    t = StubInferenceClient().transcribe("ignored.wav")
    words = " ".join(w.text for seg in t.segments for w in seg.words)
    assert "messi" in words.lower()
    assert t.duration_s > 0

def test_stub_faces_returns_stub_embedding():
    dets = StubInferenceClient().analyze_faces("ignored.jpg")
    assert len(dets) == 1
    assert dets[0].embedding == STUB_EMBEDDING
    assert len(STUB_EMBEDDING) == 512

def test_stub_llm_matches_watchlist_name_in_text():
    llm = StubLLMClient()
    out = llm.extract_mentions("messi scored a goal",
                               [("messi", "Lionel Messi", ["messi"])], ["goal"])
    assert out.mentions[0].canonical_id == "messi"
    assert out.keyword_hits[0].keyword == "goal"
    assert out.keyword_hits[0].count == 1

def test_stub_llm_no_match():
    out = StubLLMClient().extract_mentions("nothing here",
                                           [("messi", "Lionel Messi", ["messi"])], [])
    assert out.mentions == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stub_backends.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the stub inference client**

```python
# src/celebvision/inference/stub_client.py
from celebvision.interfaces import Transcript, TranscriptSegment, Word, FaceDetection

STUB_EMBEDDING: list[float] = [1.0] + [0.0] * 511

_WORDS = ["welcome", "to", "the", "match", "messi", "scored", "a", "goal"]


class StubInferenceClient:
    def transcribe(self, audio_path: str) -> Transcript:
        words = []
        for i, w in enumerate(_WORDS):
            words.append(Word(text=w, start_s=float(i), end_s=float(i) + 0.9))
        seg = TranscriptSegment(start_s=0.0, end_s=float(len(_WORDS)),
                                text=" ".join(_WORDS), words=words)
        return Transcript(segments=[seg])

    def analyze_faces(self, image_path: str) -> list[FaceDetection]:
        return [FaceDetection(bbox=(0.0, 0.0, 10.0, 10.0), embedding=list(STUB_EMBEDDING))]
```

- [ ] **Step 4: Write the stub LLM client**

```python
# src/celebvision/llm/stub_client.py
from celebvision.interfaces import MentionExtraction
from celebvision.models import SpokenMention, KeywordHit


class StubLLMClient:
    def extract_mentions(self, transcript_text, watchlist, keywords) -> MentionExtraction:
        low = transcript_text.lower()
        mentions = []
        for canonical_id, name, aliases in watchlist:
            needles = [name.split()[0].lower()] + [a.lower() for a in aliases]
            if any(n in low for n in needles):
                mentions.append(SpokenMention(
                    name=name, canonical_id=canonical_id, confidence=0.99,
                    evidence_span=transcript_text))
        hits = []
        for kw in keywords:
            c = low.count(kw.lower())
            if c > 0:
                hits.append(KeywordHit(keyword=kw, count=c, spans=[transcript_text]))
        return MentionExtraction(mentions=mentions, keyword_hits=hits)
```

- [ ] **Step 5: Write the factories**

```python
# src/celebvision/factories.py
from celebvision.interfaces import InferenceClient, LLMClient, WatchlistIndex
from celebvision.config import Settings
from celebvision.db import Database


def build_inference_client(settings: Settings) -> InferenceClient:
    if settings.inference_backend == "stub":
        from celebvision.inference.stub_client import StubInferenceClient
        return StubInferenceClient()
    if settings.inference_backend == "local":
        from celebvision.inference.local_client import LocalInferenceClient
        return LocalInferenceClient(whisper_model=settings.whisper_model,
                                    face_model=settings.face_model)
    raise ValueError(f"unknown INFERENCE_BACKEND: {settings.inference_backend}")


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_backend == "stub":
        from celebvision.llm.stub_client import StubLLMClient
        return StubLLMClient()
    if settings.llm_backend == "anthropic":
        from celebvision.llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    raise ValueError(f"unknown LLM_BACKEND: {settings.llm_backend}")


async def build_watchlist(settings: Settings, db: Database,
                          watchlist_id: str) -> WatchlistIndex:
    if settings.watchlist_backend == "pg":
        from celebvision.watchlist.pg_index import PgVectorWatchlistIndex
        return PgVectorWatchlistIndex(db, watchlist_id)
    if settings.watchlist_backend == "local":
        from celebvision.watchlist.local_index import LocalWatchlistIndex
        return LocalWatchlistIndex.load(settings.watchlist_path)
    raise ValueError(f"unknown WATCHLIST_BACKEND: {settings.watchlist_backend}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_stub_backends.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add src/celebvision/inference/stub_client.py src/celebvision/llm/stub_client.py src/celebvision/factories.py tests/test_stub_backends.py
git commit -m "feat: add stub inference/LLM backends and client factories"
```

---

### Task 7: Coordinator decision logic + runner

**Files:**
- Create: `src/celebvision/coordinator.py`
- Create: `tests/test_coordinator_decide.py`

**Interfaces:**
- Consumes: `Message` (Task 2), `Database` (Task 3), `MessageBus` (Task 2).
- Produces (`celebvision.coordinator`):
  - `@dataclass JobState`: `status: str`, `expected_scene_count: int | None`, `ingest_done: bool`, `scenes_done: bool`, `transcribe_done: bool`, `faces_done: int`, `mentions_done: int`.
  - `decide_next(event: Message, state: JobState) -> list[Message]` — pure. Rules:
    - `state.status == 'failed'` → `[]`.
    - `event.stage == 'ingest'` → `[Message(job_id, 'scenes'), Message(job_id, 'transcribe')]`.
    - `event.stage in ('scenes','transcribe')`: if `scenes_done and transcribe_done and expected_scene_count is not None` → for `sid in range(expected_scene_count)`: `Message(job_id,'faces',scene_id=sid)`, `Message(job_id,'mentions',scene_id=sid)`; else `[]`.
    - `event.stage in ('faces','mentions')`: if `expected_scene_count is not None and faces_done == expected and mentions_done == expected` → `[Message(job_id,'aggregate')]`; else `[]`.
    - `event.stage == 'aggregate'` → `[]`.
  - `async load_job_state(db, job_id) -> JobState`.
  - `async run_coordinator(bus: MessageBus, db: Database) -> None` — consume `stage.events`; on `aggregate` completion call `db.set_job_completed`; for each `out` in `decide_next(...)` produce to `f"{out.stage}.requested"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coordinator_decide.py
from celebvision.bus import Message
from celebvision.coordinator import decide_next, JobState


def _state(**kw):
    base = dict(status="running", expected_scene_count=None, ingest_done=False,
                scenes_done=False, transcribe_done=False, faces_done=0, mentions_done=0)
    base.update(kw)
    return JobState(**base)


def test_ingest_fans_out_scenes_and_transcribe():
    out = decide_next(Message(job_id="j", stage="ingest"), _state(ingest_done=True))
    assert sorted(m.stage for m in out) == ["scenes", "transcribe"]

def test_scenes_alone_does_not_fan_out():
    out = decide_next(Message(job_id="j", stage="scenes"),
                      _state(scenes_done=True, transcribe_done=False,
                             expected_scene_count=2))
    assert out == []

def test_scenes_and_transcribe_fan_out_per_scene():
    out = decide_next(Message(job_id="j", stage="transcribe"),
                      _state(scenes_done=True, transcribe_done=True,
                             expected_scene_count=2))
    stages = sorted((m.stage, m.scene_id) for m in out)
    assert stages == [("faces", 0), ("faces", 1), ("mentions", 0), ("mentions", 1)]

def test_all_scene_stages_done_triggers_aggregate():
    out = decide_next(Message(job_id="j", stage="faces"),
                      _state(expected_scene_count=2, faces_done=2, mentions_done=2))
    assert [m.stage for m in out] == ["aggregate"]

def test_partial_scene_progress_waits():
    out = decide_next(Message(job_id="j", stage="faces"),
                      _state(expected_scene_count=2, faces_done=2, mentions_done=1))
    assert out == []

def test_failed_job_emits_nothing():
    out = decide_next(Message(job_id="j", stage="ingest"),
                      _state(status="failed", ingest_done=True))
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_coordinator_decide.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the coordinator**

```python
# src/celebvision/coordinator.py
from dataclasses import dataclass
from celebvision.bus import Message, MessageBus
from celebvision.db import Database


@dataclass
class JobState:
    status: str
    expected_scene_count: int | None
    ingest_done: bool
    scenes_done: bool
    transcribe_done: bool
    faces_done: int
    mentions_done: int


def decide_next(event: Message, state: JobState) -> list[Message]:
    if state.status == "failed":
        return []
    jid = event.job_id
    stage = event.stage
    if stage == "ingest":
        return [Message(job_id=jid, stage="scenes"),
                Message(job_id=jid, stage="transcribe")]
    if stage in ("scenes", "transcribe"):
        if state.scenes_done and state.transcribe_done \
                and state.expected_scene_count is not None:
            out: list[Message] = []
            for sid in range(state.expected_scene_count):
                out.append(Message(job_id=jid, stage="faces", scene_id=sid))
                out.append(Message(job_id=jid, stage="mentions", scene_id=sid))
            return out
        return []
    if stage in ("faces", "mentions"):
        n = state.expected_scene_count
        if n is not None and state.faces_done == n and state.mentions_done == n:
            return [Message(job_id=jid, stage="aggregate")]
        return []
    return []


async def load_job_state(db: Database, job_id: str) -> JobState:
    job = await db.get_job(job_id)
    return JobState(
        status=job["status"] if job else "unknown",
        expected_scene_count=job["expected_scene_count"] if job else None,
        ingest_done=await db.is_stage_done(job_id, "ingest"),
        scenes_done=await db.is_stage_done(job_id, "scenes"),
        transcribe_done=await db.is_stage_done(job_id, "transcribe"),
        faces_done=await db.count_scenes_stage_done(job_id, "faces"),
        mentions_done=await db.count_scenes_stage_done(job_id, "mentions"),
    )


async def run_coordinator(bus: MessageBus, db: Database) -> None:
    async for event in bus.stream(["stage.events"], group="coordinator"):
        if event.stage == "aggregate":
            await db.set_job_completed(event.job_id)
        state = await load_job_state(db, event.job_id)
        for out in decide_next(event, state):
            await bus.produce(f"{out.stage}.requested", out.job_id, out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_coordinator_decide.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/coordinator.py tests/test_coordinator_decide.py
git commit -m "feat: add coordinator DAG decision logic and runner"
```

---

### Task 8: Worker base loop + context + entrypoint dispatch

**Files:**
- Create: `src/celebvision/workers/__init__.py`
- Create: `src/celebvision/workers/base.py`
- Create: `tests/test_worker_base.py`

**Interfaces:**
- Consumes: `Message`/`MessageBus` (Task 2), `Database` (Task 3), `BlobStore` (Task 4), M1 `InferenceClient`/`LLMClient`, `Settings`, `StageError` (Task 1).
- Produces (`celebvision.workers.base`):
  - `@dataclass WorkerContext`: `db: Database`, `storage: BlobStore`, `inference`, `llm`, `settings: Settings`.
  - `Handler = Callable[[Message, WorkerContext], Awaitable[None]]`.
  - `async run_worker(stage: str, bus: MessageBus, ctx: WorkerContext, handler: Handler) -> None`: for each `msg` in `bus.stream([f"{stage}.requested"], group=stage)`: `try: await handler(msg, ctx)` then `await bus.produce("stage.events", msg.job_id, Message(job_id=msg.job_id, stage=stage, scene_id=msg.scene_id))`; `except StageError as e: await ctx.db.set_job_error(msg.job_id, str(e))`. (Returning normally lets `stream` commit the offset in both cases — no poison-loop.)

- [ ] **Step 1: Write the failing test (fake bus/db — pure loop logic)**

```python
# tests/test_worker_base.py
import pytest
from celebvision.bus import Message
from celebvision.errors import StageError
from celebvision.workers.base import run_worker, WorkerContext


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


def _ctx(db):
    return WorkerContext(db=db, storage=None, inference=None, llm=None, settings=None)


async def test_success_emits_completion_event():
    bus = FakeBus([Message(job_id="j", stage="faces", scene_id=1)])
    calls = []
    async def handler(msg, ctx):
        calls.append(msg.job_id)
    await run_worker("faces", bus, _ctx(FakeDB()), handler)
    assert calls == ["j"]
    assert bus.produced == [("stage.events",
                             Message(job_id="j", stage="faces", scene_id=1))]

async def test_stage_error_marks_job_failed_and_does_not_emit():
    bus = FakeBus([Message(job_id="j", stage="ingest")])
    db = FakeDB()
    async def handler(msg, ctx):
        raise StageError("ingest", "boom")
    await run_worker("ingest", bus, _ctx(db), handler)
    assert db.errors == [("j", "ingest: boom")]
    assert bus.produced == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_worker_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.workers'`

- [ ] **Step 3: Write the worker base**

```python
# src/celebvision/workers/__init__.py
```

```python
# src/celebvision/workers/base.py
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from celebvision.bus import Message, MessageBus
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.config import Settings
from celebvision.errors import StageError


@dataclass
class WorkerContext:
    db: Database
    storage: BlobStore
    inference: object
    llm: object
    settings: Settings


Handler = Callable[[Message, WorkerContext], Awaitable[None]]


async def run_worker(stage: str, bus: MessageBus, ctx: WorkerContext,
                     handler: Handler) -> None:
    async for msg in bus.stream([f"{stage}.requested"], group=stage):
        try:
            await handler(msg, ctx)
        except StageError as e:
            await ctx.db.set_job_error(msg.job_id, str(e))
            continue
        await bus.produce("stage.events", msg.job_id,
                          Message(job_id=msg.job_id, stage=stage, scene_id=msg.scene_id))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_worker_base.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/workers/ tests/test_worker_base.py
git commit -m "feat: add worker base loop and context"
```

---

### Task 9: Ingest + scene worker handlers (+ probe_duration)

**Files:**
- Modify: `src/celebvision/media.py` (add `probe_duration`)
- Create: `src/celebvision/workers/ingest.py`
- Create: `src/celebvision/workers/scene.py`
- Create: `tests/integration/test_ingest_scene_workers.py`
- Create test fixture video: generated in the test via ffmpeg.

**Interfaces:**
- Consumes: `WorkerContext`/`run_worker` (Task 8), `Message`, `Database`, `BlobStore`, M1 `media.ingest`/`media.extract_keyframe`/`media.classify_source`, M1 `stages.scenes.detect_scenes`/`build_scene_windows`, `StageError`.
- Produces:
  - `media.probe_duration(video_path: str, runner=subprocess.run) -> float` — ffprobe duration seconds; returns 0.0 on failure.
  - `workers/ingest.py`: `async handle_ingest(msg, ctx) -> None` — load job, `classify_source`, `ingest` to a temp dir (wrap sync in `asyncio.to_thread`), upload `video`+`audio` to MinIO bucket `media` under `{job_id}/`, `upsert_job_assets(video_key, audio_key)`, `set_job_status(running)`, `mark_stage_done('ingest')`. Wrap failures in `StageError("ingest", ...)`.
  - `workers/scene.py`: `async handle_scene(msg, ctx) -> None` — download video from MinIO to temp, `detect_scenes` (to_thread); if single-shot sentinel `[(0.0, 0.0)]`, replace with `[(0.0, probe_duration(video))]` (M1-handoff #2 fix); `build_scene_windows` (keyframes to temp), upload each keyframe to bucket `keyframes` under `{job_id}/`, `insert_scene` per window (with MinIO keyframe keys), `set_expected_scene_count`, `mark_stage_done('scenes')`. Wrap failures in `StageError("scene", ...)`.

- [ ] **Step 1: Write `probe_duration` + its unit test**

```python
# tests/test_media_probe.py
from celebvision.media import probe_duration


def test_probe_duration_parses_ffprobe_output():
    def fake_runner(cmd, capture_output, text, check):
        class R:
            stdout = "12.34\n"
            returncode = 0
        assert "ffprobe" in cmd[0]
        return R()
    assert probe_duration("v.mp4", runner=fake_runner) == 12.34

def test_probe_duration_returns_zero_on_error():
    def fake_runner(cmd, capture_output, text, check):
        raise RuntimeError("no ffprobe")
    assert probe_duration("v.mp4", runner=fake_runner) == 0.0
```

- [ ] **Step 2: Run and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_media_probe.py -v`
Expected: FAIL — `ImportError: cannot import name 'probe_duration'`

- [ ] **Step 3: Add `probe_duration` to `media.py`**

```python
# append to src/celebvision/media.py

def probe_duration(video_path: str, runner=subprocess.run) -> float:
    try:
        result = runner(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
```

- [ ] **Step 4: Run and confirm it passes**

Run: `.venv/bin/python -m pytest tests/test_media_probe.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the ingest + scene handlers**

```python
# src/celebvision/workers/ingest.py
import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.media import classify_source, ingest
from celebvision.models import JobSource
from celebvision.errors import StageError


async def handle_ingest(msg: Message, ctx: WorkerContext) -> None:
    job = await ctx.db.get_job(msg.job_id)
    if job is None:
        raise StageError("ingest", f"job {msg.job_id} not found")
    await ctx.db.set_job_status(msg.job_id, "running")
    try:
        source = JobSource(kind=job["source_kind"], locator=job["source_locator"])
        workdir = tempfile.mkdtemp(prefix=f"ingest-{msg.job_id}-")
        video_path, audio_path = await asyncio.to_thread(ingest, source, workdir)
        await ctx.storage.ensure_bucket("media")
        video_key = f"{msg.job_id}/{os.path.basename(video_path)}"
        audio_key = f"{msg.job_id}/audio.wav"
        await ctx.storage.put_file("media", video_key, video_path)
        await ctx.storage.put_file("media", audio_key, audio_path)
        await ctx.db.upsert_job_assets(msg.job_id, video_key=video_key,
                                       audio_key=audio_key)
        await ctx.db.mark_stage_done(msg.job_id, "ingest")
    except StageError:
        raise
    except Exception as e:
        raise StageError("ingest", str(e)) from e
```

```python
# src/celebvision/workers/scene.py
import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.media import probe_duration
from celebvision.stages.scenes import detect_scenes, build_scene_windows
from celebvision.errors import StageError


async def handle_scene(msg: Message, ctx: WorkerContext) -> None:
    assets = await ctx.db.get_job_assets(msg.job_id)
    if assets is None or not assets["video_key"]:
        raise StageError("scene", "no video asset")
    try:
        tmp = tempfile.mkdtemp(prefix=f"scene-{msg.job_id}-")
        video_path = os.path.join(tmp, "video.mp4")
        await ctx.storage.get_file("media", assets["video_key"], video_path)

        boundaries = await asyncio.to_thread(detect_scenes, video_path)
        if boundaries == [(0.0, 0.0)]:  # single-shot sentinel -> real duration
            boundaries = [(0.0, await asyncio.to_thread(probe_duration, video_path))]

        windows = await asyncio.to_thread(build_scene_windows, video_path,
                                          boundaries, tmp)
        await ctx.storage.ensure_bucket("keyframes")
        for w in windows:
            keys = []
            for local_kf in w.keyframes:
                key = f"{msg.job_id}/{os.path.basename(local_kf)}"
                await ctx.storage.put_file("keyframes", key, local_kf)
                keys.append(key)
            await ctx.db.insert_scene(msg.job_id, w.scene_id, w.start_s, w.end_s, keys)
        await ctx.db.set_expected_scene_count(msg.job_id, len(windows))
        await ctx.db.mark_stage_done(msg.job_id, "scenes")
    except StageError:
        raise
    except Exception as e:
        raise StageError("scene", str(e)) from e
```

- [ ] **Step 6: Write the integration test (requires_stack + host ffmpeg)**

```python
# tests/integration/test_ingest_scene_workers.py
import subprocess
import tempfile
import os
import uuid
import pytest
from celebvision.bus import Message
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.workers.base import WorkerContext
from celebvision.workers.ingest import handle_ingest
from celebvision.workers.scene import handle_scene

pytestmark = pytest.mark.requires_stack


async def _ctx():
    s = Settings.from_env()
    db = Database(s.postgres_dsn)
    await db.connect()
    await db.apply_schema()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    return s, db, WorkerContext(db=db, storage=store, inference=None, llm=None, settings=s)


async def test_ingest_then_scene_populates_db_and_storage(tmp_path):
    # Make a 3s test video (color bars) with ffmpeg on the host.
    video = str(tmp_path / "sample.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "3", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "3", "-shortest", video], check=True)

    s, db, ctx = await _ctx()
    try:
        jid = str(uuid.uuid4())
        await db.create_job(jid, "file", video, "wl1", [])
        await handle_ingest(Message(job_id=jid, stage="ingest"), ctx)
        assets = await db.get_job_assets(jid)
        assert assets["video_key"] and assets["audio_key"]
        assert (await db.get_job(jid))["status"] == "running"
        assert await db.is_stage_done(jid, "ingest")

        await handle_scene(Message(job_id=jid, stage="scenes"), ctx)
        scenes = await db.get_scenes(jid)
        assert len(scenes) >= 1
        assert (await db.get_job(jid))["expected_scene_count"] == len(scenes)
        assert scenes[0]["end_s"] > 0.0  # single-shot end_s fix
        assert await db.is_stage_done(jid, "scenes")
    finally:
        await db.close()
```

- [ ] **Step 7: Run (stack up) and commit**

Run: `docker compose up -d && CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/test_media_probe.py tests/integration/test_ingest_scene_workers.py -v`
Expected: probe tests PASS; integration PASS (needs host ffmpeg + stack). Without stack: probe PASS, integration skipped.

```bash
git add src/celebvision/media.py src/celebvision/workers/ingest.py src/celebvision/workers/scene.py tests/test_media_probe.py tests/integration/test_ingest_scene_workers.py
git commit -m "feat: add ingest and scene workers with duration probe"
```

---

### Task 10: Transcribe + face + mention + aggregate worker handlers

**Files:**
- Create: `src/celebvision/workers/transcribe.py`
- Create: `src/celebvision/workers/face.py`
- Create: `src/celebvision/workers/mention.py`
- Create: `src/celebvision/workers/aggregate.py`
- Create: `tests/integration/test_downstream_workers.py`

**Interfaces:**
- Consumes: Task 8/9 outputs, M1 `stages.faces.match_faces_in_scene`, `stages.mentions.scene_transcript_text`/`extract_scene_mentions`, `stages.aggregate.build_scene_report`/`build_report`, `stages.scenes.SceneWindow`, `interfaces.Transcript`, `factories.build_watchlist`, `models.JobSource`.
- Produces (each `async handle_<stage>(msg, ctx) -> None`, wrapping failures in `StageError`):
  - `transcribe`: download audio from MinIO, `ctx.inference.transcribe` (to_thread), serialize `Transcript.model_dump_json()` to MinIO `media/{job}/transcript.json`, `upsert_job_assets(transcript_key)`, `mark_stage_done('transcribe')`.
  - `face`: read the scene row (`msg.scene_id`) + its keyframes; download keyframes from MinIO to temp; build a `SceneWindow`; `await build_watchlist(...)` (async pg search — call `match_faces_in_scene` adapted: since M1's `match_faces_in_scene` is sync and calls `watchlist.search` synchronously, the worker instead does the matching inline with `await` — see Step 3); `set_scene_faces(job, scene_id, faces_as_dicts)`.
  - `mention`: read scene row + `job_assets.transcript_key`; download transcript JSON, `Transcript.model_validate_json`; build `SceneWindow` from the scene row; `scene_transcript_text`; `extract_scene_mentions(text, watchlist_names, keywords, ctx.llm)` — but watchlist `names()` is async for pg, so fetch names first (`await watchlist.names()`) and call `ctx.llm.extract_mentions(text, names, keywords)` directly; `set_scene_mentions(job, scene_id, text, mentions_as_dicts)`.
  - `aggregate`: load all scenes, reconstruct `SceneReport`s (from stored `faces`/`mentions`/`transcript`/`keyframes`), `build_report(...)`, write `reports/{job}.json` to bucket `reports`.

  Note on async watchlist: M1's `match_faces_in_scene`/`extract_scene_mentions` expect a **sync** `WatchlistIndex`. The pg index is async. So the face/mention workers do NOT call those two M1 helpers with the pg index directly; they replicate the tiny matching loop with `await`. (The M1 `build_scene_report`/`build_report`/`scene_transcript_text` are pure and reused unchanged.)

- [ ] **Step 1: Write the failing integration test (requires_stack, stub inference/llm)**

```python
# tests/integration/test_downstream_workers.py
import json
import uuid
import pytest
from celebvision.bus import Message
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.workers.base import WorkerContext
from celebvision.workers.transcribe import handle_transcribe
from celebvision.workers.face import handle_face
from celebvision.workers.mention import handle_mention
from celebvision.workers.aggregate import handle_aggregate
from celebvision.inference.stub_client import StubInferenceClient, STUB_EMBEDDING
from celebvision.llm.stub_client import StubLLMClient
from celebvision.watchlist.pg_index import PgVectorWatchlistIndex

pytestmark = pytest.mark.requires_stack


async def _ctx(watchlist_id):
    s = Settings.from_env()  # INFERENCE_BACKEND/LLM_BACKEND default to stub
    db = Database(s.postgres_dsn)
    await db.connect()
    await db.apply_schema()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    await store.ensure_bucket("media")
    ctx = WorkerContext(db=db, storage=store, inference=StubInferenceClient(),
                        llm=StubLLMClient(), settings=s)
    return s, db, store, ctx


async def test_downstream_pipeline_with_stub(tmp_path):
    s, db, store, ctx = await _ctx("wl-dl")
    try:
        wid = "wl-" + uuid.uuid4().hex[:8]
        idx = PgVectorWatchlistIndex(db, wid)
        await idx.add("messi", "Lionel Messi", ["messi"], [list(STUB_EMBEDDING)])

        jid = str(uuid.uuid4())
        await db.create_job(jid, "file", "/x.mp4", wid, ["goal"])
        # seed audio asset + one scene + a keyframe
        await store.put_bytes("media", f"{jid}/audio.wav", b"RIFFstub")
        await db.upsert_job_assets(jid, audio_key=f"{jid}/audio.wav")
        await store.ensure_bucket("keyframes")
        await store.put_bytes("keyframes", f"{jid}/k0.jpg", b"\xff\xd8stub")
        await db.insert_scene(jid, 0, 0.0, 8.0, [f"{jid}/k0.jpg"])
        await db.set_expected_scene_count(jid, 1)

        await handle_transcribe(Message(job_id=jid, stage="transcribe"), ctx)
        assert (await db.get_job_assets(jid))["transcript_key"]

        await handle_face(Message(job_id=jid, stage="faces", scene_id=0), ctx)
        await handle_mention(Message(job_id=jid, stage="mentions", scene_id=0), ctx)
        scene = await db.get_scene(jid, 0)
        faces = scene["faces"] if isinstance(scene["faces"], list) else json.loads(scene["faces"])
        mentions = scene["mentions"] if isinstance(scene["mentions"], list) else json.loads(scene["mentions"])
        khits = scene["keyword_hits"] if isinstance(scene["keyword_hits"], list) else json.loads(scene["keyword_hits"])
        assert faces[0]["canonical_id"] == "messi"
        assert mentions[0]["canonical_id"] == "messi"
        assert khits[0]["keyword"] == "goal"

        await handle_aggregate(Message(job_id=jid, stage="aggregate"), ctx)
        report_bytes = await store.get_bytes("reports", f"{jid}.json")
        report = json.loads(report_bytes)
        assert report["job_id"] == jid
        ids = [c["canonical_id"] for c in report["celebrity_index"]]
        assert "messi" in ids
    finally:
        await db.close()
```

- [ ] **Step 2: Run and confirm it fails**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_downstream_workers.py -v`
Expected: FAIL — `ModuleNotFoundError` (handlers absent). Without stack: skipped.

- [ ] **Step 3: Write the four handlers**

```python
# src/celebvision/workers/transcribe.py
import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.errors import StageError


async def handle_transcribe(msg: Message, ctx: WorkerContext) -> None:
    assets = await ctx.db.get_job_assets(msg.job_id)
    if assets is None or not assets["audio_key"]:
        raise StageError("transcribe", "no audio asset")
    try:
        tmp = tempfile.mkdtemp(prefix=f"transcribe-{msg.job_id}-")
        audio_path = os.path.join(tmp, "audio.wav")
        await ctx.storage.get_file("media", assets["audio_key"], audio_path)
        transcript = await asyncio.to_thread(ctx.inference.transcribe, audio_path)
        key = f"{msg.job_id}/transcript.json"
        await ctx.storage.put_bytes("media", key,
                                    transcript.model_dump_json().encode("utf-8"))
        await ctx.db.upsert_job_assets(msg.job_id, transcript_key=key)
        await ctx.db.mark_stage_done(msg.job_id, "transcribe")
    except StageError:
        raise
    except Exception as e:
        raise StageError("transcribe", str(e)) from e
```

```python
# src/celebvision/workers/face.py
import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.factories import build_watchlist
from celebvision.errors import StageError

_FACE_THRESHOLD = 0.35


async def handle_face(msg: Message, ctx: WorkerContext) -> None:
    scene = await ctx.db.get_scene(msg.job_id, msg.scene_id)
    if scene is None:
        raise StageError("face", f"scene {msg.scene_id} not found")
    job = await ctx.db.get_job(msg.job_id)
    try:
        keyframes = scene["keyframes"]
        if isinstance(keyframes, str):
            import json
            keyframes = json.loads(keyframes)
        watchlist = await build_watchlist(ctx.settings, ctx.db, job["watchlist_id"])
        tmp = tempfile.mkdtemp(prefix=f"face-{msg.job_id}-{msg.scene_id}-")
        best: dict[str, dict] = {}
        for i, key in enumerate(keyframes):
            local = os.path.join(tmp, f"kf{i}.jpg")
            await ctx.storage.get_file("keyframes", key, local)
            for det in await asyncio.to_thread(ctx.inference.analyze_faces, local):
                hit = await watchlist.search(det.embedding, _FACE_THRESHOLD)
                if hit is None:
                    continue
                cid, name, score = hit
                if cid not in best or score > best[cid]["confidence"]:
                    best[cid] = {"name": name, "canonical_id": cid,
                                 "confidence": score, "bbox": list(det.bbox),
                                 "keyframe": key}
        await ctx.db.set_scene_faces(msg.job_id, msg.scene_id, list(best.values()))
    except StageError:
        raise
    except Exception as e:
        raise StageError("face", str(e)) from e
```

```python
# src/celebvision/workers/mention.py
import json
import tempfile
import os
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.factories import build_watchlist
from celebvision.interfaces import Transcript
from celebvision.stages.scenes import SceneWindow
from celebvision.stages.mentions import scene_transcript_text
from celebvision.errors import StageError


async def handle_mention(msg: Message, ctx: WorkerContext) -> None:
    scene = await ctx.db.get_scene(msg.job_id, msg.scene_id)
    if scene is None:
        raise StageError("mention", f"scene {msg.scene_id} not found")
    job = await ctx.db.get_job(msg.job_id)
    assets = await ctx.db.get_job_assets(msg.job_id)
    if assets is None or not assets["transcript_key"]:
        raise StageError("mention", "no transcript asset")
    try:
        tmp = tempfile.mkdtemp(prefix=f"mention-{msg.job_id}-{msg.scene_id}-")
        tpath = os.path.join(tmp, "transcript.json")
        await ctx.storage.get_file("media", assets["transcript_key"], tpath)
        with open(tpath) as f:
            transcript = Transcript.model_validate_json(f.read())

        window = SceneWindow(scene_id=scene["scene_id"], start_s=scene["start_s"],
                             end_s=scene["end_s"], keyframes=[])
        text = scene_transcript_text(transcript, window)

        keywords = job["keywords"]
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        watchlist = await build_watchlist(ctx.settings, ctx.db, job["watchlist_id"])
        names = await watchlist.names()
        result = ctx.llm.extract_mentions(text, names, keywords)
        mentions = [m.model_dump() for m in result.mentions]
        keyword_hits = [k.model_dump() for k in result.keyword_hits]
        await ctx.db.set_scene_mentions(msg.job_id, msg.scene_id, text, mentions,
                                        keyword_hits)
    except StageError:
        raise
    except Exception as e:
        raise StageError("mention", str(e)) from e
```

```python
# src/celebvision/workers/aggregate.py
import json
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.stages.scenes import SceneWindow
from celebvision.stages.aggregate import build_scene_report, build_report
from celebvision.models import SpokenMention, KeywordHit, OnscreenFace, JobSource
from celebvision.errors import StageError


def _as_list(v):
    return v if isinstance(v, list) else json.loads(v)


async def handle_aggregate(msg: Message, ctx: WorkerContext) -> None:
    job = await ctx.db.get_job(msg.job_id)
    if job is None:
        raise StageError("aggregate", "job not found")
    try:
        rows = await ctx.db.get_scenes(msg.job_id)
        scene_reports = []
        for row in rows:
            window = SceneWindow(scene_id=row["scene_id"], start_s=row["start_s"],
                                 end_s=row["end_s"], keyframes=_as_list(row["keyframes"]))
            mentions = [SpokenMention(**m) for m in _as_list(row["mentions"])]
            keyword_hits = [KeywordHit(**k) for k in _as_list(row["keyword_hits"])]
            faces = [OnscreenFace(**f) for f in _as_list(row["faces"])]
            scene_reports.append(build_scene_report(
                window, row["transcript"], mentions, keyword_hits, faces))
        report = build_report(
            msg.job_id, JobSource(kind=job["source_kind"], locator=job["source_locator"]),
            job["watchlist_id"], "", 0.0, scene_reports)
        await ctx.storage.ensure_bucket("reports")
        await ctx.storage.put_bytes("reports", f"{msg.job_id}.json",
                                    report.model_dump_json(indent=2).encode("utf-8"))
    except StageError:
        raise
    except Exception as e:
        raise StageError("aggregate", str(e)) from e
```

Note: `build_report`'s `completed_at`/`duration_s` are set to `""`/`0.0` here because the persisted per-scene rows are the source of truth for M2; the job's authoritative `completed_at` lives in Postgres (threading real values is deferred to M3/M4). Keyword hits ARE carried end-to-end: the mention worker persists `scenes.keyword_hits`, and the aggregate reads them into each `SceneReport`.

- [ ] **Step 4: Run integration test (stack up) to verify it passes**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_downstream_workers.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/workers/transcribe.py src/celebvision/workers/face.py src/celebvision/workers/mention.py src/celebvision/workers/aggregate.py tests/integration/test_downstream_workers.py
git commit -m "feat: add transcribe/face/mention/aggregate workers"
```

---

### Task 11: FastAPI application

**Files:**
- Create: `src/celebvision/api/__init__.py`
- Create: `src/celebvision/api/app.py`
- Create: `tests/integration/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `Database`, `MessageBus`, `Message`, `BlobStore`.
- Produces (`celebvision.api.app`):
  - `create_app(settings: Settings | None = None) -> FastAPI` — lifespan connects `Database` + `MessageBus` (+ applies schema) and stores them on `app.state`; closes on shutdown.
  - Routes: `GET /healthz` → `{"status":"ok"}`; `POST /jobs` body `JobRequest{source:str, watchlist_id:str, keywords:list[str]=[]}` → create job (`uuid4`), classify source, produce `ingest.requested`, return `{"job_id": ...}` (201); `GET /jobs/{id}` → `{status, error, expected_scene_count, scenes_done:{faces,mentions,total}}` or 404; `GET /reports/{id}` → report JSON from MinIO if job `done`, else 404.
  - `app = create_app()` module-level for uvicorn.

- [ ] **Step 1: Write the failing integration test (requires_stack)**

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from celebvision.api.app import create_app
from celebvision.bus import MessageBus
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_healthz():
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/healthz")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"


async def test_post_job_creates_row_and_emits_ingest():
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/jobs", json={"source": "/tmp/x.mp4",
                                            "watchlist_id": "wl1", "keywords": ["goal"]})
            assert r.status_code == 201
            jid = r.json()["job_id"]

            db = app.state.db
            job = await db.get_job(jid)
            assert job["status"] == "queued"
            assert job["source_kind"] == "file"

            status = await c.get(f"/jobs/{jid}")
            assert status.status_code == 200
            assert status.json()["status"] == "queued"

    # verify ingest.requested was produced
    bus = MessageBus(Settings.from_env().kafka_bootstrap)
    await bus.start()
    try:
        seen = False
        async for m in bus.stream(["ingest.requested"], group="test-api-verify"):
            if m.job_id == jid:
                seen = True
            break
        assert seen
    finally:
        await bus.stop()
```

- [ ] **Step 2: Run and confirm it fails**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError`. Without stack: skipped.

- [ ] **Step 3: Write the API**

```python
# src/celebvision/api/__init__.py
```

```python
# src/celebvision/api/app.py
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.bus import MessageBus, Message
from celebvision.storage import BlobStore
from celebvision.media import classify_source


class JobRequest(BaseModel):
    source: str
    watchlist_id: str
    keywords: list[str] = []


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.postgres_dsn)
        await db.connect()
        await db.apply_schema()
        bus = MessageBus(settings.kafka_bootstrap)
        await bus.start()
        store = BlobStore(settings.minio_endpoint, settings.minio_access_key,
                          settings.minio_secret_key)
        app.state.db = db
        app.state.bus = bus
        app.state.store = store
        app.state.settings = settings
        try:
            yield
        finally:
            await bus.stop()
            await db.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/jobs", status_code=201)
    async def create_job(req: JobRequest):
        job_id = str(uuid.uuid4())
        src = classify_source(req.source)
        await app.state.db.create_job(job_id, src.kind, src.locator,
                                      req.watchlist_id, req.keywords)
        await app.state.bus.produce("ingest.requested", job_id,
                                    Message(job_id=job_id, stage="ingest"))
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await app.state.db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        total = job["expected_scene_count"]
        return {
            "status": job["status"],
            "error": job["error"],
            "expected_scene_count": total,
            "scenes_done": {
                "faces": await app.state.db.count_scenes_stage_done(job_id, "faces"),
                "mentions": await app.state.db.count_scenes_stage_done(job_id, "mentions"),
                "total": total,
            },
        }

    @app.get("/reports/{job_id}")
    async def get_report(job_id: str):
        job = await app.state.db.get_job(job_id)
        if job is None or job["status"] != "done":
            raise HTTPException(status_code=404, detail="report not ready")
        data = await app.state.store.get_bytes("reports", f"{job_id}.json")
        return JSONResponse(content=json.loads(data))

    return app


app = create_app()
```

Note: module-level `app = create_app()` builds the app object (no I/O until lifespan runs), so importing it is safe without the stack.

- [ ] **Step 4: Run integration test (stack up) to verify it passes**

Run: `CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_api.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/api/ tests/integration/test_api.py
git commit -m "feat: add FastAPI job submission and status API"
```

---

### Task 12: Docker image, compose stack, init

**Files:**
- Create: `src/celebvision/workers/__main__.py`
- Create: `src/celebvision/coordinator_main.py`
- Create: `scripts/init_stack.py`
- Create: `docker/Dockerfile`
- Create: `docker/entrypoint.sh`
- Create: `docker-compose.yml`
- Create: `tests/test_compose_config.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable stack. `worker`/`api`/`coordinator` all use one image; the compose `command` selects the role. `init` service creates Kafka topics, applies `schema.sql`, and creates MinIO buckets, then exits.
  - `scripts/init_stack.py`: `async main()` — create topics `["ingest.requested","scenes.requested","transcribe.requested","faces.requested","mentions.requested","aggregate.requested","stage.events"]` (via aiokafka admin), apply DB schema, ensure MinIO buckets `["media","keyframes","reports"]`.
  - `celebvision/workers/__main__.py` (add): dispatch `python -m celebvision.workers <stage>` → wires `MessageBus`, `Database`, `BlobStore`, factories, and calls `run_worker(stage, bus, ctx, HANDLERS[stage])`.
  - `celebvision/coordinator __main__`: `python -m celebvision.coordinator` runs `run_coordinator`.

- [ ] **Step 1: Write the worker/coordinator entrypoints**

```python
# src/celebvision/workers/__main__.py
import asyncio
import sys
from celebvision.config import Settings
from celebvision.bus import MessageBus
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.workers.base import WorkerContext, run_worker
from celebvision.factories import build_inference_client, build_llm_client
from celebvision.workers.ingest import handle_ingest
from celebvision.workers.scene import handle_scene
from celebvision.workers.transcribe import handle_transcribe
from celebvision.workers.face import handle_face
from celebvision.workers.mention import handle_mention
from celebvision.workers.aggregate import handle_aggregate

HANDLERS = {
    "ingest": handle_ingest, "scene": handle_scene, "transcribe": handle_transcribe,
    "face": handle_face, "mention": handle_mention, "aggregate": handle_aggregate,
}
# topic stage name -> handler key (topics use the request stage names)
TOPIC_STAGE = {
    "ingest": "ingest", "scenes": "scene", "transcribe": "transcribe",
    "faces": "face", "mentions": "mention", "aggregate": "aggregate",
}


async def main(stage_topic: str) -> None:
    settings = Settings.from_env()
    db = Database(settings.postgres_dsn)
    await db.connect()
    bus = MessageBus(settings.kafka_bootstrap)
    await bus.start()
    store = BlobStore(settings.minio_endpoint, settings.minio_access_key,
                      settings.minio_secret_key)
    ctx = WorkerContext(db=db, storage=store,
                        inference=build_inference_client(settings),
                        llm=build_llm_client(settings), settings=settings)
    handler = HANDLERS[TOPIC_STAGE[stage_topic]]
    try:
        await run_worker(stage_topic, bus, ctx, handler)
    finally:
        await bus.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
```

Note: `run_worker(stage_topic, ...)` consumes `f"{stage_topic}.requested"` and emits completion events with `stage=stage_topic` (matching the Coordinator's `scenes`/`faces`/`mentions` naming). Handler keys map through `TOPIC_STAGE`.

```python
# src/celebvision/coordinator_main.py
import asyncio
from celebvision.config import Settings
from celebvision.bus import MessageBus
from celebvision.db import Database
from celebvision.coordinator import run_coordinator


async def main() -> None:
    settings = Settings.from_env()
    db = Database(settings.postgres_dsn)
    await db.connect()
    bus = MessageBus(settings.kafka_bootstrap)
    await bus.start()
    try:
        await run_coordinator(bus, db)
    finally:
        await bus.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Write init script**

```python
# scripts/init_stack.py
import asyncio
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.storage import BlobStore

TOPICS = ["ingest.requested", "scenes.requested", "transcribe.requested",
          "faces.requested", "mentions.requested", "aggregate.requested",
          "stage.events"]
BUCKETS = ["media", "keyframes", "reports"]


async def _retry(coro_factory, attempts=30, delay=2.0):
    last = None
    for _ in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 - startup readiness retry
            last = e
            await asyncio.sleep(delay)
    raise last


async def _create_topics(bootstrap: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        try:
            await admin.create_topics(
                [NewTopic(t, num_partitions=3, replication_factor=1) for t in TOPICS])
        except Exception as e:  # topic-exists is fine
            print(f"topics: {e}")
    finally:
        await admin.close()


async def _apply_schema(dsn: str) -> None:
    db = Database(dsn)
    await db.connect()
    await db.apply_schema()
    await db.close()


async def _make_buckets(s: Settings) -> None:
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    for b in BUCKETS:
        await store.ensure_bucket(b)


async def main() -> None:
    s = Settings.from_env()
    await _retry(lambda: _create_topics(s.kafka_bootstrap))
    await _retry(lambda: _apply_schema(s.postgres_dsn))
    await _retry(lambda: _make_buckets(s))
    print("init complete")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Write Dockerfile + entrypoint**

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY sql ./sql
COPY scripts ./scripts
COPY docker ./docker
RUN pip install --no-cache-dir -e '.[service,media]' && chmod +x docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
```

```bash
# docker/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail
role="${1:-api}"
case "$role" in
  api)         exec uvicorn celebvision.api.app:app --host 0.0.0.0 --port 8000 ;;
  coordinator) exec python -m celebvision.coordinator_main ;;
  worker)      exec python -m celebvision.workers "${2}" ;;
  init)        exec python scripts/init_stack.py ;;
  *) echo "unknown role: $role"; exit 1 ;;
esac
```

Mark it executable in git too: `chmod +x docker/entrypoint.sh` before committing.

- [ ] **Step 4: Write docker-compose.yml**

```yaml
# docker-compose.yml
services:
  redpanda:
    image: redpandadata/redpanda:v24.2.7
    command:
      - redpanda start
      - --smp 1
      - --overprovisioned
      - --kafka-addr PLAINTEXT://0.0.0.0:19092
      - --advertise-kafka-addr PLAINTEXT://redpanda:19092
    ports: ["19092:19092"]
    healthcheck:
      test: ["CMD-SHELL", "rpk cluster info || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 10

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: celeb
      POSTGRES_PASSWORD: celeb
      POSTGRES_DB: celeb
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U celeb"]
      interval: 5s
      timeout: 5s
      retries: 10

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    # No healthcheck: the minio server image ships neither curl nor mc. `init`
    # depends on it as service_started and retries connections (see init_stack.py).

  init:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["init"]
    environment: &appenv
      KAFKA_BOOTSTRAP: redpanda:19092
      POSTGRES_DSN: postgresql://celeb:celeb@postgres:5432/celeb
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      INFERENCE_BACKEND: stub
      LLM_BACKEND: stub
      WATCHLIST_BACKEND: pg
    depends_on:
      redpanda: {condition: service_healthy}
      postgres: {condition: service_healthy}
      minio: {condition: service_started}

  api:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["api"]
    environment: *appenv
    ports: ["8000:8000"]
    depends_on:
      init: {condition: service_completed_successfully}

  coordinator:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["coordinator"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-ingest:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "ingest"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-scene:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "scenes"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-transcribe:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "transcribe"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-face:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "faces"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-mention:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "mentions"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}

  worker-aggregate:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: ["worker", "aggregate"]
    environment: *appenv
    depends_on:
      init: {condition: service_completed_successfully}
```

- [ ] **Step 5: Write the compose-validity test**

```python
# tests/test_compose_config.py
import shutil
import subprocess
import pytest


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_is_valid():
    result = subprocess.run(["docker", "compose", "config"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "redpanda" in result.stdout
    assert "worker-aggregate" in result.stdout
```

- [ ] **Step 6: Run + commit**

Run: `chmod +x docker/entrypoint.sh && .venv/bin/python -m pytest tests/test_compose_config.py -v`
Expected: PASS (1 passed). (Validates compose file parses; does not build images.)

```bash
git add docker/ docker-compose.yml scripts/init_stack.py src/celebvision/workers/__main__.py src/celebvision/coordinator_main.py tests/test_compose_config.py
git commit -m "feat: add Dockerfile, docker-compose stack, and init"
```

---

### Task 13: End-to-end integration + milestone verification

**Files:**
- Create: `tests/integration/test_e2e.py`
- Create: `docs/M2_RUNBOOK.md`

**Interfaces:**
- Consumes: everything. Drives the full DAG **in-process over the real infra** (host ffmpeg + Docker infra), using stub inference/LLM, to prove the messaging + storage + coordinator wiring end-to-end without the app containers.

- [ ] **Step 1: Write the E2E test (requires_stack)**

```python
# tests/integration/test_e2e.py
import asyncio
import json
import subprocess
import uuid
import pytest
from celebvision.config import Settings
from celebvision.db import Database
from celebvision.storage import BlobStore
from celebvision.bus import MessageBus, Message
from celebvision.workers.base import WorkerContext, run_worker
from celebvision.workers.__main__ import HANDLERS, TOPIC_STAGE
from celebvision.coordinator import run_coordinator
from celebvision.inference.stub_client import StubInferenceClient, STUB_EMBEDDING
from celebvision.llm.stub_client import StubLLMClient
from celebvision.watchlist.pg_index import PgVectorWatchlistIndex

pytestmark = pytest.mark.requires_stack


async def _run_bg(coro):
    task = asyncio.create_task(coro)
    return task


async def test_full_pipeline_end_to_end(tmp_path):
    s = Settings.from_env()
    db = Database(s.postgres_dsn)
    await db.connect()
    await db.apply_schema()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    for b in ("media", "keyframes", "reports"):
        await store.ensure_bucket(b)
    bus = MessageBus(s.kafka_bootstrap)
    await bus.start()
    ctx = WorkerContext(db=db, storage=store, inference=StubInferenceClient(),
                        llm=StubLLMClient(), settings=s)

    # make a real 3s test video
    video = str(tmp_path / "sample.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "3", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "3", "-shortest", video], check=True)

    wid = "wl-" + uuid.uuid4().hex[:8]
    await PgVectorWatchlistIndex(db, wid).add("messi", "Lionel Messi", ["messi"],
                                              [list(STUB_EMBEDDING)])

    jid = str(uuid.uuid4())
    await db.create_job(jid, "file", video, wid, ["goal"])

    # launch coordinator + all workers as background tasks
    tasks = [await _run_bg(run_coordinator(bus, db))]
    for topic in TOPIC_STAGE:
        tasks.append(await _run_bg(run_worker(topic, bus, ctx, HANDLERS[TOPIC_STAGE[topic]])))

    # kick off
    await bus.produce("ingest.requested", jid, Message(job_id=jid, stage="ingest"))

    # poll until done or timeout
    for _ in range(120):
        job = await db.get_job(jid)
        if job["status"] in ("done", "failed"):
            break
        await asyncio.sleep(0.5)

    for t in tasks:
        t.cancel()
    try:
        job = await db.get_job(jid)
        assert job["status"] == "done", f"job ended {job['status']}: {job['error']}"
        report = json.loads(await store.get_bytes("reports", f"{jid}.json"))
        assert report["job_id"] == jid
        assert "messi" in [c["canonical_id"] for c in report["celebrity_index"]]
    finally:
        await bus.stop()
        await db.close()
```

- [ ] **Step 2: Run the E2E test (full stack up)**

Run:
```bash
docker compose up -d redpanda postgres minio
CELEBVISION_STACK=1 .venv/bin/python scripts/init_stack.py   # create topics/buckets/schema on host-mapped ports
CELEBVISION_STACK=1 .venv/bin/python -m pytest tests/integration/test_e2e.py -v
```
Expected: PASS (1 passed). (Uses host ffmpeg + host-run workers against Docker infra on localhost. Note: for host runs, `KAFKA_BOOTSTRAP=localhost:19092`, `POSTGRES_DSN=postgresql://celeb:celeb@localhost:5432/celeb`, `MINIO_ENDPOINT=http://localhost:9000` — the defaults.)

- [ ] **Step 3: Write the runbook**

```markdown
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
```

- [ ] **Step 4: Full-suite gate + commit**

Run: `.venv/bin/python -m pytest -q` (unit subset green; requires_stack skipped) and `.venv/bin/ruff check src tests scripts`.
Then, stack up: `CELEBVISION_STACK=1 .venv/bin/python -m pytest -q`.
Expected: both green.

```bash
git add tests/integration/test_e2e.py docs/M2_RUNBOOK.md
git commit -m "feat: add end-to-end integration test and M2 runbook"
```

---

## Milestone exit criteria

- [ ] Unit subset (`.venv/bin/python -m pytest -q`, no stack) green; `requires_stack` tests skipped cleanly.
- [ ] With stack up (`CELEBVISION_STACK=1`), all integration tests green, including the E2E DAG run producing a `done` job + schema-valid report with `celebrity_index`.
- [ ] `docker compose config` valid; `docker compose up --build` boots the full stack (manual smoke: `POST /jobs` a file/YouTube source reaches `done`, `GET /reports/{id}` returns the report).
- [ ] `ruff check src tests scripts` clean.
- [ ] No M1 stage logic changed except `media.probe_duration` and the single-shot `end_s` fix in the scene worker.

## Handoff to M3

M3 (Triton serving) adds a `TritonClient` implementing `InferenceClient` and `INFERENCE_BACKEND=triton`; the worker image adds a `local`/Triton build. M4 adds per-stage retries, `<stage>.dlq` topics, metrics, and the precision/recall eval harness. Report `duration_s`/`completed_at` threading and scene-level `keyword_hits` in the aggregate can be tightened then.
