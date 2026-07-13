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
