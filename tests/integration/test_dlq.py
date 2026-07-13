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
    import os
    s = Settings.from_env({**os.environ, "MAX_ATTEMPTS": "1"})   # 1 => straight to DLQ on first failure
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
