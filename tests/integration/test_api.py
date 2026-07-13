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
