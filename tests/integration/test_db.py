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
