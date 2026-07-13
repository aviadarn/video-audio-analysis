import subprocess
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
