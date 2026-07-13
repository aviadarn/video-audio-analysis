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
        assert report["completed_at"] != ""
        assert report["duration_s"] > 0.0
    finally:
        await db.close()
