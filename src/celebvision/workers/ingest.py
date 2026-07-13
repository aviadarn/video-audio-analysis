import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.media import ingest
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
