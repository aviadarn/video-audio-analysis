import asyncio
import os
import shutil
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.errors import StageError


async def handle_transcribe(msg: Message, ctx: WorkerContext) -> None:
    assets = await ctx.db.get_job_assets(msg.job_id)
    if assets is None or not assets["audio_key"]:
        raise StageError("transcribe", "no audio asset")
    tmp = tempfile.mkdtemp(prefix=f"transcribe-{msg.job_id}-")
    try:
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
