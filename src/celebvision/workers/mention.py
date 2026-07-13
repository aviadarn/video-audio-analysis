import json
import os
import shutil
import tempfile
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
    tmp = tempfile.mkdtemp(prefix=f"mention-{msg.job_id}-{msg.scene_id}-")
    try:
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
