import asyncio
import os
import shutil
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
    tmp = tempfile.mkdtemp(prefix=f"scene-{msg.job_id}-")
    try:
        video_path = os.path.join(tmp, "video.mp4")
        await ctx.storage.get_file("media", assets["video_key"], video_path)

        boundaries = await asyncio.to_thread(detect_scenes, video_path)
        if boundaries == [(0.0, 0.0)]:  # single-shot sentinel -> real duration
            boundaries = [(0.0, await asyncio.to_thread(probe_duration, video_path))]

        windows = await asyncio.to_thread(
            build_scene_windows, video_path, boundaries, tmp,
            frames_per_scene=ctx.settings.keyframes_per_scene)
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
