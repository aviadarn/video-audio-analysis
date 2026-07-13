import asyncio
import os
import tempfile
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.factories import build_watchlist
from celebvision.errors import StageError

_FACE_THRESHOLD = 0.35


async def handle_face(msg: Message, ctx: WorkerContext) -> None:
    scene = await ctx.db.get_scene(msg.job_id, msg.scene_id)
    if scene is None:
        raise StageError("face", f"scene {msg.scene_id} not found")
    job = await ctx.db.get_job(msg.job_id)
    try:
        keyframes = scene["keyframes"]
        if isinstance(keyframes, str):
            import json
            keyframes = json.loads(keyframes)
        watchlist = await build_watchlist(ctx.settings, ctx.db, job["watchlist_id"])
        tmp = tempfile.mkdtemp(prefix=f"face-{msg.job_id}-{msg.scene_id}-")
        best: dict[str, dict] = {}
        for i, key in enumerate(keyframes):
            local = os.path.join(tmp, f"kf{i}.jpg")
            await ctx.storage.get_file("keyframes", key, local)
            for det in await asyncio.to_thread(ctx.inference.analyze_faces, local):
                hit = await watchlist.search(det.embedding, _FACE_THRESHOLD)
                if hit is None:
                    continue
                cid, name, score = hit
                if cid not in best or score > best[cid]["confidence"]:
                    best[cid] = {"name": name, "canonical_id": cid,
                                 "confidence": score, "bbox": list(det.bbox),
                                 "keyframe": key}
        await ctx.db.set_scene_faces(msg.job_id, msg.scene_id, list(best.values()))
    except StageError:
        raise
    except Exception as e:
        raise StageError("face", str(e)) from e
