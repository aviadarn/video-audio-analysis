import json
from celebvision.bus import Message
from celebvision.workers.base import WorkerContext
from celebvision.stages.scenes import SceneWindow
from celebvision.stages.aggregate import build_scene_report, build_report
from celebvision.models import SpokenMention, KeywordHit, OnscreenFace, JobSource
from celebvision.errors import StageError


def _as_list(v):
    return v if isinstance(v, list) else json.loads(v)


async def handle_aggregate(msg: Message, ctx: WorkerContext) -> None:
    job = await ctx.db.get_job(msg.job_id)
    if job is None:
        raise StageError("aggregate", "job not found")
    try:
        rows = await ctx.db.get_scenes(msg.job_id)
        scene_reports = []
        for row in rows:
            window = SceneWindow(scene_id=row["scene_id"], start_s=row["start_s"],
                                 end_s=row["end_s"], keyframes=_as_list(row["keyframes"]))
            mentions = [SpokenMention(**m) for m in _as_list(row["mentions"])]
            keyword_hits = [KeywordHit(**k) for k in _as_list(row["keyword_hits"])]
            faces = [OnscreenFace(**f) for f in _as_list(row["faces"])]
            scene_reports.append(build_scene_report(
                window, row["transcript"], mentions, keyword_hits, faces))
        report = build_report(
            msg.job_id, JobSource(kind=job["source_kind"], locator=job["source_locator"]),
            job["watchlist_id"], "", 0.0, scene_reports)
        await ctx.storage.ensure_bucket("reports")
        await ctx.storage.put_bytes("reports", f"{msg.job_id}.json",
                                    report.model_dump_json(indent=2).encode("utf-8"))
    except StageError:
        raise
    except Exception as e:
        raise StageError("aggregate", str(e)) from e
