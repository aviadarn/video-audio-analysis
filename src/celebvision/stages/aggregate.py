from celebvision.stages.scenes import SceneWindow
from celebvision.models import (
    SpokenMention, KeywordHit, OnscreenFace, SceneReport, Report,
    CelebrityIndexEntry, JobSource,
)


def build_scene_report(scene: SceneWindow, transcript_text: str,
                       mentions: list[SpokenMention], keyword_hits: list[KeywordHit],
                       faces: list[OnscreenFace]) -> SceneReport:
    return SceneReport(
        scene_id=scene.scene_id, start_s=scene.start_s, end_s=scene.end_s,
        keyframes=scene.keyframes, transcript=transcript_text,
        spoken_mentions=mentions, keyword_hits=keyword_hits, onscreen_faces=faces,
    )


def build_celebrity_index(scene_reports: list[SceneReport]) -> list[CelebrityIndexEntry]:
    acc: dict[str, dict] = {}
    for sr in scene_reports:
        for m in sr.spoken_mentions:
            e = acc.setdefault(m.canonical_id,
                               {"name": m.name, "scenes": set(), "modalities": set()})
            e["scenes"].add(sr.scene_id)
            e["modalities"].add("audio")
        for f in sr.onscreen_faces:
            e = acc.setdefault(f.canonical_id,
                               {"name": f.name, "scenes": set(), "modalities": set()})
            e["scenes"].add(sr.scene_id)
            e["modalities"].add("face")
    return [CelebrityIndexEntry(canonical_id=cid, name=v["name"],
                                scenes=sorted(v["scenes"]),
                                modalities=sorted(v["modalities"]))
            for cid, v in acc.items()]


def build_report(job_id: str, source: JobSource, watchlist_id: str,
                 completed_at: str, duration_s: float,
                 scene_reports: list[SceneReport]) -> Report:
    return Report(
        job_id=job_id, source=source, duration_s=duration_s,
        watchlist_id=watchlist_id, completed_at=completed_at,
        scenes=scene_reports, celebrity_index=build_celebrity_index(scene_reports),
    )
