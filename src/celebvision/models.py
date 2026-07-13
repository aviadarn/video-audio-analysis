from typing import Literal
from pydantic import BaseModel

SourceKind = Literal["youtube", "hls", "file"]


class JobSource(BaseModel):
    kind: SourceKind
    locator: str


class SpokenMention(BaseModel):
    name: str
    canonical_id: str
    confidence: float
    evidence_span: str


class KeywordHit(BaseModel):
    keyword: str
    count: int
    spans: list[str]


class OnscreenFace(BaseModel):
    name: str
    canonical_id: str
    confidence: float
    bbox: tuple[float, float, float, float]
    keyframe: str


class SceneReport(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    keyframes: list[str]
    transcript: str
    spoken_mentions: list[SpokenMention]
    keyword_hits: list[KeywordHit]
    onscreen_faces: list[OnscreenFace]


class CelebrityIndexEntry(BaseModel):
    name: str
    canonical_id: str
    scenes: list[int]
    modalities: list[str]


class Report(BaseModel):
    job_id: str
    source: JobSource
    duration_s: float
    watchlist_id: str
    completed_at: str
    scenes: list[SceneReport]
    celebrity_index: list[CelebrityIndexEntry]
