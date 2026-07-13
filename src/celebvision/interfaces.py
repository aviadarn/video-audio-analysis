from typing import Protocol
from pydantic import BaseModel
from celebvision.models import SpokenMention, KeywordHit


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    def words_in(self, start_s: float, end_s: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            for w in seg.words:
                mid = (w.start_s + w.end_s) / 2.0
                if start_s <= mid < end_s:
                    out.append(w)
        return out


class FaceDetection(BaseModel):
    bbox: tuple[float, float, float, float]
    embedding: list[float]


class MentionExtraction(BaseModel):
    mentions: list[SpokenMention]
    keyword_hits: list[KeywordHit]


class InferenceClient(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...
    def analyze_faces(self, image_path: str) -> list[FaceDetection]: ...


class LLMClient(Protocol):
    def extract_mentions(
        self,
        transcript_text: str,
        watchlist: list[tuple[str, str, list[str]]],
        keywords: list[str],
    ) -> MentionExtraction: ...


class WatchlistIndex(Protocol):
    watchlist_id: str

    def search(
        self, embedding: list[float], threshold: float
    ) -> tuple[str, str, float] | None: ...
    def names(self) -> list[tuple[str, str, list[str]]]: ...
