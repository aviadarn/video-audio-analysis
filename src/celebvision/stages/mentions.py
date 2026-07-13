from celebvision.interfaces import Transcript, LLMClient, WatchlistIndex
from celebvision.stages.scenes import SceneWindow
from celebvision.models import SpokenMention, KeywordHit


def scene_transcript_text(transcript: Transcript, scene: SceneWindow) -> str:
    if scene.end_s == 0.0:  # single-shot sentinel: whole transcript
        return " ".join(w.text for seg in transcript.segments for w in seg.words) \
            or " ".join(seg.text for seg in transcript.segments)
    words = transcript.words_in(scene.start_s, scene.end_s)
    return " ".join(w.text for w in words)


def extract_scene_mentions(text: str, watchlist: WatchlistIndex,
                           keywords: list[str], llm: LLMClient
                           ) -> tuple[list[SpokenMention], list[KeywordHit]]:
    if not text.strip():
        return [], []
    result = llm.extract_mentions(text, watchlist.names(), keywords)
    return result.mentions, result.keyword_hits
