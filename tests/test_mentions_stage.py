from celebvision.stages.mentions import scene_transcript_text, extract_scene_mentions
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import Transcript, TranscriptSegment, Word, MentionExtraction
from celebvision.models import SpokenMention, KeywordHit
from celebvision.llm.anthropic_client import AnthropicLLMClient


def _transcript():
    words = [Word(text="Messi", start_s=1.0, end_s=1.5),
             Word(text="scored", start_s=1.5, end_s=2.0),
             Word(text="Ronaldo", start_s=6.0, end_s=6.5)]
    return Transcript(segments=[TranscriptSegment(start_s=0, end_s=7,
                      text="Messi scored Ronaldo", words=words)])


def test_scene_transcript_text_windows_words():
    scene = SceneWindow(scene_id=0, start_s=0.0, end_s=5.0, keyframes=[])
    assert scene_transcript_text(_transcript(), scene) == "Messi scored"

def test_scene_transcript_text_single_shot_uses_all():
    scene = SceneWindow(scene_id=0, start_s=0.0, end_s=0.0, keyframes=[])
    assert scene_transcript_text(_transcript(), scene) == "Messi scored Ronaldo"

class FakeLLM:
    def extract_mentions(self, transcript_text, watchlist, keywords):
        return MentionExtraction(
            mentions=[SpokenMention(name="Lionel Messi", canonical_id="messi",
                                    confidence=0.95, evidence_span="Messi scored")],
            keyword_hits=[KeywordHit(keyword="scored", count=1, spans=["Messi scored"])],
        )

class FakeWatchlist:
    def names(self): return [("messi", "Lionel Messi", ["messi"])]
    def search(self, e, t): return None

def test_extract_scene_mentions_delegates_to_llm():
    mentions, hits = extract_scene_mentions("Messi scored", FakeWatchlist(),
                                            ["scored"], FakeLLM())
    assert mentions[0].canonical_id == "messi"
    assert hits[0].keyword == "scored"

def test_parse_response_maps_json():
    payload = {"mentions": [{"name": "Lionel Messi", "canonical_id": "messi",
                             "confidence": 0.9, "evidence_span": "Messi scored"}],
               "keyword_hits": [{"keyword": "scored", "count": 2,
                                 "spans": ["a", "b"]}]}
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    result = client._parse_response(payload, ["scored"])
    assert result.mentions[0].canonical_id == "messi"
    assert result.keyword_hits[0].count == 2

def test_parse_response_tolerates_missing_keys():
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    result = client._parse_response({}, [])
    assert result.mentions == []
    assert result.keyword_hits == []
