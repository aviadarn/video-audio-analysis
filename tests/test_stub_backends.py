from celebvision.inference.stub_client import StubInferenceClient, STUB_EMBEDDING
from celebvision.llm.stub_client import StubLLMClient


def test_stub_transcribe_is_deterministic():
    t = StubInferenceClient().transcribe("ignored.wav")
    words = " ".join(w.text for seg in t.segments for w in seg.words)
    assert "messi" in words.lower()
    assert t.duration_s > 0

def test_stub_faces_returns_stub_embedding():
    dets = StubInferenceClient().analyze_faces("ignored.jpg")
    assert len(dets) == 1
    assert dets[0].embedding == STUB_EMBEDDING
    assert len(STUB_EMBEDDING) == 512

def test_stub_llm_matches_watchlist_name_in_text():
    llm = StubLLMClient()
    out = llm.extract_mentions("messi scored a goal",
                               [("messi", "Lionel Messi", ["messi"])], ["goal"])
    assert out.mentions[0].canonical_id == "messi"
    assert out.keyword_hits[0].keyword == "goal"
    assert out.keyword_hits[0].count == 1

def test_stub_llm_no_match():
    out = StubLLMClient().extract_mentions("nothing here",
                                           [("messi", "Lionel Messi", ["messi"])], [])
    assert out.mentions == []
