import celebvision.pipeline as pipeline_mod
from celebvision.pipeline import run_pipeline
from celebvision.interfaces import (
    Transcript, TranscriptSegment, Word, FaceDetection, MentionExtraction,
)
from celebvision.models import SpokenMention
from celebvision.enroll import enroll_identity
from celebvision.watchlist.local_index import LocalWatchlistIndex


class FakeInference:
    def transcribe(self, audio_path):
        return Transcript(segments=[TranscriptSegment(
            start_s=0.0, end_s=5.0, text="Messi scored",
            words=[Word(text="Messi", start_s=1.0, end_s=1.5),
                   Word(text="scored", start_s=1.5, end_s=2.0)])])
    def analyze_faces(self, image_path):
        return [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0])]


class FakeLLM:
    def extract_mentions(self, transcript_text, watchlist, keywords):
        return MentionExtraction(
            mentions=[SpokenMention(name="Lionel Messi", canonical_id="messi",
                                    confidence=0.9, evidence_span=transcript_text)],
            keyword_hits=[])


class FakeWatchlist:
    watchlist_id = "w1"
    def search(self, embedding, threshold):
        return ("messi", "Lionel Messi", 0.8)
    def names(self): return [("messi", "Lionel Messi", ["messi"])]


def test_run_pipeline_end_to_end(tmp_path, monkeypatch):
    # Stub ingest + scene detection + keyframe so no ffmpeg/network is needed.
    monkeypatch.setattr(pipeline_mod, "ingest",
                        lambda source, workdir, **k: ("v.mp4", "a.wav"))
    monkeypatch.setattr(pipeline_mod, "detect_scenes",
                        lambda video_path, **k: [(0.0, 5.0)])
    monkeypatch.setattr(pipeline_mod, "extract_keyframe",
                        lambda v, t, out, **k: out)

    report = run_pipeline(
        locator="/tmp/a.mp4", watchlist=FakeWatchlist(), keywords=["scored"],
        inference=FakeInference(), llm=FakeLLM(), workdir=str(tmp_path),
        job_id="j1", completed_at="2026-07-13T00:00:00Z",
    )
    assert report.job_id == "j1"
    assert len(report.scenes) == 1
    scene = report.scenes[0]
    assert scene.spoken_mentions[0].canonical_id == "messi"
    assert scene.onscreen_faces[0].canonical_id == "messi"
    assert report.celebrity_index[0].modalities == ["audio", "face"]


def test_enroll_identity_adds_embeddings():
    class FI:
        def analyze_faces(self, image_path):
            return [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0])]
        def transcribe(self, audio_path): raise NotImplementedError
    idx = LocalWatchlistIndex("w1")
    enroll_identity(idx, FI(), "messi", "Lionel Messi", ["messi"],
                    ["a.jpg", "b.jpg"])
    assert idx.names() == [("messi", "Lionel Messi", ["messi"])]
    assert idx.search([1.0, 0.0], threshold=0.5)[0] == "messi"
