from celebvision.models import (
    JobSource, SpokenMention, OnscreenFace, SceneReport, Report,
)

def test_report_round_trip():
    scene = SceneReport(
        scene_id=0, start_s=0.0, end_s=5.0, keyframes=["k0.jpg"],
        transcript="messi scored", spoken_mentions=[
            SpokenMention(name="Lionel Messi", canonical_id="messi",
                          confidence=0.9, evidence_span="messi scored")],
        keyword_hits=[], onscreen_faces=[
            OnscreenFace(name="Lionel Messi", canonical_id="messi",
                         confidence=0.7, bbox=(1, 2, 3, 4), keyframe="k0.jpg")],
    )
    report = Report(
        job_id="j1", source=JobSource(kind="file", locator="a.mp4"),
        duration_s=5.0, watchlist_id="w1", completed_at="2026-07-13T00:00:00Z",
        scenes=[scene], celebrity_index=[],
    )
    dumped = report.model_dump_json()
    again = Report.model_validate_json(dumped)
    assert again.scenes[0].spoken_mentions[0].canonical_id == "messi"
    assert again.scenes[0].onscreen_faces[0].bbox == (1, 2, 3, 4)

def test_source_kind_rejects_bad_value():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JobSource(kind="ftp", locator="x")
