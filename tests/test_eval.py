# tests/test_eval.py
from celebvision.eval import GroundTruth, score_report
from celebvision.models import (
    Report, SceneReport, JobSource, SpokenMention, OnscreenFace, CelebrityIndexEntry,
)


def _report():
    scene = SceneReport(
        scene_id=0, start_s=0, end_s=5, keyframes=[], transcript="",
        spoken_mentions=[SpokenMention(name="Messi", canonical_id="messi",
                                       confidence=0.9, evidence_span="")],
        keyword_hits=[],
        onscreen_faces=[OnscreenFace(name="Ronaldo", canonical_id="ronaldo",
                                     confidence=0.8, bbox=(0, 0, 1, 1), keyframe="")],
    )
    return Report(job_id="j1", source=JobSource(kind="file", locator="a.mp4"),
                  duration_s=5.0, watchlist_id="w1", completed_at="",
                  scenes=[scene],
                  celebrity_index=[
                      CelebrityIndexEntry(name="Messi", canonical_id="messi",
                                          scenes=[0], modalities=["audio"]),
                      CelebrityIndexEntry(name="Ronaldo", canonical_id="ronaldo",
                                          scenes=[0], modalities=["face"])])


def test_score_report_combined_prf():
    # expected {messi, ronaldo, neymar}; detected combined {messi, ronaldo}
    gt = GroundTruth(job_id="j1", expected=["messi", "ronaldo", "neymar"])
    res = score_report(_report(), gt)
    c = res.video["combined"]
    assert c.tp == 2 and c.fp == 0 and c.fn == 1
    assert abs(c.precision - 1.0) < 1e-9
    assert abs(c.recall - 2/3) < 1e-9

def test_score_report_per_modality():
    gt = GroundTruth(job_id="j1", expected=["messi", "ronaldo"],
                     expected_by_modality={"audio": ["messi"], "face": ["ronaldo"]})
    res = score_report(_report(), gt)
    assert res.video["audio"].f1 == 1.0   # detected audio {messi} == expected {messi}
    assert res.video["face"].f1 == 1.0     # detected face {ronaldo} == expected {ronaldo}

def test_empty_expected_and_detected_is_zero_not_crash():
    gt = GroundTruth(job_id="j1", expected=[])
    from celebvision.models import Report as R, JobSource as JS
    empty = R(job_id="j1", source=JS(kind="file", locator="a"), duration_s=0,
              watchlist_id="w", completed_at="", scenes=[], celebrity_index=[])
    res = score_report(empty, gt)
    assert res.video["combined"].f1 == 0.0
