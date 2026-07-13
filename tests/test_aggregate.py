# tests/test_aggregate.py
from celebvision.stages.aggregate import (
    build_scene_report, build_celebrity_index, build_report,
)
from celebvision.stages.scenes import SceneWindow
from celebvision.models import SpokenMention, OnscreenFace, JobSource


def _scene_report(sid, mention_id=None, face_id=None):
    scene = SceneWindow(scene_id=sid, start_s=sid, end_s=sid + 1, keyframes=["k.jpg"])
    mentions = ([SpokenMention(name="Lionel Messi", canonical_id=mention_id,
                 confidence=0.9, evidence_span="x")] if mention_id else [])
    faces = ([OnscreenFace(name="Lionel Messi", canonical_id=face_id, confidence=0.7,
              bbox=(0, 0, 1, 1), keyframe="k.jpg")] if face_id else [])
    return build_scene_report(scene, "text", mentions, [], faces)


def test_celebrity_index_merges_modalities_across_scenes():
    reports = [_scene_report(0, mention_id="messi"),
               _scene_report(1, face_id="messi"),
               _scene_report(2, face_id="ronaldo")]
    index = build_celebrity_index(reports)
    by_id = {e.canonical_id: e for e in index}
    assert sorted(by_id["messi"].scenes) == [0, 1]
    assert by_id["messi"].modalities == ["audio", "face"]
    assert by_id["ronaldo"].modalities == ["face"]
    assert by_id["ronaldo"].scenes == [2]

def test_build_report_assembles_everything():
    reports = [_scene_report(0, mention_id="messi")]
    report = build_report("j1", JobSource(kind="file", locator="a.mp4"),
                          "w1", "2026-07-13T00:00:00Z", 10.0, reports)
    assert report.job_id == "j1"
    assert len(report.scenes) == 1
    assert report.celebrity_index[0].canonical_id == "messi"
