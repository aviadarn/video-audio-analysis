# tests/test_coordinator_decide.py
from celebvision.bus import Message
from celebvision.coordinator import decide_next, JobState


def _state(**kw):
    base = dict(status="running", expected_scene_count=None, ingest_done=False,
                scenes_done=False, transcribe_done=False, faces_done=0, mentions_done=0)
    base.update(kw)
    return JobState(**base)


def test_ingest_fans_out_scenes_and_transcribe():
    out = decide_next(Message(job_id="j", stage="ingest"), _state(ingest_done=True))
    assert sorted(m.stage for m in out) == ["scenes", "transcribe"]

def test_scenes_alone_does_not_fan_out():
    out = decide_next(Message(job_id="j", stage="scenes"),
                      _state(scenes_done=True, transcribe_done=False,
                             expected_scene_count=2))
    assert out == []

def test_scenes_and_transcribe_fan_out_per_scene():
    out = decide_next(Message(job_id="j", stage="transcribe"),
                      _state(scenes_done=True, transcribe_done=True,
                             expected_scene_count=2))
    stages = sorted((m.stage, m.scene_id) for m in out)
    assert stages == [("faces", 0), ("faces", 1), ("mentions", 0), ("mentions", 1)]

def test_all_scene_stages_done_triggers_aggregate():
    out = decide_next(Message(job_id="j", stage="faces"),
                      _state(expected_scene_count=2, faces_done=2, mentions_done=2))
    assert [m.stage for m in out] == ["aggregate"]

def test_partial_scene_progress_waits():
    out = decide_next(Message(job_id="j", stage="faces"),
                      _state(expected_scene_count=2, faces_done=2, mentions_done=1))
    assert out == []

def test_failed_job_emits_nothing():
    out = decide_next(Message(job_id="j", stage="ingest"),
                      _state(status="failed", ingest_done=True))
    assert out == []

def test_done_job_emits_nothing():
    out = decide_next(Message(job_id="j", stage="faces"),
                      _state(status="done", expected_scene_count=2,
                             faces_done=2, mentions_done=2))
    assert out == []

def test_zero_scenes_triggers_aggregate():
    out = decide_next(Message(job_id="j", stage="transcribe"),
                      _state(scenes_done=True, transcribe_done=True,
                             expected_scene_count=0))
    assert [m.stage for m in out] == ["aggregate"]
