# tests/test_scenes.py
from celebvision.stages.scenes import build_scene_windows, SceneWindow


def test_build_scene_windows_extracts_midpoint_keyframe(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append((t_s, out_path))
        return out_path
    boundaries = [(0.0, 4.0), (4.0, 10.0)]
    windows = build_scene_windows("v.mp4", boundaries, str(tmp_path),
                                  keyframe_fn=fake_keyframe)
    assert [w.scene_id for w in windows] == [0, 1]
    assert isinstance(windows[0], SceneWindow)
    # midpoints 2.0 and 7.0
    assert grabbed[0][0] == 2.0
    assert grabbed[1][0] == 7.0
    assert windows[0].keyframes == [grabbed[0][1]]
