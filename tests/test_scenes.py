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


def test_build_scene_windows_multiple_frames_evenly_spaced(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append((t_s, out_path))
        return out_path
    windows = build_scene_windows("v.mp4", [(0.0, 8.0)], str(tmp_path),
                                  keyframe_fn=fake_keyframe, frames_per_scene=3)
    # 3 evenly-spaced frames within (0,8) at fractions 1/4, 2/4, 3/4 -> 2, 4, 6
    assert [t for t, _ in grabbed] == [2.0, 4.0, 6.0]
    assert len(windows[0].keyframes) == 3
    assert windows[0].keyframes == [g[1] for g in grabbed]
    assert len(set(windows[0].keyframes)) == 3  # distinct filenames


def test_build_scene_windows_default_is_single_midpoint(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append(t_s)
        return out_path
    windows = build_scene_windows("v.mp4", [(0.0, 10.0)], str(tmp_path),
                                  keyframe_fn=fake_keyframe)  # default frames_per_scene=1
    assert grabbed == [5.0]
    assert len(windows[0].keyframes) == 1


def test_build_scene_windows_resolves_single_shot_sentinel(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append(t_s)
        return out_path
    # single-shot sentinel [(0,0)] -> resolved via duration_fn to (0, 12), then spread
    windows = build_scene_windows("v.mp4", [(0.0, 0.0)], str(tmp_path),
                                  keyframe_fn=fake_keyframe, frames_per_scene=3,
                                  duration_fn=lambda p: 12.0)
    assert grabbed == [3.0, 6.0, 9.0]      # fractions 1/4, 2/4, 3/4 of 12
    assert windows[0].end_s == 12.0        # resolved duration recorded on the window


def test_build_scene_windows_degenerate_span_no_descending(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append(t_s)
        return out_path
    # inverted boundary -> span clamped to 0, all frames at start_s (no descending/out-of-range)
    build_scene_windows("v.mp4", [(10.0, 5.0)], str(tmp_path),
                        keyframe_fn=fake_keyframe, frames_per_scene=3)
    assert grabbed == [10.0, 10.0, 10.0]
