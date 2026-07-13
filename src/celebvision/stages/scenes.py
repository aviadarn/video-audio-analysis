import os
from pydantic import BaseModel
from celebvision.media import extract_keyframe


class SceneWindow(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    keyframes: list[str]


def _default_detector(video_path: str) -> list[tuple[float, float]]:
    from scenedetect import detect, AdaptiveDetector
    scene_list = detect(video_path, AdaptiveDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def detect_scenes(video_path: str, detector_fn=_default_detector) -> list[tuple[float, float]]:
    boundaries = detector_fn(video_path)
    if not boundaries:  # single-shot video -> one scene covering whole file
        return [(0.0, 0.0)]
    return boundaries


def build_scene_windows(video_path: str, boundaries: list[tuple[float, float]],
                        workdir: str, keyframe_fn=extract_keyframe) -> list[SceneWindow]:
    os.makedirs(workdir, exist_ok=True)
    windows: list[SceneWindow] = []
    for i, (start_s, end_s) in enumerate(boundaries):
        mid = (start_s + end_s) / 2.0
        out = os.path.join(workdir, f"scene_{i:04d}.jpg")
        keyframe_fn(video_path, mid, out)
        windows.append(SceneWindow(scene_id=i, start_s=start_s, end_s=end_s,
                                   keyframes=[out]))
    return windows
