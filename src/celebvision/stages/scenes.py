import os
from pydantic import BaseModel
from celebvision.media import extract_keyframe, probe_duration


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
                        workdir: str, keyframe_fn=extract_keyframe,
                        frames_per_scene: int = 1,
                        duration_fn=probe_duration) -> list[SceneWindow]:
    """Extract ``frames_per_scene`` keyframes per scene, evenly spaced across the
    scene's span (fractions 1/(N+1) .. N/(N+1)). N=1 yields the scene midpoint,
    preserving the original single-keyframe behavior. More frames give the face
    stage more chances to catch an on-screen match.

    The single-shot sentinel ``[(0.0, 0.0)]`` from ``detect_scenes`` is resolved
    here (once, for all callers) to ``[(0.0, duration_fn(video_path))]`` so the
    keyframes spread across the whole video instead of collapsing to t=0."""
    os.makedirs(workdir, exist_ok=True)
    if boundaries == [(0.0, 0.0)]:  # single-shot sentinel -> real duration
        boundaries = [(0.0, duration_fn(video_path))]
    n = max(1, frames_per_scene)
    windows: list[SceneWindow] = []
    for i, (start_s, end_s) in enumerate(boundaries):
        span = max(0.0, end_s - start_s)  # guard inverted/zero-duration boundaries
        keyframes: list[str] = []
        for j in range(n):
            t = start_s + (j + 1) / (n + 1) * span
            out = os.path.join(workdir, f"scene_{i:04d}_{j:02d}.jpg")
            keyframe_fn(video_path, t, out)
            keyframes.append(out)
        windows.append(SceneWindow(scene_id=i, start_s=start_s, end_s=end_s,
                                   keyframes=keyframes))
    return windows
