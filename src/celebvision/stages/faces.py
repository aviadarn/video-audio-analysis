# src/celebvision/stages/faces.py
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import InferenceClient, WatchlistIndex
from celebvision.models import OnscreenFace


def match_faces_in_scene(scene: SceneWindow, inference: InferenceClient,
                         watchlist: WatchlistIndex, threshold: float = 0.35
                         ) -> list[OnscreenFace]:
    best: dict[str, OnscreenFace] = {}
    for keyframe in scene.keyframes:
        for det in inference.analyze_faces(keyframe):
            hit = watchlist.search(det.embedding, threshold)
            if hit is None:
                continue
            canonical_id, name, score = hit
            face = OnscreenFace(name=name, canonical_id=canonical_id,
                                confidence=score, bbox=det.bbox, keyframe=keyframe)
            if canonical_id not in best or score > best[canonical_id].confidence:
                best[canonical_id] = face
    return list(best.values())
