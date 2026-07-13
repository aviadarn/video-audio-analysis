# tests/test_faces_stage.py
from celebvision.stages.faces import match_faces_in_scene
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import FaceDetection


class FakeInference:
    def __init__(self, dets): self._dets = dets
    def analyze_faces(self, image_path): return self._dets
    def transcribe(self, audio_path): raise NotImplementedError


class FakeWatchlist:
    def __init__(self, table): self._table = table  # embedding-tuple -> (id,name,score)
    def search(self, embedding, threshold):
        return self._table.get(tuple(embedding))
    def names(self): return []


def test_matches_and_dedupes_by_identity():
    dets = [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0]),
            FaceDetection(bbox=(2, 2, 3, 3), embedding=[0.9, 0.1])]
    wl = FakeWatchlist({
        (1.0, 0.0): ("messi", "Lionel Messi", 0.8),
        (0.9, 0.1): ("messi", "Lionel Messi", 0.6),  # same identity, lower score
    })
    scene = SceneWindow(scene_id=0, start_s=0, end_s=5, keyframes=["k.jpg"])
    faces = match_faces_in_scene(scene, FakeInference(dets), wl, threshold=0.35)
    assert len(faces) == 1
    assert faces[0].canonical_id == "messi"
    assert faces[0].confidence == 0.8  # highest kept

def test_no_match_returns_empty():
    dets = [FaceDetection(bbox=(0, 0, 1, 1), embedding=[0.0, 1.0])]
    scene = SceneWindow(scene_id=0, start_s=0, end_s=5, keyframes=["k.jpg"])
    faces = match_faces_in_scene(scene, FakeInference(dets), FakeWatchlist({}),
                                 threshold=0.35)
    assert faces == []
