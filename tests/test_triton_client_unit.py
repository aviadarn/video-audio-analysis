# tests/test_triton_client_unit.py
from types import SimpleNamespace
import numpy as np
from celebvision.inference.triton_client import TritonFaceClient
from celebvision.config import Settings


def test_analyze_faces_maps_detection_and_embedding(monkeypatch, tmp_path):
    # a fake detector returning one face with bbox + 5 landmarks
    face = SimpleNamespace(bbox=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                           kps=np.array([[30, 30], [80, 30], [55, 55],
                                         [35, 80], [75, 80]], dtype=np.float32))
    client = TritonFaceClient(Settings.from_env({}),
                              infer_fn=lambda blob: np.ones((blob.shape[0], 512),
                                                            dtype=np.float32))
    client._detector = SimpleNamespace(get=lambda img: [face])
    # stub image read + norm_crop so no real image/model is needed
    import celebvision.inference.triton_client as mod
    monkeypatch.setattr(mod.cv2 if hasattr(mod, "cv2") else __import__("cv2"),
                        "imread", lambda p: np.zeros((120, 120, 3), dtype=np.uint8),
                        raising=False)
    monkeypatch.setattr("insightface.utils.face_align.norm_crop",
                        lambda img, landmark, image_size=112:
                        np.zeros((112, 112, 3), dtype=np.uint8))

    dets = client.analyze_faces(str(tmp_path / "x.jpg"))
    assert len(dets) == 1
    assert dets[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert len(dets[0].embedding) == 512
    assert abs(np.linalg.norm(dets[0].embedding) - 1.0) < 1e-5  # normalized (all-ones -> unit)
