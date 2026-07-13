from types import SimpleNamespace
import numpy as np
import pytest
from celebvision.inference.local_client import LocalInferenceClient


def test_to_face_detections_maps_bbox_and_embedding():
    face = SimpleNamespace(
        bbox=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        normed_embedding=np.array([0.6, 0.8], dtype=np.float32),
    )
    client = LocalInferenceClient.__new__(LocalInferenceClient)
    dets = client._to_face_detections([face])
    assert len(dets) == 1
    assert dets[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert dets[0].embedding == pytest.approx([0.6, 0.8], abs=1e-6)

def test_to_face_detections_empty():
    client = LocalInferenceClient.__new__(LocalInferenceClient)
    assert client._to_face_detections([]) == []
