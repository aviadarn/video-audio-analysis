import os
import numpy as np
import pytest

pytestmark = pytest.mark.slow

_ARCFACE = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")


def _onnx_infer_fn():
    import onnxruntime as ort
    sess = ort.InferenceSession(_ARCFACE, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    def infer(blob):
        return sess.run(None, {iname: blob})[0]
    return infer


def test_embed_matches_insightface_recognition():
    from celebvision.inference.triton_client import TritonFaceClient
    from celebvision.config import Settings
    from insightface.model_zoo import get_model

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)

    client = TritonFaceClient(Settings.from_env({}), infer_fn=_onnx_infer_fn())
    got = np.array(client._embed([crop])[0], dtype=np.float32)

    # InsightFace's own ArcFace recognition on the same crop, normalized
    rec = get_model(_ARCFACE)
    rec.prepare(ctx_id=-1)
    ref = rec.get_feat([crop])[0]
    ref = ref / np.linalg.norm(ref)

    assert got.shape == (512,)
    assert np.allclose(np.linalg.norm(got), 1.0, atol=1e-5)
    assert float(np.dot(got, ref)) > 0.999   # same model + same preprocessing → ~identical
