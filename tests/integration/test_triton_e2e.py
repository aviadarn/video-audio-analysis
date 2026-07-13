# tests/integration/test_triton_e2e.py
import os
import numpy as np
import pytest


def _triton_ready(url):
    try:
        import tritonclient.http as httpclient
        return httpclient.InferenceServerClient(url=url).is_server_ready()
    except Exception:
        return False


_URL = os.environ.get("TRITON_URL", "localhost:8000")

pytestmark = pytest.mark.skipif(not _triton_ready(_URL),
                                reason="no reachable Triton at TRITON_URL")


def test_embed_against_real_triton():
    from celebvision.inference.triton_client import TritonFaceClient
    from celebvision.config import Settings
    client = TritonFaceClient(Settings.from_env({"TRITON_URL": _URL}))
    crop = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    emb = client._embed([crop])[0]
    assert len(emb) == 512
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
