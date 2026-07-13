# tests/test_factory_triton.py
import celebvision.factories as fac
from celebvision.config import Settings
from celebvision.inference.composite import CompositeInferenceClient
from celebvision.inference.stub_client import StubInferenceClient


def test_triton_backend_builds_composite(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(fac, "TritonFaceClient", lambda settings: sentinel,
                        raising=False)
    s = Settings.from_env({"INFERENCE_BACKEND": "triton", "ASR_BACKEND": "stub"})
    client = fac.build_inference_client(s)
    assert isinstance(client, CompositeInferenceClient)
    assert isinstance(client._transcriber, StubInferenceClient)
    assert client._face_analyzer is sentinel
