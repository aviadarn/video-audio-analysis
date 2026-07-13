from celebvision.inference.composite import CompositeInferenceClient


class FakeTranscriber:
    def transcribe(self, audio_path):
        return f"transcript:{audio_path}"
    def analyze_faces(self, image_path):
        raise AssertionError("composite must not route faces here")


class FakeFace:
    def analyze_faces(self, image_path):
        return [f"face:{image_path}"]
    def transcribe(self, audio_path):
        raise AssertionError("composite must not route transcribe here")


def test_composite_routes_each_capability():
    c = CompositeInferenceClient(transcriber=FakeTranscriber(), face_analyzer=FakeFace())
    assert c.transcribe("a.wav") == "transcript:a.wav"
    assert c.analyze_faces("k.jpg") == ["face:k.jpg"]
