from celebvision.interfaces import Transcript, FaceDetection


class CompositeInferenceClient:
    def __init__(self, transcriber, face_analyzer) -> None:
        self._transcriber = transcriber
        self._face_analyzer = face_analyzer

    def transcribe(self, audio_path: str) -> Transcript:
        return self._transcriber.transcribe(audio_path)

    def analyze_faces(self, image_path: str) -> list[FaceDetection]:
        return self._face_analyzer.analyze_faces(image_path)
