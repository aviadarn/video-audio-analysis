from celebvision.interfaces import Transcript, TranscriptSegment, Word, FaceDetection

STUB_EMBEDDING: list[float] = [1.0] + [0.0] * 511

_WORDS = ["welcome", "to", "the", "match", "messi", "scored", "a", "goal"]


class StubInferenceClient:
    def transcribe(self, audio_path: str) -> Transcript:
        words = []
        for i, w in enumerate(_WORDS):
            words.append(Word(text=w, start_s=float(i), end_s=float(i) + 0.9))
        seg = TranscriptSegment(start_s=0.0, end_s=float(len(_WORDS)),
                                text=" ".join(_WORDS), words=words)
        return Transcript(segments=[seg])

    def analyze_faces(self, image_path: str) -> list[FaceDetection]:
        return [FaceDetection(bbox=(0.0, 0.0, 10.0, 10.0), embedding=list(STUB_EMBEDDING))]
