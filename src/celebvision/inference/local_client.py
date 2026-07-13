from celebvision.interfaces import Transcript, TranscriptSegment, Word, FaceDetection


class LocalInferenceClient:
    def __init__(self, whisper_model: str = "small", device: str = "cpu",
                 compute_type: str = "int8", face_model: str = "buffalo_l") -> None:
        self._whisper_model_name = whisper_model
        self._device = device
        self._compute_type = compute_type
        self._face_model_name = face_model
        self._whisper = None   # lazy
        self._face = None      # lazy (Task 6)

    def _load_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(
                self._whisper_model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._whisper

    def _to_transcript(self, segments) -> Transcript:
        out_segs: list[TranscriptSegment] = []
        for seg in segments:
            words = []
            for w in (getattr(seg, "words", None) or []):
                words.append(Word(text=w.word.strip(), start_s=float(w.start),
                                  end_s=float(w.end)))
            out_segs.append(TranscriptSegment(
                start_s=float(seg.start), end_s=float(seg.end),
                text=seg.text.strip(), words=words,
            ))
        return Transcript(segments=out_segs)

    def transcribe(self, audio_path: str) -> Transcript:
        model = self._load_whisper()
        segments, _info = model.transcribe(audio_path, word_timestamps=True)
        return self._to_transcript(list(segments))

    def _load_face(self):
        if self._face is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=self._face_model_name,
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._face = app
        return self._face

    def _to_face_detections(self, faces) -> list[FaceDetection]:
        dets: list[FaceDetection] = []
        for f in faces:
            bbox = tuple(float(x) for x in f.bbox)  # (x1, y1, x2, y2)
            dets.append(FaceDetection(bbox=bbox,
                                      embedding=[float(x) for x in f.normed_embedding]))
        return dets

    def analyze_faces(self, image_path: str) -> list[FaceDetection]:
        import cv2
        app = self._load_face()
        img = cv2.imread(image_path)
        if img is None:
            return []
        return self._to_face_detections(app.get(img))
