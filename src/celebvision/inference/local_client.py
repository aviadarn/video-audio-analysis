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

    # analyze_faces added in Task 6
