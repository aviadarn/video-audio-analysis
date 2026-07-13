import numpy as np
from celebvision.config import Settings
from celebvision.interfaces import FaceDetection
from celebvision.errors import StageError


class TritonFaceClient:
    def __init__(self, settings: Settings, infer_fn=None, detector=None) -> None:
        self._settings = settings
        self._infer_fn = infer_fn
        self._detector = detector
        self._client = None

    def _triton_infer(self, blob: np.ndarray) -> np.ndarray:
        import tritonclient.http as httpclient
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self._settings.triton_url)
        inp = httpclient.InferInput("input.1", blob.shape, "FP32")
        inp.set_data_from_numpy(blob)
        out = httpclient.InferRequestedOutput("683")
        resp = self._client.infer(self._settings.triton_model, inputs=[inp],
                                  outputs=[out])
        return resp.as_numpy("683")

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        fn = self._infer_fn if self._infer_fn is not None else self._triton_infer
        try:
            return fn(blob)
        except Exception as e:
            raise StageError("face", f"triton infer failed: {e}") from e

    def _preprocess(self, crops):
        import cv2
        return cv2.dnn.blobFromImages(
            crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)

    def _embed(self, crops) -> list[list[float]]:
        blob = self._preprocess(crops)
        feats = np.asarray(self._infer(blob), dtype=np.float32)
        out = []
        for row in feats:
            n = float(np.linalg.norm(row))
            out.append((row / n).tolist() if n > 0 else row.tolist())
        return out

    def transcribe(self, audio_path: str):
        raise NotImplementedError("TritonFaceClient does not transcribe")

    # analyze_faces added in Task 5
