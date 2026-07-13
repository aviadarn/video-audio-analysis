import cv2
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
        self._in_name = None
        self._out_name = None

    def _ensure_ready(self, client):
        if not client.is_model_ready(self._settings.triton_model):
            raise StageError("face", f"triton model {self._settings.triton_model} not ready")
        if self._in_name is None:
            meta = client.get_model_metadata(self._settings.triton_model)
            self._in_name = meta["inputs"][0]["name"]
            self._out_name = meta["outputs"][0]["name"]

    def _triton_infer(self, blob: np.ndarray) -> np.ndarray:
        import tritonclient.http as httpclient
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self._settings.triton_url)
        self._ensure_ready(self._client)
        inp = httpclient.InferInput(self._in_name, blob.shape, "FP32")
        inp.set_data_from_numpy(blob)
        out = httpclient.InferRequestedOutput(self._out_name)
        resp = self._client.infer(self._settings.triton_model, inputs=[inp], outputs=[out])
        return resp.as_numpy(self._out_name)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        fn = self._infer_fn if self._infer_fn is not None else self._triton_infer
        try:
            return fn(blob)
        except Exception as e:
            raise StageError("face", f"triton infer failed: {e}") from e

    def _preprocess(self, crops):
        return cv2.dnn.blobFromImages(
            crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)

    def _embed(self, crops) -> list[list[float]]:
        blob = self._preprocess(crops)
        feats = np.asarray(self._infer(blob), dtype=np.float32)
        out = []
        for row in feats:
            n = float(np.linalg.norm(row))
            if n <= 0.0:
                raise StageError("face", "zero-norm embedding from Triton")
            out.append((row / n).tolist())
        return out

    def transcribe(self, audio_path: str):
        raise NotImplementedError("TritonFaceClient does not transcribe")

    def _load_detector(self):
        if self._detector is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._detector = app
        return self._detector

    def analyze_faces(self, image_path: str) -> list[FaceDetection]:
        from insightface.utils.face_align import norm_crop
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            faces = self._load_detector().get(img)
            if not faces:
                return []
            crops = [norm_crop(img, landmark=f.kps, image_size=112) for f in faces]
            embeddings = self._embed(crops)
            return [
                FaceDetection(bbox=tuple(float(x) for x in f.bbox), embedding=emb)
                for f, emb in zip(faces, embeddings)
            ]
        except StageError:
            raise
        except Exception as e:
            raise StageError("face", str(e)) from e
