# M3: Triton Serving (faces) — Design

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Builds on:** M2 Async Infrastructure (PR #2). M3 branches off `main` after M2 merges.

## 1. Purpose

Serve the face **recognition** model (ArcFace) through **NVIDIA Triton Inference Server** with
dynamic batching, behind the existing M1 `InferenceClient` protocol, so the celebvision face
worker can run against Triton (`INFERENCE_BACKEND=triton`) with no worker/DAG changes. Detection
+ alignment stay local (InsightFace); only the batchable embedding step moves to Triton.

## 2. Hard constraint & decisions

Dev machine is Apple Silicon (no NVIDIA GPU). Real Triton ships as Linux/amd64 + CUDA; it will,
at best, run its ONNX-Runtime **CPU** backend under qemu emulation, and may not start at all.

| Decision | Choice |
|---|---|
| Local verification | **Attempt a real Triton CPU container**; gate the real-Triton test on a reachable `TRITON_URL`, fall back to parity + mocked-client coverage if it won't start |
| Models on Triton | **Faces only** — ArcFace recognition (native ONNX). ASR stays local/stub |
| Face split | **ArcFace embedding on Triton; SCRFD detect + 5-pt align stay local (InsightFace)** |
| Backend wiring | **`CompositeInferenceClient`** (analyze_faces→Triton, transcribe→`ASR_BACKEND`); M1 `InferenceClient` protocol unchanged |
| ASR in triton profile | `ASR_BACKEND=stub` default |

Out of scope (M4+): ASR-on-Triton (Parakeet/Canary), TensorRT/FP16 conversion, Triton
ensemble/Python-backend full pipeline, retries/DLQ, autoscaling.

## 3. Testing strategy (no local GPU)

- **Task 1 — feasibility spike:** pull `tritonserver`, load the ArcFace model on the ONNX-Runtime
  CPU backend (no `--gpus`), and curl `/v2/health/ready` under emulation. Record the result
  (`docs/M3_RUNBOOK.md`). Everything after is written so the milestone completes whether or not
  the container starts.
- **Embedding parity (no real face needed):** the ArcFace embedding produced by `TritonFaceClient`
  for a given 112×112 crop must equal (within tolerance) InsightFace's own recognition embedding
  for the same crop. Verified via a real `onnxruntime` stand-in, and via real Triton if the
  container runs.
- **Request-mapping unit test:** with `tritonclient` mocked (canned infer response) and detection
  mocked, `analyze_faces` returns the correct `FaceDetection`s (bbox from detection, normalized
  embedding from the Triton response).
- **Real-Triton test:** `@pytest.mark.skipif` unless `TRITON_URL` is reachable — enrol a
  watchlist embedding, analyze a crop, assert the match. Skipped cleanly when Triton isn't up.
- **Regression:** the full M2 stack suite (64) must still pass unchanged — Triton is additive and
  `stub` remains the default backend.

## 4. Architecture

```
face worker (INFERENCE_BACKEND=triton)
      │ analyze_faces(image)
      ▼
 TritonFaceClient
   1. InsightFace SCRFD detect + 5-pt align (LOCAL, CPU)  → 112x112 crops + bboxes
   2. preprocess crops (blobFromImages, RGB, (x-127.5)/127.5, NCHW)
   3. batch → Triton ArcFace (dynamic batching)  ──gRPC/HTTP──▶  Triton Server
   4. L2-normalize embeddings
   5. → [FaceDetection(bbox, embedding), ...]
transcribe(audio)  ─▶ CompositeInferenceClient delegates to ASR_BACKEND (stub|local)
```

### 4.1 Components

- **`TritonFaceClient`** (`celebvision/inference/triton_client.py`) — implements
  `analyze_faces(image_path) -> list[FaceDetection]`. Holds a `tritonclient` handle to
  `TRITON_URL`, the model name (`arcface`), and an InsightFace detector for detect+align. Pure
  helper `_embed(crops: np.ndarray) -> list[list[float]]` (batched Triton infer + normalize) is
  unit-tested against InsightFace's recognition model for parity. `transcribe` raises
  `NotImplementedError` (it is never the transcriber in the composite).
- **`CompositeInferenceClient`** (`celebvision/inference/composite.py`) —
  `__init__(transcriber, face_analyzer)`; `transcribe` → `transcriber.transcribe`; `analyze_faces`
  → `face_analyzer.analyze_faces`.
- **Factory change** (`celebvision/factories.py`): `build_inference_client(settings)` for
  `inference_backend == "triton"` returns `CompositeInferenceClient(transcriber=_build_asr(settings),
  face_analyzer=TritonFaceClient(settings))`, where `_build_asr` dispatches on `settings.asr_backend`
  (`stub` | `local`). Existing `stub`/`local` branches unchanged.
- **Settings** (`celebvision/config.py`): add `triton_url` (`TRITON_URL`, default `localhost:8000`),
  `asr_backend` (`ASR_BACKEND`, default `stub`), `triton_model` (`TRITON_MODEL`, default `arcface`).

### 4.2 ArcFace preprocessing contract (client-side)

Matches InsightFace `ArcFaceONNX`: `cv2.dnn.blobFromImages(crops, 1.0/127.5, (112,112),
(127.5,127.5,127.5), swapRB=True)` → float32 NCHW; Triton output is `[N,512]`; embeddings are
L2-normalized before return (so cosine = dot, matching M1/M2 watchlist search).

## 5. Model repository

Prep script `scripts/prepare_triton_models.py`:
1. Ensure InsightFace `buffalo_l` is downloaded (its `w600k_r50.onnx` recognition model).
2. Copy it to `model_repository/arcface/1/model.onnx`.
3. Write `model_repository/arcface/config.pbtxt`.

`config.pbtxt`:
```
name: "arcface"
platform: "onnxruntime_onnx"
max_batch_size: 16
dynamic_batching { }
input  [ { name: "<arcface_input>"  data_type: TYPE_FP32 dims: [3, 112, 112] } ]
output [ { name: "<arcface_output>" data_type: TYPE_FP32 dims: [512] } ]
instance_group [ { kind: KIND_CPU } ]     # gpu profile overrides to KIND_GPU
```
The input/output tensor names are read from the ONNX model during prep (the script prints them and
writes them into config.pbtxt); the client also reads model metadata from Triton at startup rather
than hard-coding names.

## 6. Deployment

- `docker-compose.yml`: add a `triton` service (`nvcr.io/nvidia/tritonserver:<tag>-py3` or the
  documented CPU-capable tag), `command: tritonserver --model-repository=/models`, mount
  `./model_repository:/models`, expose `8000/8001/8002`, healthcheck on `/v2/health/ready`. Runs
  CPU-only by default.
- `--profile gpu` overlay: GPU device reservation on `triton`; face worker env
  `INFERENCE_BACKEND=triton`, `TRITON_URL=triton:8000`.
- A `triton` compose profile points the **face worker** at Triton while leaving other workers +
  ASR on stub/local. The default (profile-less) M2 stack is unchanged (stub).

## 7. File structure (new/changed in M3)

```
src/celebvision/inference/triton_client.py   # TritonFaceClient
src/celebvision/inference/composite.py        # CompositeInferenceClient
src/celebvision/config.py                     # + triton_url, asr_backend, triton_model (modify)
src/celebvision/factories.py                  # triton branch -> composite (modify)
scripts/prepare_triton_models.py              # export/copy ArcFace onnx + write config.pbtxt
model_repository/arcface/config.pbtxt          # (generated, committed)
docker-compose.yml                             # + triton service, gpu/triton profiles (modify)
docs/M3_RUNBOOK.md                             # feasibility result + GPU-host instructions
pyproject.toml                                 # + triton extra [tritonclient[http]] (modify)
tests/test_triton_client_unit.py               # mocked tritonclient request mapping
tests/integration/test_triton_parity.py        # ArcFace parity (onnxruntime stand-in; real Triton if up)
tests/integration/test_triton_e2e.py           # skipif TRITON_URL unreachable
```
Note: `model_repository/arcface/1/model.onnx` is a large binary produced by the prep script; it is
git-ignored (regenerated locally / on the GPU host), not committed.

## 8. Error handling

`TritonFaceClient` wraps Triton connection/inference failures in `StageError("face", ...)` (the
face worker already maps `StageError` to a failed job). A watchdog on client init verifies the
model is `READY` via Triton metadata, failing fast with a clear message.

## 9. Milestone exit criteria

- Feasibility result recorded in `docs/M3_RUNBOOK.md`.
- Parity test green: `TritonFaceClient` embedding == InsightFace recognition embedding (tolerance)
  on the same crop (onnxruntime stand-in; real Triton if the container ran).
- Mocked request-mapping unit test green.
- Real-Triton test present and either green (if Triton up) or cleanly skipped.
- Full M2 stack suite (64) still green; `ruff check src tests scripts` clean.
- No M1/M2 stage or DAG logic changed — only additive inference-backend code + config/compose.

## 10. Handoff to M4

M4: ASR-on-Triton (Parakeet/Canary), TensorRT/FP16 conversion, per-stage retries + `<stage>.dlq`,
metrics, precision/recall eval harness, and the deferred M2 polish (report duration/completed_at,
streaming storage).
