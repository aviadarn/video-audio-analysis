# docs/M3_RUNBOOK.md

## What M3 adds
ArcFace recognition served via Triton behind `INFERENCE_BACKEND=triton`
(CompositeInferenceClient: faces→Triton, transcribe→ASR_BACKEND). Detection
+ alignment stay local (InsightFace).

## Prepare the model repository
```bash
.venv/bin/python scripts/prepare_triton_models.py
# -> model_repository/arcface/1/model.onnx (+ config.pbtxt)
```

## Feasibility on this machine (Apple Silicon, no GPU)
**CONFIRMED feasible.** `tritonserver` runs under linux/amd64 QEMU emulation on
this Apple Silicon Mac. The ArcFace model (`buffalo_l` ONNX) loads successfully
after the dynamic-batch fix (max_batch_size: 0 removed from `config.pbtxt`).
The real-Triton E2E test (`test_embed_against_real_triton`) ran green against
`localhost:8000` — embedding shape (512,) correct, L2-norm ≈ 1.0. CPU-only
inference is slow (~4 s/image under emulation) but functionally correct.

## Run Triton locally (CPU, best-effort under emulation)
```bash
docker compose --profile triton up triton
# health:
curl localhost:8000/v2/health/ready
# then run the gated tests:
TRITON_URL=localhost:8000 .venv/bin/python -m pytest tests/integration/test_triton_e2e.py -v
```

## GPU host (production)
```bash
docker compose --profile gpu up
# face worker runs with INFERENCE_BACKEND=triton, TRITON_URL=triton:8000
```

## Tests
```bash
.venv/bin/python -m pytest -q                 # fast unit (no model/triton)
.venv/bin/python -m pytest -m slow -v         # ArcFace parity (needs cached buffalo_l)
```

## M3 regression gate (all five must be green)
```bash
# 1. Fast unit (no stack, no model)
.venv/bin/python -m pytest -q

# 2. Ruff
.venv/bin/ruff check src tests scripts

# 3. Slow M3 model tests (ArcFace parity + prep + dynamic-batch guard)
.venv/bin/python -m pytest -m slow -v

# 4. M2 stack regression (Postgres on 55432, infra up)
CELEBVISION_STACK=1 POSTGRES_DSN=postgresql://celeb:celeb@localhost:55432/celeb \
  .venv/bin/python -m pytest -q

# 5. Real-Triton E2E (cv-triton container up)
TRITON_URL=localhost:8000 .venv/bin/python -m pytest tests/integration/test_triton_e2e.py -v
```
