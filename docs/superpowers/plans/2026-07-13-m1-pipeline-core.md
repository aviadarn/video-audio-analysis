# M1: Celebrity Video Analysis — Pipeline Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI (`celebvision analyze <youtube-url|hls|file>`) that runs the full pipeline in-process on CPU and writes a per-scene JSON report of celebrities detected by spoken mention and on-screen face, plus a `celebvision enroll` CLI to build the watchlist.

**Architecture:** A pipeline library of pure-ish stage functions (ingest → scene → transcribe → face → mention → aggregate) orchestrated sequentially in one process. All heavy/external dependencies sit behind three interfaces — `InferenceClient` (ASR + faces), `LLMClient` (mention extraction), `WatchlistIndex` (face match). M1 ships the `local` implementations (faster-whisper, InsightFace-CPU, numpy index); later milestones swap in Triton/Kafka/pgvector without touching stage logic.

**Tech Stack:** Python 3.11, Pydantic v2, faster-whisper, InsightFace (onnxruntime CPU), PySceneDetect, yt-dlp, ffmpeg (subprocess), Anthropic SDK, numpy, OpenCV (headless), Typer, pytest.

## Global Constraints

- Python **3.11+** (`requires-python = ">=3.11"`).
- All external deps accessed **only** through `InferenceClient`, `LLMClient`, `WatchlistIndex` protocols — stage functions never import faster-whisper/insightface/anthropic directly.
- `src/` layout; package name **`celebvision`**; import root `celebvision`.
- Every stage function is **pure w.r.t. its inputs + injected clients** (no global state), so it is unit-testable with fakes.
- Timestamps are **float seconds**. Audio is **16 kHz mono WAV**.
- LLM model id: **`claude-haiku-4-5-20251001`**.
- Tests that require real models/network are marked `@pytest.mark.slow` and excluded from the default `pytest` run.
- Lint/format: **ruff**. Type hints required on all public functions.
- Commit after every task (Conventional Commits).
- **Interpreter:** a project venv created with `python3.13` lives at `.venv/`. In every step, `python` means `.venv/bin/python` and `pip` means `.venv/bin/pip` (shell state does not persist between commands, so use the explicit paths — do not rely on `source .venv/bin/activate`).
- **Dependency split:** base `dependencies` are light (pydantic, typer, numpy) so `pip install -e '.[dev]'` is fast and needs no C compilation. Heavy ML runtime libs (faster-whisper, insightface, onnxruntime, opencv, scenedetect, yt-dlp, anthropic) live in the optional `local` extra — they are imported lazily and are NOT needed for any unit test. Do not add them to base `dependencies`.

---

### Task 1: Project scaffold + tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/celebvision/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_version.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `celebvision` with `__version__: str`; `pytest` runnable; `ruff` configured; marker `slow` registered.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_version.py
import celebvision

def test_has_version():
    assert isinstance(celebvision.__version__, str)
    assert celebvision.__version__.count(".") >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_version.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision'`

- [ ] **Step 3: Create pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "celebvision"
version = "0.1.0"
description = "Celebrity video analysis pipeline (spoken mentions + on-screen faces)"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "typer>=0.12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]
# Heavy ML runtime backends — imported lazily, only needed for real (non-test) runs.
local = [
    "faster-whisper>=1.0",
    "insightface>=0.7.3",
    "onnxruntime>=1.17",
    "opencv-python-headless>=4.9",
    "scenedetect>=0.6.3",
    "yt-dlp>=2024.4.9",
    "anthropic>=0.34",
]

[project.scripts]
celebvision = "celebvision.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/celebvision"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["slow: requires real models or network (deselected by default)"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 4: Create package init and .gitignore**

```python
# src/celebvision/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

```gitignore
# .gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
.superpowers/
*.egg-info/
build/
dist/
/data/
/models/
*.wav
*.mp4
*.jpg
```

- [ ] **Step 5: Create venv, install dev deps, run test to verify it passes**

Run: `python3.13 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e '.[dev]'` then `.venv/bin/python -m pytest tests/test_version.py -v`
Expected: PASS (1 passed). (`.venv/` is git-ignored via the `.venv/` entry.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/celebvision/__init__.py tests/__init__.py tests/test_version.py .gitignore
git commit -m "chore: scaffold celebvision package and tooling"
```

---

### Task 2: Domain models

**Files:**
- Create: `src/celebvision/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces (Pydantic v2 models, all importable from `celebvision.models`):
  - `SourceKind` = `Literal["youtube", "hls", "file"]`
  - `JobSource(kind: SourceKind, locator: str)`
  - `SpokenMention(name: str, canonical_id: str, confidence: float, evidence_span: str)`
  - `KeywordHit(keyword: str, count: int, spans: list[str])`
  - `OnscreenFace(name: str, canonical_id: str, confidence: float, bbox: tuple[float,float,float,float], keyframe: str)`
  - `SceneReport(scene_id: int, start_s: float, end_s: float, keyframes: list[str], transcript: str, spoken_mentions: list[SpokenMention], keyword_hits: list[KeywordHit], onscreen_faces: list[OnscreenFace])`
  - `CelebrityIndexEntry(name: str, canonical_id: str, scenes: list[int], modalities: list[str])`
  - `Report(job_id: str, source: JobSource, duration_s: float, watchlist_id: str, completed_at: str, scenes: list[SceneReport], celebrity_index: list[CelebrityIndexEntry])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from celebvision.models import (
    JobSource, SpokenMention, OnscreenFace, SceneReport, Report,
)

def test_report_round_trip():
    scene = SceneReport(
        scene_id=0, start_s=0.0, end_s=5.0, keyframes=["k0.jpg"],
        transcript="messi scored", spoken_mentions=[
            SpokenMention(name="Lionel Messi", canonical_id="messi",
                          confidence=0.9, evidence_span="messi scored")],
        keyword_hits=[], onscreen_faces=[
            OnscreenFace(name="Lionel Messi", canonical_id="messi",
                         confidence=0.7, bbox=(1, 2, 3, 4), keyframe="k0.jpg")],
    )
    report = Report(
        job_id="j1", source=JobSource(kind="file", locator="a.mp4"),
        duration_s=5.0, watchlist_id="w1", completed_at="2026-07-13T00:00:00Z",
        scenes=[scene], celebrity_index=[],
    )
    dumped = report.model_dump_json()
    again = Report.model_validate_json(dumped)
    assert again.scenes[0].spoken_mentions[0].canonical_id == "messi"
    assert again.scenes[0].onscreen_faces[0].bbox == (1, 2, 3, 4)

def test_source_kind_rejects_bad_value():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JobSource(kind="ftp", locator="x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.models'`

- [ ] **Step 3: Write the models**

```python
# src/celebvision/models.py
from typing import Literal
from pydantic import BaseModel

SourceKind = Literal["youtube", "hls", "file"]


class JobSource(BaseModel):
    kind: SourceKind
    locator: str


class SpokenMention(BaseModel):
    name: str
    canonical_id: str
    confidence: float
    evidence_span: str


class KeywordHit(BaseModel):
    keyword: str
    count: int
    spans: list[str]


class OnscreenFace(BaseModel):
    name: str
    canonical_id: str
    confidence: float
    bbox: tuple[float, float, float, float]
    keyframe: str


class SceneReport(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    keyframes: list[str]
    transcript: str
    spoken_mentions: list[SpokenMention]
    keyword_hits: list[KeywordHit]
    onscreen_faces: list[OnscreenFace]


class CelebrityIndexEntry(BaseModel):
    name: str
    canonical_id: str
    scenes: list[int]
    modalities: list[str]


class Report(BaseModel):
    job_id: str
    source: JobSource
    duration_s: float
    watchlist_id: str
    completed_at: str
    scenes: list[SceneReport]
    celebrity_index: list[CelebrityIndexEntry]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/models.py tests/test_models.py
git commit -m "feat: add domain models for pipeline report"
```

---

### Task 3: Client interfaces + inference data types

**Files:**
- Create: `src/celebvision/interfaces.py`
- Create: `tests/test_interfaces.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all importable from `celebvision.interfaces`):
  - `Word(text: str, start_s: float, end_s: float)` — Pydantic model
  - `TranscriptSegment(start_s: float, end_s: float, text: str, words: list[Word])`
  - `Transcript(segments: list[TranscriptSegment])` with property `duration_s: float` (end of last segment, 0.0 if empty) and method `words_in(start_s, end_s) -> list[Word]` (words whose midpoint falls in `[start_s, end_s)`).
  - `FaceDetection(bbox: tuple[float,float,float,float], embedding: list[float])`
  - `MentionExtraction(mentions: list[SpokenMention], keyword_hits: list[KeywordHit])`
  - `class InferenceClient(Protocol)`: `transcribe(self, audio_path: str) -> Transcript`; `analyze_faces(self, image_path: str) -> list[FaceDetection]`
  - `class LLMClient(Protocol)`: `extract_mentions(self, transcript_text: str, watchlist: list[tuple[str, str, list[str]]], keywords: list[str]) -> MentionExtraction` — watchlist tuple is `(canonical_id, name, aliases)`.
  - `class WatchlistIndex(Protocol)`: attribute `watchlist_id: str`; `search(self, embedding: list[float], threshold: float) -> tuple[str, str, float] | None` returns `(canonical_id, name, score)` of best match above threshold else `None`; `names(self) -> list[tuple[str, str, list[str]]]` returns `(canonical_id, name, aliases)` for every enrolled identity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interfaces.py
from celebvision.interfaces import Word, TranscriptSegment, Transcript

def test_duration_of_empty_transcript_is_zero():
    assert Transcript(segments=[]).duration_s == 0.0

def test_words_in_uses_midpoint():
    words = [Word(text="a", start_s=0.0, end_s=2.0),   # mid 1.0 -> in [0,5)
             Word(text="b", start_s=4.0, end_s=8.0)]   # mid 6.0 -> not in [0,5)
    seg = TranscriptSegment(start_s=0.0, end_s=8.0, text="a b", words=words)
    t = Transcript(segments=[seg])
    assert t.duration_s == 8.0
    got = [w.text for w in t.words_in(0.0, 5.0)]
    assert got == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_interfaces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.interfaces'`

- [ ] **Step 3: Write the interfaces**

```python
# src/celebvision/interfaces.py
from typing import Protocol
from pydantic import BaseModel
from celebvision.models import SpokenMention, KeywordHit


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    def words_in(self, start_s: float, end_s: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            for w in seg.words:
                mid = (w.start_s + w.end_s) / 2.0
                if start_s <= mid < end_s:
                    out.append(w)
        return out


class FaceDetection(BaseModel):
    bbox: tuple[float, float, float, float]
    embedding: list[float]


class MentionExtraction(BaseModel):
    mentions: list[SpokenMention]
    keyword_hits: list[KeywordHit]


class InferenceClient(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...
    def analyze_faces(self, image_path: str) -> list[FaceDetection]: ...


class LLMClient(Protocol):
    def extract_mentions(
        self,
        transcript_text: str,
        watchlist: list[tuple[str, str, list[str]]],
        keywords: list[str],
    ) -> MentionExtraction: ...


class WatchlistIndex(Protocol):
    watchlist_id: str

    def search(
        self, embedding: list[float], threshold: float
    ) -> tuple[str, str, float] | None: ...
    def names(self) -> list[tuple[str, str, list[str]]]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_interfaces.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/interfaces.py tests/test_interfaces.py
git commit -m "feat: add client protocols and inference data types"
```

---

### Task 4: Local watchlist index (numpy, persisted)

**Files:**
- Create: `src/celebvision/watchlist/__init__.py`
- Create: `src/celebvision/watchlist/local_index.py`
- Create: `tests/test_local_index.py`

**Interfaces:**
- Consumes: `WatchlistIndex` protocol from Task 3; `FaceDetection`.
- Produces: `LocalWatchlistIndex` (implements `WatchlistIndex`), importable from `celebvision.watchlist.local_index`:
  - `LocalWatchlistIndex(watchlist_id: str)` — in-memory, empty.
  - `add(self, canonical_id: str, name: str, aliases: list[str], embeddings: list[list[float]]) -> None`
  - `search(...)` cosine (embeddings assumed L2-normalized; dot product), returns best above threshold.
  - `names(...)`.
  - `save(self, path: str) -> None` and classmethod `load(cls, path: str) -> "LocalWatchlistIndex"` — persists to a single `.npz` (embeddings matrix + JSON metadata as a string array).
  - Attribute `watchlist_id: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_index.py
import math
from celebvision.watchlist.local_index import LocalWatchlistIndex


def _unit(vec):
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec]


def test_search_returns_best_match_above_threshold(tmp_path):
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi"], [_unit([1.0, 0.0, 0.0])])
    idx.add("ronaldo", "Cristiano Ronaldo", ["cr7"], [_unit([0.0, 1.0, 0.0])])

    hit = idx.search(_unit([0.9, 0.1, 0.0]), threshold=0.5)
    assert hit is not None
    canonical_id, name, score = hit
    assert canonical_id == "messi"
    assert score > 0.5

def test_search_returns_none_below_threshold():
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi"], [_unit([1.0, 0.0, 0.0])])
    assert idx.search(_unit([0.0, 0.0, 1.0]), threshold=0.5) is None

def test_save_load_round_trip(tmp_path):
    idx = LocalWatchlistIndex("w1")
    idx.add("messi", "Lionel Messi", ["messi", "leo"], [_unit([1.0, 0.0, 0.0])])
    path = str(tmp_path / "wl.npz")
    idx.save(path)
    loaded = LocalWatchlistIndex.load(path)
    assert loaded.watchlist_id == "w1"
    assert loaded.names() == [("messi", "Lionel Messi", ["messi", "leo"])]
    assert loaded.search(_unit([1.0, 0.0, 0.0]), threshold=0.5)[0] == "messi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_local_index.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the index**

```python
# src/celebvision/watchlist/__init__.py
```

```python
# src/celebvision/watchlist/local_index.py
import json
import numpy as np


class LocalWatchlistIndex:
    def __init__(self, watchlist_id: str) -> None:
        self.watchlist_id = watchlist_id
        self._vectors: list[list[float]] = []   # one row per embedding
        self._row_id: list[int] = []            # identity index per row
        self._identities: list[dict] = []       # {canonical_id, name, aliases}

    def add(self, canonical_id: str, name: str, aliases: list[str],
            embeddings: list[list[float]]) -> None:
        ident_idx = len(self._identities)
        self._identities.append(
            {"canonical_id": canonical_id, "name": name, "aliases": list(aliases)}
        )
        for emb in embeddings:
            self._vectors.append(list(emb))
            self._row_id.append(ident_idx)

    def search(self, embedding, threshold: float):
        if not self._vectors:
            return None
        mat = np.asarray(self._vectors, dtype=np.float32)   # (N, D), L2-normed rows
        q = np.asarray(embedding, dtype=np.float32)
        scores = mat @ q                                    # cosine (rows unit-norm)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < threshold:
            return None
        ident = self._identities[self._row_id[best]]
        return (ident["canonical_id"], ident["name"], score)

    def names(self) -> list[tuple[str, str, list[str]]]:
        return [(i["canonical_id"], i["name"], list(i["aliases"]))
                for i in self._identities]

    def save(self, path: str) -> None:
        meta = json.dumps({
            "watchlist_id": self.watchlist_id,
            "row_id": self._row_id,
            "identities": self._identities,
        })
        vectors = np.asarray(self._vectors, dtype=np.float32) if self._vectors \
            else np.zeros((0, 0), dtype=np.float32)
        np.savez(path, vectors=vectors, meta=np.array(meta))

    @classmethod
    def load(cls, path: str) -> "LocalWatchlistIndex":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        idx = cls(meta["watchlist_id"])
        idx._identities = meta["identities"]
        idx._row_id = meta["row_id"]
        idx._vectors = data["vectors"].tolist()
        return idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_local_index.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/watchlist/ tests/test_local_index.py
git commit -m "feat: add numpy-backed local watchlist index"
```

---

### Task 5: Local inference client — transcription (faster-whisper)

**Files:**
- Create: `src/celebvision/inference/__init__.py`
- Create: `src/celebvision/inference/local_client.py`
- Create: `tests/test_local_client_transcribe.py`

**Interfaces:**
- Consumes: `InferenceClient`, `Transcript`, `TranscriptSegment`, `Word` from Task 3.
- Produces: `LocalInferenceClient` (partial — `transcribe` only in this task), importable from `celebvision.inference.local_client`:
  - `LocalInferenceClient(whisper_model: str = "small", device: str = "cpu", compute_type: str = "int8", face_model: str = "buffalo_l")` — lazy-loads models on first use.
  - `transcribe(self, audio_path: str) -> Transcript`.
  - Internal `_to_transcript(segments) -> Transcript` (pure, converts faster-whisper segment objects to our `Transcript`) — unit-tested directly with fakes, so no real model is needed in CI.

- [ ] **Step 1: Write the failing test** (tests the pure conversion, no model download)

```python
# tests/test_local_client_transcribe.py
from types import SimpleNamespace
from celebvision.inference.local_client import LocalInferenceClient


def test_to_transcript_maps_words_and_segments():
    # Mimic faster-whisper Segment/Word duck types.
    words = [SimpleNamespace(word=" Messi", start=0.1, end=0.5),
             SimpleNamespace(word=" scored", start=0.5, end=0.9)]
    seg = SimpleNamespace(start=0.0, end=1.0, text=" Messi scored", words=words)

    client = LocalInferenceClient.__new__(LocalInferenceClient)  # skip __init__/model load
    transcript = client._to_transcript([seg])

    assert len(transcript.segments) == 1
    s = transcript.segments[0]
    assert s.text == "Messi scored"
    assert [w.text for w in s.words] == ["Messi", "scored"]
    assert s.words[0].start_s == 0.1
    assert transcript.duration_s == 1.0

def test_to_transcript_handles_segment_without_words():
    seg = SimpleNamespace(start=0.0, end=2.0, text="hi", words=None)
    client = LocalInferenceClient.__new__(LocalInferenceClient)
    transcript = client._to_transcript([seg])
    assert transcript.segments[0].words == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_local_client_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the client (transcription half)**

```python
# src/celebvision/inference/__init__.py
```

```python
# src/celebvision/inference/local_client.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_local_client_transcribe.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: (Optional) real smoke test, marked slow**

```python
# append to tests/test_local_client_transcribe.py
import pytest

@pytest.mark.slow
def test_transcribe_real_audio(tmp_path):
    # Generate 1s of silence via ffmpeg; assert it runs without error.
    import subprocess
    wav = str(tmp_path / "s.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=16000:cl=mono", "-t", "1", wav], check=True)
    t = LocalInferenceClient(whisper_model="tiny").transcribe(wav)
    assert isinstance(t.duration_s, float)
```

- [ ] **Step 6: Commit**

```bash
git add src/celebvision/inference/ tests/test_local_client_transcribe.py
git commit -m "feat: add local transcription via faster-whisper"
```

---

### Task 6: Local inference client — faces (InsightFace)

**Files:**
- Modify: `src/celebvision/inference/local_client.py` (add `analyze_faces` + `_load_face`)
- Create: `tests/test_local_client_faces.py`

**Interfaces:**
- Consumes: `FaceDetection` (Task 3), `LocalInferenceClient` (Task 5).
- Produces: `LocalInferenceClient.analyze_faces(self, image_path: str) -> list[FaceDetection]`, and pure helper `_to_face_detections(faces) -> list[FaceDetection]` converting InsightFace face objects (`.bbox` ndarray, `.normed_embedding` ndarray) to `FaceDetection`.

- [ ] **Step 1: Write the failing test** (pure conversion, no model)

```python
# tests/test_local_client_faces.py
from types import SimpleNamespace
import numpy as np
from celebvision.inference.local_client import LocalInferenceClient


def test_to_face_detections_maps_bbox_and_embedding():
    face = SimpleNamespace(
        bbox=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        normed_embedding=np.array([0.6, 0.8], dtype=np.float32),
    )
    client = LocalInferenceClient.__new__(LocalInferenceClient)
    dets = client._to_face_detections([face])
    assert len(dets) == 1
    assert dets[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert dets[0].embedding == [0.6, 0.8]

def test_to_face_detections_empty():
    client = LocalInferenceClient.__new__(LocalInferenceClient)
    assert client._to_face_detections([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_local_client_faces.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_to_face_detections'`

- [ ] **Step 3: Add face methods to the client**

```python
# append inside class LocalInferenceClient in src/celebvision/inference/local_client.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_local_client_faces.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/inference/local_client.py tests/test_local_client_faces.py
git commit -m "feat: add local face detection/embedding via InsightFace"
```

---

### Task 7: Media I/O — ingest (download + audio extract + keyframe)

**Files:**
- Create: `src/celebvision/media.py`
- Create: `tests/test_media.py`

**Interfaces:**
- Consumes: `JobSource` (Task 2).
- Produces (importable from `celebvision.media`):
  - `classify_source(locator: str) -> JobSource` — `youtube` if host is youtube.com/youtu.be; `hls` if ends `.m3u8`; else `file`.
  - `ingest(source: JobSource, workdir: str, downloader=..., runner=subprocess.run) -> tuple[str, str]` returns `(video_path, audio_path)`. `downloader(url, out_template) -> str` (returns downloaded video path) is injectable; defaults to a yt-dlp-backed function. Audio is extracted to 16 kHz mono WAV via ffmpeg using `runner`.
  - `extract_keyframe(video_path: str, t_s: float, out_path: str, runner=subprocess.run) -> str` — grabs one frame at `t_s` seconds via ffmpeg; returns `out_path`.

- [ ] **Step 1: Write the failing test** (inject fakes for downloader + runner; no network/ffmpeg)

```python
# tests/test_media.py
import os
from celebvision.media import classify_source, ingest, extract_keyframe
from celebvision.models import JobSource


def test_classify_source():
    assert classify_source("https://youtu.be/abc").kind == "youtube"
    assert classify_source("https://www.youtube.com/watch?v=abc").kind == "youtube"
    assert classify_source("http://x/playlist.m3u8").kind == "hls"
    assert classify_source("/tmp/a.mp4").kind == "file"

def test_ingest_downloads_youtube_then_extracts_audio(tmp_path):
    calls = []
    def fake_downloader(url, out_template):
        p = str(tmp_path / "video.mp4")
        open(p, "w").close()
        return p
    def fake_runner(cmd, check):
        calls.append(cmd)
        # simulate ffmpeg producing the output file (last arg)
        open(cmd[-1], "w").close()
        class R: returncode = 0
        return R()

    src = JobSource(kind="youtube", locator="https://youtu.be/abc")
    video, audio = ingest(src, str(tmp_path), downloader=fake_downloader,
                          runner=fake_runner)
    assert os.path.exists(video)
    assert audio.endswith(".wav")
    assert os.path.exists(audio)
    # ffmpeg called with 16k mono flags
    assert any("-ar" in c and "16000" in c for c in calls)

def test_ingest_file_source_skips_download(tmp_path):
    vid = str(tmp_path / "in.mp4")
    open(vid, "w").close()
    def boom(*a, **k): raise AssertionError("should not download")
    def fake_runner(cmd, check):
        open(cmd[-1], "w").close()
        class R: returncode = 0
        return R()
    src = JobSource(kind="file", locator=vid)
    video, audio = ingest(src, str(tmp_path), downloader=boom, runner=fake_runner)
    assert video == vid

def test_extract_keyframe_builds_ffmpeg_command(tmp_path):
    seen = {}
    def fake_runner(cmd, check):
        seen["cmd"] = cmd
        open(cmd[-1], "w").close()
        class R: returncode = 0
        return R()
    out = extract_keyframe("v.mp4", 12.5, str(tmp_path / "k.jpg"),
                           runner=fake_runner)
    assert out.endswith("k.jpg")
    assert "12.5" in seen["cmd"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_media.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.media'`

- [ ] **Step 3: Write the media module**

```python
# src/celebvision/media.py
import os
import subprocess
from urllib.parse import urlparse
from celebvision.models import JobSource


def classify_source(locator: str) -> JobSource:
    parsed = urlparse(locator)
    host = (parsed.netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return JobSource(kind="youtube", locator=locator)
    if locator.lower().endswith(".m3u8"):
        return JobSource(kind="hls", locator=locator)
    return JobSource(kind="file", locator=locator)


def _default_downloader(url: str, out_template: str) -> str:
    from yt_dlp import YoutubeDL
    opts = {"format": "mp4/best", "outtmpl": out_template, "quiet": True,
            "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def ingest(source: JobSource, workdir: str, downloader=_default_downloader,
           runner=subprocess.run) -> tuple[str, str]:
    os.makedirs(workdir, exist_ok=True)
    if source.kind == "file":
        video_path = source.locator
    else:  # youtube or hls -> download/remux to local mp4
        video_path = downloader(source.locator, os.path.join(workdir, "video.%(ext)s"))
    audio_path = os.path.join(workdir, "audio.wav")
    runner(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
            "-ar", "16000", audio_path], check=True)
    return video_path, audio_path


def extract_keyframe(video_path: str, t_s: float, out_path: str,
                     runner=subprocess.run) -> str:
    runner(["ffmpeg", "-y", "-ss", str(t_s), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path], check=True)
    return out_path
```

Note: `_default_downloader` also works for HLS — yt-dlp reads `.m3u8` directly and remuxes to mp4.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_media.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/media.py tests/test_media.py
git commit -m "feat: add media ingest, audio extraction, keyframe capture"
```

---

### Task 8: Scene detection stage

**Files:**
- Create: `src/celebvision/stages/__init__.py`
- Create: `src/celebvision/stages/scenes.py`
- Create: `tests/test_scenes.py`

**Interfaces:**
- Consumes: `extract_keyframe` (Task 7).
- Produces (importable from `celebvision.stages.scenes`):
  - `SceneWindow(scene_id: int, start_s: float, end_s: float, keyframes: list[str])` — Pydantic model.
  - `detect_scenes(video_path: str, detector_fn=...) -> list[tuple[float, float]]` — wraps PySceneDetect `AdaptiveDetector`; `detector_fn(video_path) -> list[(start_s, end_s)]` injectable for tests.
  - `build_scene_windows(video_path, boundaries, workdir, keyframe_fn=extract_keyframe) -> list[SceneWindow]` — extracts one keyframe at each scene midpoint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenes.py
from celebvision.stages.scenes import build_scene_windows, SceneWindow


def test_build_scene_windows_extracts_midpoint_keyframe(tmp_path):
    grabbed = []
    def fake_keyframe(video_path, t_s, out_path, runner=None):
        grabbed.append((t_s, out_path))
        return out_path
    boundaries = [(0.0, 4.0), (4.0, 10.0)]
    windows = build_scene_windows("v.mp4", boundaries, str(tmp_path),
                                  keyframe_fn=fake_keyframe)
    assert [w.scene_id for w in windows] == [0, 1]
    assert isinstance(windows[0], SceneWindow)
    # midpoints 2.0 and 7.0
    assert grabbed[0][0] == 2.0
    assert grabbed[1][0] == 7.0
    assert windows[0].keyframes == [grabbed[0][1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the scenes stage**

```python
# src/celebvision/stages/__init__.py
```

```python
# src/celebvision/stages/scenes.py
import os
from pydantic import BaseModel
from celebvision.media import extract_keyframe


class SceneWindow(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    keyframes: list[str]


def _default_detector(video_path: str) -> list[tuple[float, float]]:
    from scenedetect import detect, AdaptiveDetector
    scene_list = detect(video_path, AdaptiveDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def detect_scenes(video_path: str, detector_fn=_default_detector) -> list[tuple[float, float]]:
    boundaries = detector_fn(video_path)
    if not boundaries:  # single-shot video -> one scene covering whole file
        return [(0.0, 0.0)]
    return boundaries


def build_scene_windows(video_path: str, boundaries: list[tuple[float, float]],
                        workdir: str, keyframe_fn=extract_keyframe) -> list[SceneWindow]:
    os.makedirs(workdir, exist_ok=True)
    windows: list[SceneWindow] = []
    for i, (start_s, end_s) in enumerate(boundaries):
        mid = (start_s + end_s) / 2.0
        out = os.path.join(workdir, f"scene_{i:04d}.jpg")
        keyframe_fn(video_path, mid, out)
        windows.append(SceneWindow(scene_id=i, start_s=start_s, end_s=end_s,
                                   keyframes=[out]))
    return windows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenes.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/stages/ tests/test_scenes.py
git commit -m "feat: add scene detection stage with keyframe extraction"
```

---

### Task 9: Face-matching stage

**Files:**
- Create: `src/celebvision/stages/faces.py`
- Create: `tests/test_faces_stage.py`

**Interfaces:**
- Consumes: `SceneWindow` (Task 8), `InferenceClient` + `FaceDetection` (Task 3), `WatchlistIndex` (Task 3), `OnscreenFace` (Task 2).
- Produces: `match_faces_in_scene(scene: SceneWindow, inference: InferenceClient, watchlist: WatchlistIndex, threshold: float = 0.35) -> list[OnscreenFace]` — for each keyframe, detect faces, match each embedding against the watchlist; emit an `OnscreenFace` per match above threshold. Deduplicate by `canonical_id` keeping the highest confidence.

- [ ] **Step 1: Write the failing test** (fakes for inference + watchlist)

```python
# tests/test_faces_stage.py
from celebvision.stages.faces import match_faces_in_scene
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import FaceDetection


class FakeInference:
    def __init__(self, dets): self._dets = dets
    def analyze_faces(self, image_path): return self._dets
    def transcribe(self, audio_path): raise NotImplementedError


class FakeWatchlist:
    def __init__(self, table): self._table = table  # embedding-tuple -> (id,name,score)
    def search(self, embedding, threshold):
        return self._table.get(tuple(embedding))
    def names(self): return []


def test_matches_and_dedupes_by_identity():
    dets = [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0]),
            FaceDetection(bbox=(2, 2, 3, 3), embedding=[0.9, 0.1])]
    wl = FakeWatchlist({
        (1.0, 0.0): ("messi", "Lionel Messi", 0.8),
        (0.9, 0.1): ("messi", "Lionel Messi", 0.6),  # same identity, lower score
    })
    scene = SceneWindow(scene_id=0, start_s=0, end_s=5, keyframes=["k.jpg"])
    faces = match_faces_in_scene(scene, FakeInference(dets), wl, threshold=0.35)
    assert len(faces) == 1
    assert faces[0].canonical_id == "messi"
    assert faces[0].confidence == 0.8  # highest kept

def test_no_match_returns_empty():
    dets = [FaceDetection(bbox=(0, 0, 1, 1), embedding=[0.0, 1.0])]
    scene = SceneWindow(scene_id=0, start_s=0, end_s=5, keyframes=["k.jpg"])
    faces = match_faces_in_scene(scene, FakeInference(dets), FakeWatchlist({}),
                                 threshold=0.35)
    assert faces == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_faces_stage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the face stage**

```python
# src/celebvision/stages/faces.py
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import InferenceClient, WatchlistIndex
from celebvision.models import OnscreenFace


def match_faces_in_scene(scene: SceneWindow, inference: InferenceClient,
                         watchlist: WatchlistIndex, threshold: float = 0.35
                         ) -> list[OnscreenFace]:
    best: dict[str, OnscreenFace] = {}
    for keyframe in scene.keyframes:
        for det in inference.analyze_faces(keyframe):
            hit = watchlist.search(det.embedding, threshold)
            if hit is None:
                continue
            canonical_id, name, score = hit
            face = OnscreenFace(name=name, canonical_id=canonical_id,
                                confidence=score, bbox=det.bbox, keyframe=keyframe)
            if canonical_id not in best or score > best[canonical_id].confidence:
                best[canonical_id] = face
    return list(best.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_faces_stage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/stages/faces.py tests/test_faces_stage.py
git commit -m "feat: add face-matching stage against watchlist"
```

---

### Task 10: Mention + keyword stage

**Files:**
- Create: `src/celebvision/stages/mentions.py`
- Create: `src/celebvision/llm/__init__.py`
- Create: `src/celebvision/llm/anthropic_client.py`
- Create: `tests/test_mentions_stage.py`

**Interfaces:**
- Consumes: `Transcript` (Task 3), `SceneWindow` (Task 8), `LLMClient` + `MentionExtraction` (Task 3), `WatchlistIndex` (Task 3), `SpokenMention`/`KeywordHit` (Task 2).
- Produces:
  - `scene_transcript_text(transcript: Transcript, scene: SceneWindow) -> str` (in `stages/mentions.py`) — joins words whose midpoint falls in the scene window; if `scene.end_s == 0.0` (single-shot sentinel) use the whole transcript text.
  - `extract_scene_mentions(text: str, watchlist: WatchlistIndex, keywords: list[str], llm: LLMClient) -> tuple[list[SpokenMention], list[KeywordHit]]` — calls `llm.extract_mentions`, returns its lists.
  - `AnthropicLLMClient(model: str = "claude-haiku-4-5-20251001", api_key: str | None = None)` in `celebvision.llm.anthropic_client` implementing `LLMClient`, with a pure `_parse_response(payload: dict, keywords: list[str]) -> MentionExtraction` (unit-tested; no network).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mentions_stage.py
from celebvision.stages.mentions import scene_transcript_text, extract_scene_mentions
from celebvision.stages.scenes import SceneWindow
from celebvision.interfaces import Transcript, TranscriptSegment, Word, MentionExtraction
from celebvision.models import SpokenMention, KeywordHit
from celebvision.llm.anthropic_client import AnthropicLLMClient


def _transcript():
    words = [Word(text="Messi", start_s=1.0, end_s=1.5),
             Word(text="scored", start_s=1.5, end_s=2.0),
             Word(text="Ronaldo", start_s=6.0, end_s=6.5)]
    return Transcript(segments=[TranscriptSegment(start_s=0, end_s=7,
                      text="Messi scored Ronaldo", words=words)])


def test_scene_transcript_text_windows_words():
    scene = SceneWindow(scene_id=0, start_s=0.0, end_s=5.0, keyframes=[])
    assert scene_transcript_text(_transcript(), scene) == "Messi scored"

def test_scene_transcript_text_single_shot_uses_all():
    scene = SceneWindow(scene_id=0, start_s=0.0, end_s=0.0, keyframes=[])
    assert scene_transcript_text(_transcript(), scene) == "Messi scored Ronaldo"

class FakeLLM:
    def extract_mentions(self, transcript_text, watchlist, keywords):
        return MentionExtraction(
            mentions=[SpokenMention(name="Lionel Messi", canonical_id="messi",
                                    confidence=0.95, evidence_span="Messi scored")],
            keyword_hits=[KeywordHit(keyword="scored", count=1, spans=["Messi scored"])],
        )

class FakeWatchlist:
    def names(self): return [("messi", "Lionel Messi", ["messi"])]
    def search(self, e, t): return None

def test_extract_scene_mentions_delegates_to_llm():
    mentions, hits = extract_scene_mentions("Messi scored", FakeWatchlist(),
                                            ["scored"], FakeLLM())
    assert mentions[0].canonical_id == "messi"
    assert hits[0].keyword == "scored"

def test_parse_response_maps_json():
    payload = {"mentions": [{"name": "Lionel Messi", "canonical_id": "messi",
                             "confidence": 0.9, "evidence_span": "Messi scored"}],
               "keyword_hits": [{"keyword": "scored", "count": 2,
                                 "spans": ["a", "b"]}]}
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    result = client._parse_response(payload, ["scored"])
    assert result.mentions[0].canonical_id == "messi"
    assert result.keyword_hits[0].count == 2

def test_parse_response_tolerates_missing_keys():
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    result = client._parse_response({}, [])
    assert result.mentions == []
    assert result.keyword_hits == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mentions_stage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the mention stage**

```python
# src/celebvision/stages/mentions.py
from celebvision.interfaces import Transcript, LLMClient, WatchlistIndex
from celebvision.stages.scenes import SceneWindow
from celebvision.models import SpokenMention, KeywordHit


def scene_transcript_text(transcript: Transcript, scene: SceneWindow) -> str:
    if scene.end_s == 0.0:  # single-shot sentinel: whole transcript
        return " ".join(w.text for seg in transcript.segments for w in seg.words) \
            or " ".join(seg.text for seg in transcript.segments)
    words = transcript.words_in(scene.start_s, scene.end_s)
    return " ".join(w.text for w in words)


def extract_scene_mentions(text: str, watchlist: WatchlistIndex,
                           keywords: list[str], llm: LLMClient
                           ) -> tuple[list[SpokenMention], list[KeywordHit]]:
    if not text.strip():
        return [], []
    result = llm.extract_mentions(text, watchlist.names(), keywords)
    return result.mentions, result.keyword_hits
```

- [ ] **Step 4: Write the Anthropic LLM client**

```python
# src/celebvision/llm/__init__.py
```

```python
# src/celebvision/llm/anthropic_client.py
import json
from celebvision.interfaces import MentionExtraction
from celebvision.models import SpokenMention, KeywordHit

_PROMPT = """You are extracting celebrity mentions and keyword hits from a transcript.

Only report a celebrity if it matches one of the watchlist identities (by name or alias).
Return STRICT JSON, no prose, matching this schema:
{{"mentions": [{{"name": str, "canonical_id": str, "confidence": float 0..1,
  "evidence_span": str}}],
 "keyword_hits": [{{"keyword": str, "count": int, "spans": [str]}}]}}

Watchlist (canonical_id | name | aliases):
{watchlist}

Keywords to count: {keywords}

Transcript:
{text}
"""


class AnthropicLLMClient:
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key) if self._api_key \
                else Anthropic()
        return self._client

    def _parse_response(self, payload: dict, keywords: list[str]) -> MentionExtraction:
        mentions = [SpokenMention(**m) for m in payload.get("mentions", [])]
        hits = [KeywordHit(**h) for h in payload.get("keyword_hits", [])]
        return MentionExtraction(mentions=mentions, keyword_hits=hits)

    def extract_mentions(self, transcript_text, watchlist, keywords) -> MentionExtraction:
        wl_lines = "\n".join(f"{cid} | {name} | {', '.join(aliases)}"
                             for cid, name, aliases in watchlist)
        prompt = _PROMPT.format(watchlist=wl_lines or "(empty)",
                                keywords=", ".join(keywords) or "(none)",
                                text=transcript_text)
        client = self._load()
        msg = client.messages.create(
            model=self._model, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        payload = json.loads(raw)
        return self._parse_response(payload, keywords)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_mentions_stage.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add src/celebvision/stages/mentions.py src/celebvision/llm/ tests/test_mentions_stage.py
git commit -m "feat: add mention/keyword stage and Anthropic LLM client"
```

---

### Task 11: Aggregate stage — build Report + celebrity_index

**Files:**
- Create: `src/celebvision/stages/aggregate.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `SceneWindow` (Task 8), `SpokenMention`/`KeywordHit`/`OnscreenFace`/`SceneReport`/`Report`/`CelebrityIndexEntry`/`JobSource` (Task 2).
- Produces (in `celebvision.stages.aggregate`):
  - `build_scene_report(scene, transcript_text, mentions, keyword_hits, faces) -> SceneReport`
  - `build_celebrity_index(scene_reports: list[SceneReport]) -> list[CelebrityIndexEntry]` — one entry per `canonical_id` seen in any modality; `scenes` sorted unique; `modalities` = subset of `["audio","face"]` present, sorted.
  - `build_report(job_id, source, watchlist_id, completed_at, duration_s, scene_reports) -> Report`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate.py
from celebvision.stages.aggregate import (
    build_scene_report, build_celebrity_index, build_report,
)
from celebvision.stages.scenes import SceneWindow
from celebvision.models import SpokenMention, OnscreenFace, JobSource


def _scene_report(sid, mention_id=None, face_id=None):
    scene = SceneWindow(scene_id=sid, start_s=sid, end_s=sid + 1, keyframes=["k.jpg"])
    mentions = ([SpokenMention(name="Lionel Messi", canonical_id=mention_id,
                 confidence=0.9, evidence_span="x")] if mention_id else [])
    faces = ([OnscreenFace(name="Lionel Messi", canonical_id=face_id, confidence=0.7,
              bbox=(0, 0, 1, 1), keyframe="k.jpg")] if face_id else [])
    return build_scene_report(scene, "text", mentions, [], faces)


def test_celebrity_index_merges_modalities_across_scenes():
    reports = [_scene_report(0, mention_id="messi"),
               _scene_report(1, face_id="messi"),
               _scene_report(2, face_id="ronaldo")]
    index = build_celebrity_index(reports)
    by_id = {e.canonical_id: e for e in index}
    assert sorted(by_id["messi"].scenes) == [0, 1]
    assert by_id["messi"].modalities == ["audio", "face"]
    assert by_id["ronaldo"].modalities == ["face"]
    assert by_id["ronaldo"].scenes == [2]

def test_build_report_assembles_everything():
    reports = [_scene_report(0, mention_id="messi")]
    report = build_report("j1", JobSource(kind="file", locator="a.mp4"),
                          "w1", "2026-07-13T00:00:00Z", 10.0, reports)
    assert report.job_id == "j1"
    assert len(report.scenes) == 1
    assert report.celebrity_index[0].canonical_id == "messi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the aggregate stage**

```python
# src/celebvision/stages/aggregate.py
from celebvision.stages.scenes import SceneWindow
from celebvision.models import (
    SpokenMention, KeywordHit, OnscreenFace, SceneReport, Report,
    CelebrityIndexEntry, JobSource,
)


def build_scene_report(scene: SceneWindow, transcript_text: str,
                       mentions: list[SpokenMention], keyword_hits: list[KeywordHit],
                       faces: list[OnscreenFace]) -> SceneReport:
    return SceneReport(
        scene_id=scene.scene_id, start_s=scene.start_s, end_s=scene.end_s,
        keyframes=scene.keyframes, transcript=transcript_text,
        spoken_mentions=mentions, keyword_hits=keyword_hits, onscreen_faces=faces,
    )


def build_celebrity_index(scene_reports: list[SceneReport]) -> list[CelebrityIndexEntry]:
    acc: dict[str, dict] = {}
    for sr in scene_reports:
        for m in sr.spoken_mentions:
            e = acc.setdefault(m.canonical_id,
                               {"name": m.name, "scenes": set(), "modalities": set()})
            e["scenes"].add(sr.scene_id)
            e["modalities"].add("audio")
        for f in sr.onscreen_faces:
            e = acc.setdefault(f.canonical_id,
                               {"name": f.name, "scenes": set(), "modalities": set()})
            e["scenes"].add(sr.scene_id)
            e["modalities"].add("face")
    return [CelebrityIndexEntry(canonical_id=cid, name=v["name"],
                                scenes=sorted(v["scenes"]),
                                modalities=sorted(v["modalities"]))
            for cid, v in acc.items()]


def build_report(job_id: str, source: JobSource, watchlist_id: str,
                 completed_at: str, duration_s: float,
                 scene_reports: list[SceneReport]) -> Report:
    return Report(
        job_id=job_id, source=source, duration_s=duration_s,
        watchlist_id=watchlist_id, completed_at=completed_at,
        scenes=scene_reports, celebrity_index=build_celebrity_index(scene_reports),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/celebvision/stages/aggregate.py tests/test_aggregate.py
git commit -m "feat: add aggregate stage building report and celebrity index"
```

---

### Task 12: Pipeline orchestrator + CLI (end-to-end)

**Files:**
- Create: `src/celebvision/pipeline.py`
- Create: `src/celebvision/enroll.py`
- Create: `src/celebvision/cli.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–11.
- Produces:
  - `run_pipeline(locator, watchlist, keywords, inference, llm, workdir, job_id, completed_at, face_threshold=0.35) -> Report` in `celebvision.pipeline` — runs classify → ingest → transcribe → detect_scenes → build_scene_windows → per-scene (faces + mentions) → aggregate. `completed_at` is injected (no wall-clock in core logic).
  - `enroll_identity(index, inference, canonical_id, name, aliases, image_paths) -> None` in `celebvision.enroll` — computes one embedding per image (first detected face) and adds to index.
  - `celebvision.cli:app` Typer app with commands `analyze` and `enroll`.

- [ ] **Step 1: Write the failing pipeline test** (fakes for inference/llm, fake ingest via monkeypatch)

```python
# tests/test_pipeline.py
import celebvision.pipeline as pipeline_mod
from celebvision.pipeline import run_pipeline
from celebvision.interfaces import (
    Transcript, TranscriptSegment, Word, FaceDetection, MentionExtraction,
)
from celebvision.models import SpokenMention


class FakeInference:
    def transcribe(self, audio_path):
        return Transcript(segments=[TranscriptSegment(
            start_s=0.0, end_s=5.0, text="Messi scored",
            words=[Word(text="Messi", start_s=1.0, end_s=1.5),
                   Word(text="scored", start_s=1.5, end_s=2.0)])])
    def analyze_faces(self, image_path):
        return [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0])]


class FakeLLM:
    def extract_mentions(self, transcript_text, watchlist, keywords):
        return MentionExtraction(
            mentions=[SpokenMention(name="Lionel Messi", canonical_id="messi",
                                    confidence=0.9, evidence_span=transcript_text)],
            keyword_hits=[])


class FakeWatchlist:
    watchlist_id = "w1"
    def search(self, embedding, threshold):
        return ("messi", "Lionel Messi", 0.8)
    def names(self): return [("messi", "Lionel Messi", ["messi"])]


def test_run_pipeline_end_to_end(tmp_path, monkeypatch):
    # Stub ingest + scene detection + keyframe so no ffmpeg/network is needed.
    monkeypatch.setattr(pipeline_mod, "ingest",
                        lambda source, workdir, **k: ("v.mp4", "a.wav"))
    monkeypatch.setattr(pipeline_mod, "detect_scenes",
                        lambda video_path, **k: [(0.0, 5.0)])
    monkeypatch.setattr(pipeline_mod, "extract_keyframe",
                        lambda v, t, out, **k: out)

    report = run_pipeline(
        locator="/tmp/a.mp4", watchlist=FakeWatchlist(), keywords=["scored"],
        inference=FakeInference(), llm=FakeLLM(), workdir=str(tmp_path),
        job_id="j1", completed_at="2026-07-13T00:00:00Z",
    )
    assert report.job_id == "j1"
    assert len(report.scenes) == 1
    scene = report.scenes[0]
    assert scene.spoken_mentions[0].canonical_id == "messi"
    assert scene.onscreen_faces[0].canonical_id == "messi"
    assert report.celebrity_index[0].modalities == ["audio", "face"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.pipeline'`

- [ ] **Step 3: Write the pipeline orchestrator**

```python
# src/celebvision/pipeline.py
from celebvision.media import classify_source, ingest, extract_keyframe
from celebvision.stages.scenes import detect_scenes, build_scene_windows
from celebvision.stages.faces import match_faces_in_scene
from celebvision.stages.mentions import scene_transcript_text, extract_scene_mentions
from celebvision.stages.aggregate import build_scene_report, build_report
from celebvision.interfaces import InferenceClient, LLMClient, WatchlistIndex
from celebvision.models import Report


def run_pipeline(locator: str, watchlist: WatchlistIndex, keywords: list[str],
                 inference: InferenceClient, llm: LLMClient, workdir: str,
                 job_id: str, completed_at: str, face_threshold: float = 0.35) -> Report:
    source = classify_source(locator)
    video_path, audio_path = ingest(source, workdir)
    transcript = inference.transcribe(audio_path)
    boundaries = detect_scenes(video_path)
    windows = build_scene_windows(video_path, boundaries, workdir,
                                  keyframe_fn=extract_keyframe)

    scene_reports = []
    for scene in windows:
        faces = match_faces_in_scene(scene, inference, watchlist, face_threshold)
        text = scene_transcript_text(transcript, scene)
        mentions, hits = extract_scene_mentions(text, watchlist, keywords, llm)
        scene_reports.append(build_scene_report(scene, text, mentions, hits, faces))

    return build_report(job_id, source, watchlist.watchlist_id, completed_at,
                        transcript.duration_s, scene_reports)
```

- [ ] **Step 4: Run pipeline test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the enroll test**

```python
# append to tests/test_pipeline.py
from celebvision.enroll import enroll_identity
from celebvision.watchlist.local_index import LocalWatchlistIndex


def test_enroll_identity_adds_embeddings():
    class FI:
        def analyze_faces(self, image_path):
            return [FaceDetection(bbox=(0, 0, 1, 1), embedding=[1.0, 0.0])]
        def transcribe(self, audio_path): raise NotImplementedError
    idx = LocalWatchlistIndex("w1")
    enroll_identity(idx, FI(), "messi", "Lionel Messi", ["messi"],
                    ["a.jpg", "b.jpg"])
    assert idx.names() == [("messi", "Lionel Messi", ["messi"])]
    assert idx.search([1.0, 0.0], threshold=0.5)[0] == "messi"
```

- [ ] **Step 6: Run enroll test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::test_enroll_identity_adds_embeddings -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.enroll'`

- [ ] **Step 7: Write the enroll module**

```python
# src/celebvision/enroll.py
from celebvision.interfaces import InferenceClient
from celebvision.watchlist.local_index import LocalWatchlistIndex


def enroll_identity(index: LocalWatchlistIndex, inference: InferenceClient,
                    canonical_id: str, name: str, aliases: list[str],
                    image_paths: list[str]) -> None:
    embeddings: list[list[float]] = []
    for path in image_paths:
        dets = inference.analyze_faces(path)
        if dets:  # take the first (largest) detected face
            embeddings.append(dets[0].embedding)
    if not embeddings:
        raise ValueError(f"no faces detected in reference images for {name}")
    index.add(canonical_id, name, aliases, embeddings)
```

- [ ] **Step 8: Run enroll test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Write the CLI test**

```python
# tests/test_cli.py
import json
from typer.testing import CliRunner
import celebvision.cli as cli_mod
from celebvision.cli import app
from celebvision.models import Report, JobSource

runner = CliRunner()


def test_analyze_writes_report(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    fake = Report(job_id="j1", source=JobSource(kind="file", locator="a.mp4"),
                  duration_s=1.0, watchlist_id="w1",
                  completed_at="2026-07-13T00:00:00Z", scenes=[], celebrity_index=[])
    # Stub heavy construction + pipeline so the CLI wiring is what's tested.
    monkeypatch.setattr(cli_mod, "_load_clients", lambda wl_path: (object(), object(),
                        object()))
    monkeypatch.setattr(cli_mod, "run_pipeline",
                        lambda **kwargs: fake)
    result = runner.invoke(app, ["analyze", "a.mp4", "--watchlist",
                                 str(tmp_path / "wl.npz"), "--out", str(out),
                                 "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written["job_id"] == "j1"
```

- [ ] **Step 10: Run CLI test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'celebvision.cli'`

- [ ] **Step 11: Write the CLI**

```python
# src/celebvision/cli.py
import json
import os
import uuid
from datetime import datetime, timezone
import typer
from celebvision.pipeline import run_pipeline
from celebvision.enroll import enroll_identity
from celebvision.inference.local_client import LocalInferenceClient
from celebvision.llm.anthropic_client import AnthropicLLMClient
from celebvision.watchlist.local_index import LocalWatchlistIndex

app = typer.Typer(help="Celebrity video analysis pipeline")


def _load_clients(watchlist_path: str):
    inference = LocalInferenceClient()
    llm = AnthropicLLMClient()
    watchlist = LocalWatchlistIndex.load(watchlist_path)
    return inference, llm, watchlist


@app.command()
def analyze(
    source: str = typer.Argument(..., help="YouTube URL, HLS .m3u8, or file path"),
    watchlist: str = typer.Option(..., help="Path to enrolled watchlist .npz"),
    out: str = typer.Option("report.json", help="Output report path"),
    workdir: str = typer.Option("./data/work", help="Scratch dir for media"),
    keyword: list[str] = typer.Option([], help="Keyword to count (repeatable)"),
):
    inference, llm, wl = _load_clients(watchlist)
    report = run_pipeline(
        locator=source, watchlist=wl, keywords=list(keyword),
        inference=inference, llm=llm, workdir=workdir,
        job_id=str(uuid.uuid4()),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(report.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(report.scenes)} scenes, "
               f"{len(report.celebrity_index)} celebrities)")


@app.command()
def enroll(
    watchlist_id: str = typer.Option(..., help="Watchlist id"),
    name: str = typer.Option(..., help="Celebrity display name"),
    canonical_id: str = typer.Option(..., help="Stable id"),
    images: str = typer.Option(..., help="Directory of reference photos"),
    out: str = typer.Option(..., help="Watchlist .npz path (created/updated)"),
    alias: list[str] = typer.Option([], help="Alias (repeatable)"),
):
    inference = LocalInferenceClient()
    index = (LocalWatchlistIndex.load(out) if os.path.exists(out)
             else LocalWatchlistIndex(watchlist_id))
    image_paths = [os.path.join(images, f) for f in sorted(os.listdir(images))
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    enroll_identity(index, inference, canonical_id, name, list(alias), image_paths)
    index.save(out)
    typer.echo(f"enrolled {name} ({len(image_paths)} images) -> {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 12: Run CLI test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 13: Run the full suite + lint**

Run: `python -m pytest -v && ruff check src tests`
Expected: all tests PASS (default `-m 'not slow'`), ruff clean.

- [ ] **Step 14: Commit**

```bash
git add src/celebvision/pipeline.py src/celebvision/enroll.py src/celebvision/cli.py \
        tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: add pipeline orchestrator and analyze/enroll CLI"
```

---

## Manual end-to-end verification (after Task 12)

Real run against YouTube. Prerequisites: install the ML backend + ffmpeg first
(`.venv/bin/pip install -e '.[local]'`; `brew install ffmpeg`), set `ANTHROPIC_API_KEY`;
models download on first use. This is the acceptance check, not an automated test:

```bash
# 1. Enroll a couple of celebrities from reference photos
celebvision enroll --watchlist-id demo --canonical-id messi --name "Lionel Messi" \
  --alias messi --images ./refs/messi --out ./data/demo.npz
celebvision enroll --watchlist-id demo --canonical-id ronaldo --name "Cristiano Ronaldo" \
  --alias cr7 --images ./refs/ronaldo --out ./data/demo.npz

# 2. Analyze a short YouTube clip
celebvision analyze "https://www.youtube.com/watch?v=<id>" \
  --watchlist ./data/demo.npz --keyword goal --keyword final \
  --out ./data/report.json --workdir ./data/work

# 3. Inspect
python -c "import json; r=json.load(open('data/report.json')); \
print(len(r['scenes']), 'scenes;', [c['name'] for c in r['celebrity_index']])"
```

Expected: `report.json` conforms to the schema, scenes have timestamps, and celebrities present by face or spoken mention appear in `celebrity_index` with the right modalities.

---

## Milestone exit criteria

- [ ] `python -m pytest` green (all non-slow tests).
- [ ] `ruff check src tests` clean.
- [ ] `celebvision enroll` builds a watchlist `.npz`.
- [ ] `celebvision analyze <youtube-url>` produces a schema-valid per-scene report with `celebrity_index`.
- [ ] No stage function imports faster-whisper / insightface / anthropic directly (all via interfaces) — grep to confirm.

## Handoff to M2

M2 (async infra) will: wrap each stage as a Kafka consumer; add the Coordinator DAG; add FastAPI submit/status/report endpoints; replace `LocalWatchlistIndex` with a `pgvector`-backed `WatchlistIndex` (same protocol); persist media/keyframes/report to MinIO and job state to Postgres; ship docker-compose. No changes to stage logic — only new adapters behind the Task 3 protocols.
