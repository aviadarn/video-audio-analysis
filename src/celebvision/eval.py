from pydantic import BaseModel
from celebvision.models import Report


class GroundTruth(BaseModel):
    job_id: str
    expected: list[str]
    expected_by_modality: dict[str, list[str]] = {}
    expected_by_scene: dict[int, list[str]] = {}


class ModalityScore(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class EvalResult(BaseModel):
    job_id: str
    video: dict[str, ModalityScore]
    scene_mean_f1: float | None = None


def _prf(expected: set[str], detected: set[str]) -> ModalityScore:
    tp = len(expected & detected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ModalityScore(precision=precision, recall=recall, f1=f1,
                         tp=tp, fp=fp, fn=fn)


def _expected_for(gt: GroundTruth, modality: str) -> set[str]:
    return set(gt.expected_by_modality.get(modality, gt.expected))


def score_report(report: Report, gt: GroundTruth) -> EvalResult:
    audio = {m.canonical_id for s in report.scenes for m in s.spoken_mentions}
    face = {f.canonical_id for s in report.scenes for f in s.onscreen_faces}
    combined = {e.canonical_id for e in report.celebrity_index}

    video = {
        "audio": _prf(_expected_for(gt, "audio"), audio),
        "face": _prf(_expected_for(gt, "face"), face),
        "combined": _prf(set(gt.expected), combined),
    }

    scene_mean_f1 = None
    if gt.expected_by_scene:
        by_scene = {s.scene_id: {e.canonical_id for e in s.spoken_mentions}
                    | {f.canonical_id for f in s.onscreen_faces}
                    for s in report.scenes}
        f1s = []
        # by design: only scenes present in both report and ground truth (spec §5)
        for sid, exp in gt.expected_by_scene.items():
            if sid in by_scene:
                f1s.append(_prf(set(exp), by_scene[sid]).f1)
        scene_mean_f1 = sum(f1s) / len(f1s) if f1s else None

    return EvalResult(job_id=report.job_id, video=video, scene_mean_f1=scene_mean_f1)
