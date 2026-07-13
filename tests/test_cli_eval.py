import json
from typer.testing import CliRunner
from celebvision.cli import app

runner = CliRunner()


def test_eval_command_writes_result(tmp_path):
    report = {
        "job_id": "j1", "source": {"kind": "file", "locator": "a.mp4"},
        "duration_s": 5.0, "watchlist_id": "w1", "completed_at": "",
        "scenes": [], "celebrity_index": [
            {"name": "Messi", "canonical_id": "messi", "scenes": [0],
             "modalities": ["audio"]}]}
    truth = {"job_id": "j1", "expected": ["messi", "ronaldo"]}
    rp = tmp_path / "report.json"; rp.write_text(json.dumps(report))
    tp = tmp_path / "truth.json"; tp.write_text(json.dumps(truth))
    out = tmp_path / "eval.json"
    result = runner.invoke(app, ["eval", "--report", str(rp), "--truth", str(tp),
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written["job_id"] == "j1"
    # detected combined {messi}; expected {messi, ronaldo} -> recall 0.5, precision 1.0
    assert abs(written["video"]["combined"]["recall"] - 0.5) < 1e-9
