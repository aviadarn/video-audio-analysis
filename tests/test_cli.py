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
