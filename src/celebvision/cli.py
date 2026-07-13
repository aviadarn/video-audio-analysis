import os
import uuid
from datetime import datetime, timezone
import typer
from celebvision.pipeline import run_pipeline
from celebvision.enroll import enroll_identity
from celebvision.inference.local_client import LocalInferenceClient
from celebvision.llm.anthropic_client import AnthropicLLMClient
from celebvision.watchlist.local_index import LocalWatchlistIndex
from celebvision.models import Report
from celebvision.eval import GroundTruth, score_report

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


@app.command()
def eval(
    report: str = typer.Option(..., help="Path to a report.json"),
    truth: str = typer.Option(..., help="Path to a ground-truth.json"),
    out: str = typer.Option("eval.json", help="Output EvalResult path"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    with open(truth) as f:
        gt = GroundTruth.model_validate_json(f.read())
    result = score_report(rep, gt)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))
    typer.echo(f"combined F1={result.video['combined'].f1:.3f} -> {out}")


if __name__ == "__main__":
    app()
