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


def probe_duration(video_path: str, runner=subprocess.run) -> float:
    try:
        result = runner(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
