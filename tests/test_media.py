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
