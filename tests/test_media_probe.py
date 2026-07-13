from celebvision.media import probe_duration


def test_probe_duration_parses_ffprobe_output():
    def fake_runner(cmd, capture_output, text, check):
        class R:
            stdout = "12.34\n"
            returncode = 0
        assert "ffprobe" in cmd[0]
        return R()
    assert probe_duration("v.mp4", runner=fake_runner) == 12.34

def test_probe_duration_returns_zero_on_error():
    def fake_runner(cmd, capture_output, text, check):
        raise RuntimeError("no ffprobe")
    assert probe_duration("v.mp4", runner=fake_runner) == 0.0
