from types import SimpleNamespace
import pytest
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

@pytest.mark.slow
def test_transcribe_real_audio(tmp_path):
    # Generate 1s of silence via ffmpeg; assert it runs without error.
    import subprocess
    wav = str(tmp_path / "s.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=16000:cl=mono", "-t", "1", wav], check=True)
    t = LocalInferenceClient(whisper_model="tiny").transcribe(wav)
    assert isinstance(t.duration_s, float)
