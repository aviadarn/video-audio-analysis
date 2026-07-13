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
