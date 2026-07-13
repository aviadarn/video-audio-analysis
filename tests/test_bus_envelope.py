from celebvision.bus import Message, encode, decode


def test_envelope_round_trip():
    m = Message(job_id="j1", stage="faces", scene_id=3, payload={"k": "v"})
    again = decode(encode(m))
    assert again.job_id == "j1"
    assert again.stage == "faces"
    assert again.scene_id == 3
    assert again.payload == {"k": "v"}

def test_envelope_scene_id_optional():
    m = decode(encode(Message(job_id="j1", stage="ingest")))
    assert m.scene_id is None
    assert m.payload == {}
