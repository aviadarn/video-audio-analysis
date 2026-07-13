import celebvision

def test_has_version():
    assert isinstance(celebvision.__version__, str)
    assert celebvision.__version__.count(".") >= 1
