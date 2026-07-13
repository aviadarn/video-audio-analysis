import os
import pytest
from scripts.prepare_triton_models import render_config_pbtxt


def test_render_config_pbtxt_has_required_fields():
    txt = render_config_pbtxt("arcface", "input.1", [3, 112, 112], "683", [512])
    assert 'name: "arcface"' in txt
    assert 'platform: "onnxruntime_onnx"' in txt
    assert "max_batch_size: 16" in txt
    assert "dynamic_batching" in txt
    assert 'name: "input.1"' in txt
    assert "dims: [ 3, 112, 112 ]" in txt
    assert 'name: "683"' in txt
    assert "dims: [ 512 ]" in txt
    assert "KIND_CPU" in txt


@pytest.mark.slow
def test_prepare_copies_model_and_writes_config(tmp_path):
    from scripts.prepare_triton_models import prepare
    model_path, config_path = prepare(model_repo=str(tmp_path))
    assert os.path.exists(model_path) and model_path.endswith("arcface/1/model.onnx")
    assert os.path.getsize(model_path) > 1_000_000
    cfg = open(config_path).read()
    assert 'name: "arcface"' in cfg
    assert 'name: "input.1"' in cfg  # verified ArcFace input tensor name
    assert 'name: "683"' in cfg      # verified ArcFace output tensor name


@pytest.mark.slow
def test_prepared_model_has_dynamic_batch(tmp_path):
    import onnx
    from scripts.prepare_triton_models import prepare
    model_path, _ = prepare(model_repo=str(tmp_path))
    m = onnx.load(model_path)
    out0 = m.graph.output[0].type.tensor_type.shape.dim[0]
    assert out0.dim_param != "" or out0.dim_value == 0  # dynamic batch, not fixed 1
    in0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    assert in0.dim_param != "" or in0.dim_value == 0
