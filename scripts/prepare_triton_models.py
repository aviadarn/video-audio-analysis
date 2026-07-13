import os
import shutil
from pathlib import Path

_BUFFALO_DIR = Path(os.path.expanduser("~/.insightface/models/buffalo_l"))
_ARCFACE_ONNX = "w600k_r50.onnx"


def render_config_pbtxt(name, input_name, input_dims, output_name, output_dims,
                        max_batch_size=16, kind="KIND_CPU"):
    in_dims = ", ".join(str(d) for d in input_dims)
    out_dims = ", ".join(str(d) for d in output_dims)
    return (
        f'name: "{name}"\n'
        'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch_size}\n"
        "dynamic_batching { }\n"
        f'input [ {{ name: "{input_name}" data_type: TYPE_FP32 dims: [ {in_dims} ] }} ]\n'
        f'output [ {{ name: "{output_name}" data_type: TYPE_FP32 dims: [ {out_dims} ] }} ]\n'
        f"instance_group [ {{ kind: {kind} }} ]\n"
    )


def _ensure_buffalo():
    if (_BUFFALO_DIR / _ARCFACE_ONNX).exists():
        return
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))


def _make_batchable(model_path):
    import onnx
    m = onnx.load(model_path)
    for t in list(m.graph.input) + list(m.graph.output):
        dim0 = t.type.tensor_type.shape.dim[0]
        dim0.ClearField("dim_value")
        dim0.dim_param = "N"
    onnx.save(m, model_path)


def _introspect(onnx_path):
    import onnx
    m = onnx.load(onnx_path)
    inp = m.graph.input[0]
    out = m.graph.output[0]
    in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim][1:]   # drop batch
    out_dims = [d.dim_value for d in out.type.tensor_type.shape.dim][1:]
    return inp.name, in_dims, out.name, out_dims


def prepare(model_repo="model_repository", cache_dir=None, kind="KIND_CPU"):
    _ensure_buffalo()
    src = (Path(cache_dir) if cache_dir else _BUFFALO_DIR) / _ARCFACE_ONNX
    dst_dir = Path(model_repo) / "arcface" / "1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    model_path = dst_dir / "model.onnx"
    shutil.copyfile(src, model_path)
    _make_batchable(str(model_path))
    in_name, in_dims, out_name, out_dims = _introspect(str(model_path))
    config_path = Path(model_repo) / "arcface" / "config.pbtxt"
    config_path.write_text(
        render_config_pbtxt("arcface", in_name, in_dims, out_name, out_dims, kind=kind))
    return str(model_path), str(config_path)


if __name__ == "__main__":
    import sys
    kind = "KIND_CPU"
    if "--kind" in sys.argv:
        kind = sys.argv[sys.argv.index("--kind") + 1]
    mp, cp = prepare(kind=kind)
    print(f"model: {mp}\nconfig: {cp}")
