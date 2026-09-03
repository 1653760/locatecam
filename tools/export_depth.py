"""Export Depth Anything V2 Metric (Indoor, ViT-S) to a self-normalizing ONNX graph.

The exported graph takes raw RGB floats in [0, 255] with shape [1, 3, 518, 518]
and outputs a metric depth map in meters [1, 518, 518]. ImageNet mean/std
normalization and resize are baked into the graph, mirroring the proven
LocateCam/WalkCam deployment pattern (raw /255 input, int8 MatMul quant).
"""

import os
import shutil

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
INPUT_SIZE = 518  # 37 * 14 patches
REPO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def get_test_image():
    for url in (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://ultralytics.com/images/bus.jpg",
    ):
        try:
            data = requests.get(url, timeout=30).content
            with open("bus.jpg", "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"download failed: {e}")
    return False


def load_model():
    from transformers import DepthAnythingForDepthEstimation

    return DepthAnythingForDepthEstimation.from_pretrained(REPO_ID).eval()


def export_onnx(model, path):
    import torch
    import torch.nn.functional as F

    class Deploy(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, images):
            x = (images / 255.0 - self.mean) / self.std
            depth = self.net(pixel_values=x).predicted_depth
            if depth.dim() == 4:
                depth = depth.squeeze(1)
            if depth.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
                depth = F.interpolate(
                    depth.unsqueeze(1).float(),
                    size=(INPUT_SIZE, INPUT_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            return depth

    wrap = Deploy(model).eval()
    dummy = torch.randint(0, 256, (1, 3, INPUT_SIZE, INPUT_SIZE)).float()
    with torch.no_grad():
        out = wrap(dummy)
    print(f"torch output: shape={tuple(out.shape)} min={out.min():.2f} max={out.max():.2f}")
    try:
        torch.onnx.export(
            wrap, dummy, path, opset_version=17,
            input_names=["images"], output_names=["depth"], dynamo=False,
        )
    except TypeError:
        torch.onnx.export(wrap, dummy, path, opset_version=17, input_names=["images"], output_names=["depth"])
    print(f"fp32 onnx: {os.path.getsize(path) / 1e6:.1f} MB")


def quantize_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"int8 onnx: {os.path.getsize(int8_path) / 1e6:.1f} MB")


def region_median(depth, box_frac=(0.35, 0.55, 0.65, 0.95)):
    """Median depth of a normalized box region (x0, y0, x1, y1) in image fractions."""
    h, w = depth.shape
    x0 = int(box_frac[0] * w); y0 = int(box_frac[1] * h)
    x1 = int(box_frac[2] * w); y1 = int(box_frac[3] * h)
    return float(np.median(depth[y0:y1, x0:x1]))


def validate(path):
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    im = Image.open("bus.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]
    out = sess.run(None, {"images": x})[0]
    print(f"onnx output: shape={out.shape} dtype={out.dtype} min={out.min():.2f} max={out.max():.2f}")
    med = region_median(out[0] if out.ndim == 3 else out[0, 0])
    print(f"region median depth = {med:.2f} m")
    return med


def main():
    os.makedirs(ASSETS, exist_ok=True)
    if not get_test_image():
        raise SystemExit("no test image")
    model = load_model()
    export_onnx(model, "depth_fp32.onnx")
    quantize_int8("depth_fp32.onnx", "depth_int8.onnx")

    med32 = validate("depth_fp32.onnx")
    med8 = validate("depth_int8.onnx")

    assert 0.05 < med32 < 60, f"fp32 depth implausible: {med32}"
    assert abs(med8 - med32) / med32 < 0.20, f"int8 drift too large: {med32} vs {med8}"

    final = "depth_int8.onnx" if med8 > 0.05 else "depth_fp32.onnx"
    shutil.copyfile(final, os.path.join(ASSETS, "depth.onnx"))
    print(f"FINAL DEPTH MODEL: {final} ({os.path.getsize(final) / 1e6:.1f} MB, {REPO_ID})")


if __name__ == "__main__":
    main()
