import importlib
import math
import os
import shutil
import sys

import numpy as np
import requests
import torch

OSTRACK_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/OSTrack"
ASSETS = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
WEIGHTS_URL = "https://hf-mirror.com/eek/OSTrack_vitb_384_mae_ce_32x4_ep300/resolve/main/OSTrack_vitb_384_mae_ce_32x4_ep300.safetensors"

TEMPLATE = 192
SEARCH = 384
FEAT = 24
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_net():
    sys.path.insert(0, OSTRACK_DIR)
    config_module = importlib.import_module("lib.config.ostrack.config")
    cfg = config_module.cfg
    cfg_file = os.path.join(OSTRACK_DIR, "experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml")
    try:
        config_module.update_config_from_file(cfg_file)
    except TypeError:
        config_module.update_config_from_file(cfg, cfg_file)

    from lib.models.ostrack import build_ostrack

    net = build_ostrack(cfg, training=False)
    net.cpu()
    net.eval()
    return net


def load_weights(net):
    from safetensors.torch import load_file

    r = requests.get(WEIGHTS_URL, timeout=600)
    with open("ostrack.safetensors", "wb") as f:
        f.write(r.content)
    sd = load_file("ostrack.safetensors")
    print(f"raw keys: {len(sd)}, sample: {list(sd.keys())[:3]}")

    candidates = [sd]
    if any(k.startswith("net.") for k in sd):
        candidates.insert(0, {k[len("net."):]: v for k, v in sd.items()})
    if any(k.startswith("module.") for k in sd):
        candidates.append({k[len("module."):]: v for k, v in sd.items()})

    for c in candidates:
        missing, unexpected = net.load_state_dict(c, strict=False)
        print(f"missing={len(missing)} unexpected={len(unexpected)}")
        if not missing:
            if unexpected:
                print(f"unexpected (ignored): {unexpected[:5]}")
            return True
        print(f"missing sample: {missing[:5]}")
    return False


def export_onnx(net, path):
    dummy = (torch.randn(1, 3, TEMPLATE, TEMPLATE), torch.randn(1, 3, SEARCH, SEARCH))
    kwargs = dict(
        opset_version=17,
        input_names=["z", "x"],
        output_names=["score_map", "size_map", "offset_map"],
    )
    try:
        torch.onnx.export(net, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, dummy, path, **kwargs)
    try:
        import onnxsim

        model, ok = onnxsim.simplify(path)
        if ok:
            import onnx

            onnx.save(model, path)
            print("onnxsim ok")
    except Exception as e:
        print(f"onnxsim skipped: {e}")


def crop(frame, box, factor, out_sz):
    H, W, _ = frame.shape
    x, y, w, h = box
    s = int(math.ceil(math.sqrt(w * h) * factor))
    s = max(s, 16)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(cx - s / 2))
    y0 = int(round(cy - s / 2))
    img = np.zeros((out_sz, out_sz, 3), dtype=np.float32)
    for oy in range(out_sz):
        sy = y0 + oy * (s - 1) // (out_sz - 1)
        for ox in range(out_sz):
            sx = x0 + ox * (s - 1) // (out_sz - 1)
            if 0 <= sx < W and 0 <= sy < H:
                img[oy, ox] = frame[sy, sx].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    t = img.transpose(2, 0, 1)[None].astype(np.float32)
    rf = out_sz / s
    return t, rf


def hann2d(n):
    h = 0.5 * (1 - np.cos(2 * math.pi * (np.arange(1, n + 1)) / (n + 1)))
    return (h.reshape(-1, 1) * h.reshape(1, -1)).astype(np.float32).flatten()


def decode(score_map, size_map, offset_map, prev_box, rf):
    resp = hann2d(FEAT) * score_map.flatten()
    idx = int(resp.argmax())
    idx_y, idx_x = idx // FEAT, idx % FEAT
    score = float(resp[idx])
    w_n = float(size_map[0, idx_y, idx_x])
    h_n = float(size_map[1, idx_y, idx_x])
    off_x = float(offset_map[0, idx_y, idx_x])
    off_y = float(offset_map[1, idx_y, idx_x])
    cx = (idx_x + off_x) / FEAT * SEARCH
    cy = (idx_y + off_y) / FEAT * SEARCH
    w = w_n * SEARCH
    h = h_n * SEARCH
    px, py, pw, ph = prev_box
    half = 0.5 * SEARCH / rf
    cx_real = cx / rf + (px + 0.5 * pw) - half
    cy_real = cy / rf + (py + 0.5 * ph) - half
    w_real = w / rf
    h_real = h / rf
    return [cx_real - 0.5 * w_real, cy_real - 0.5 * h_real, w_real, h_real], score


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0


def synth_test(onnx_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    zname = names[0]
    xname = names[1]

    W, H = 480, 640
    f1 = np.full((H, W, 3), 128, dtype=np.uint8)
    f1[220:300, 200:280] = 230
    dx = 30
    f2 = np.full((H, W, 3), 120, dtype=np.uint8)
    f2[220:300, 200 + dx:280 + dx] = 230
    box = [200, 220, 80, 80]
    gt = [200 + dx, 220, 80, 80]

    z, _ = crop(f1, box, 2.0, TEMPLATE)
    x, rf = crop(f2, box, 4.0, SEARCH)
    outs = sess.run(None, {zname: z, xname: x})
    pred, score = decode(outs[0][0], outs[1][0], outs[2][0], box, rf)
    val = iou(pred, gt)
    print(f"synthetic: pred={list(map(lambda v: round(v, 1), pred))}, gt={gt}, IoU={val:.3f}, score={score:.3f}")
    return val > 0.35


def main():
    os.makedirs(ASSETS, exist_ok=True)
    net = build_net()
    if not load_weights(net):
        raise SystemExit("failed to load weights into model")
    fp32 = "tracker_fp32.onnx"
    export_onnx(net, fp32)
    print(f"fp32 size: {os.path.getsize(fp32) / 1e6:.1f} MB")
    if not synth_test(fp32):
        raise SystemExit("fp32 tracker failed synthetic test")

    final = fp32
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(fp32, "tracker_int8.onnx", weight_type=QuantType.QInt8)
        print(f"int8 size: {os.path.getsize('tracker_int8.onnx') / 1e6:.1f} MB")
        if synth_test("tracker_int8.onnx"):
            final = "tracker_int8.onnx"
        else:
            print("int8 failed synthetic test, using fp32")
    except Exception as e:
        print(f"quantization skipped: {e}")

    shutil.copyfile(final, os.path.join(ASSETS, "tracker.onnx"))
    print(f"FINAL TRACKER: {final} -> assets ({os.path.getsize(final) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
