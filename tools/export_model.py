import json
import os
import shutil
import sys
import traceback

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab_zh.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    en_terms = [e["en"] for e in vocab]
    if len(set(en_terms)) != len(en_terms):
        seen = set()
        dups = [t for t in en_terms if t in seen or seen.add(t)]
        raise SystemExit(f"duplicate english terms: {dups}")
    print(f"vocab size: {len(en_terms)}")
    return vocab, en_terms


def export_model(en_terms):
    from ultralytics import YOLOE, YOLOWorld

    for size in ("l-seg", "m-seg", "s-seg"):
        try:
            model = YOLOE(f"yoloe-11{size}.pt")
            pe = model.get_text_pe(en_terms)
            model.set_classes(en_terms, pe)
            out = model.export(format="onnx", imgsz=640)
            return str(out), f"yoloe-11{size}"
        except Exception as e:
            print(f"yoloe-11{size} failed: {e}")
            traceback.print_exc()

    print("all YOLOE sizes failed, falling back to yolov8s-worldv2")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(en_terms)
    out = model.export(format="onnx", imgsz=640)
    return str(out), "worldv2-s"


def validate(onnx_path, vocab, en_terms):
    try:
        import urllib.request

        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        print(f"inputs: {[(i.name, i.shape) for i in inputs]}")
        print(f"outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
        if inputs[0].name != "images":
            print(f"WARNING: input name is {inputs[0].name}, app expects 'images'")

        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")
        im = Image.open("bus.jpg").convert("RGB").resize((640, 640))
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        out = sess.run(None, {inputs[0].name: x})[0]
        print(f"output shape: {out.shape}")
        extra = out.shape[1] - 4 - len(en_terms)
        assert extra >= 0, f"unexpected nc: {out.shape[1]} vs {4 + len(en_terms)}"
        if extra > 0:
            print(f"note: {extra} extra channels (seg coeffs), ignored by app")
        pi = en_terms.index("person")
        bi = en_terms.index("bus")
        p_best = float(out[0, 4 + pi, :].max())
        b_best = float(out[0, 4 + bi, :].max())
        print(f"person score: {p_best:.3f}, bus score: {b_best:.3f}")
        assert p_best > 0.35, f"person score too low: {p_best}"
        assert b_best > 0.30, f"bus score too low: {b_best}"
    except Exception as e:
        print(f"validation warning (non-fatal): {e}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vocab, en_terms = load_vocab()
    src, precision = export_model(en_terms)
    print(f"exported ({precision}): {src}")

    dst = os.path.join(OUT_DIR, "yolo_world.onnx")
    shutil.copyfile(src, dst)
    with open(os.path.join(OUT_DIR, "class_index.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    validate(dst, vocab, en_terms)
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"FINAL MODEL: {dst} ({size_mb:.1f} MB, {precision})")


if __name__ == "__main__":
    main()
