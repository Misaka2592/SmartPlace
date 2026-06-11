import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--foreground", required=True)
    parser.add_argument("--foreground_mask", required=True)
    parser.add_argument("--composite", required=True)
    parser.add_argument("--composite_mask", required=True)
    parser.add_argument("--bbox", required=True, help="JSON list: [x1, y1, x2, y2]")
    parser.add_argument("--models", nargs="+", default=["fopa", "fos", "harmony", "pctnet"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="outputs/libcom_multimodel")
    parser.add_argument("--lbm_steps", type=int, default=4)
    parser.add_argument("--lbm_resolution", type=int, default=768)
    return parser.parse_args()


def json_safe(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def read_image_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
      return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_mask_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def write_image_unicode(path: str, image) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {path}")
    encoded.tofile(path)


def normalize_device(device: str):
    if isinstance(device, str) and device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    try:
        return int(device)
    except Exception:
        return device


def run_with_error_capture(name: str, fn):
    start = time.time()
    try:
        result = fn()
        result["ok"] = True
        result["time_sec"] = time.time() - start
        return result
    except Exception as exc:
        return {
            "ok": False,
            "model": name,
            "error": repr(exc),
            "time_sec": time.time() - start,
        }


def run_fopa(args, device) -> Dict:
    from libcom import FOPAHeatMapModel

    net = FOPAHeatMapModel(device=device)
    cache_dir = os.path.join(args.output_dir, "fopa_cache")
    heatmap_dir = os.path.join(args.output_dir, "fopa_heatmap")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    bboxes, heatmaps = net(
        args.foreground,
        args.foreground_mask,
        args.background,
        cache_dir=cache_dir,
        heatmap_dir=heatmap_dir,
    )
    return {
        "model": "FOPAHeatMapModel",
        "description": "Predicts rationality heatmaps over locations/scales.",
        "bboxes": json_safe(bboxes),
        "heatmaps": [str(path) for path in heatmaps],
        "preview_path": str(heatmaps[0]) if heatmaps else "",
    }


def run_fos(args, device, bbox: List[int]) -> Dict:
    from libcom import FOSScoreModel

    net = FOSScoreModel(device=device, model_type="FOS_D")
    background = read_image_unicode(args.background)
    foreground = read_image_unicode(args.foreground)
    foreground_mask = read_image_unicode(args.foreground_mask)
    if background is None:
        raise RuntimeError(f"Failed to decode background image: {args.background}")
    if foreground is None:
        raise RuntimeError(f"Failed to decode foreground image: {args.foreground}")
    if foreground_mask is None:
        raise RuntimeError(f"Failed to decode foreground mask: {args.foreground_mask}")
    bg_h, bg_w = background.shape[:2]
    x1, y1, x2, y2 = bbox
    clipped_bbox = [
        max(0, min(int(x1), bg_w - 1)),
        max(0, min(int(y1), bg_h - 1)),
        max(1, min(int(x2), bg_w)),
        max(1, min(int(y2), bg_h)),
    ]
    if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
        raise RuntimeError(f"Invalid clipped bbox for FOS: original={bbox}, clipped={clipped_bbox}")
    score = net(background, foreground, clipped_bbox, foreground_mask=foreground_mask)
    return {
        "model": "FOSScoreModel",
        "model_type": "FOS_D",
        "description": "Scores foreground/background compatibility from geometry and semantics.",
        "score": float(score),
        "bbox_used": clipped_bbox,
    }


def run_harmony(args, device) -> Dict:
    from libcom import HarmonyScoreModel

    net = HarmonyScoreModel(device=device, model_type="BargainNet")
    composite = read_image_unicode(args.composite)
    composite_mask = read_mask_unicode(args.composite_mask)
    if composite is None:
        raise RuntimeError(f"Failed to decode composite image: {args.composite}")
    if composite_mask is None:
        raise RuntimeError(f"Failed to decode composite mask: {args.composite_mask}")
    score = net(composite, composite_mask)
    return {
        "model": "HarmonyScoreModel",
        "model_type": "BargainNet",
        "description": "Scores foreground/background visual harmony.",
        "score": float(score),
    }


def run_harmonization(args, device, model_type: str) -> Dict:
    from libcom.image_harmonization import ImageHarmonizationModel

    net = ImageHarmonizationModel(device=device, model_type=model_type)
    kwargs = {}
    if model_type == "LBM":
        kwargs = {"steps": args.lbm_steps, "resolution": args.lbm_resolution}
    if model_type == "LBM":
        output = net(args.composite, args.composite_mask, **kwargs)
    else:
        composite = read_image_unicode(args.composite)
        composite_mask = read_mask_unicode(args.composite_mask)
        if composite is None:
            raise RuntimeError(f"Failed to decode composite image: {args.composite}")
        if composite_mask is None:
            raise RuntimeError(f"Failed to decode composite mask: {args.composite_mask}")
        output = net(composite, composite_mask, **kwargs)
    out_path = os.path.join(args.output_dir, f"harmonized_{model_type}.png")
    write_image_unicode(out_path, output)
    return {
        "model": "ImageHarmonizationModel",
        "model_type": model_type,
        "description": "Adjusts foreground illumination/color to match the background.",
        "output_path": out_path,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bbox = [int(v) for v in json.loads(args.bbox)]
    device = normalize_device(args.device)
    requested = {item.lower() for item in args.models}

    results = []
    start_all = time.time()

    with contextlib.redirect_stdout(sys.stderr):
        if "fopa" in requested:
            results.append(run_with_error_capture("FOPAHeatMapModel", lambda: run_fopa(args, device)))
        if "fos" in requested:
            results.append(run_with_error_capture("FOSScoreModel", lambda: run_fos(args, device, bbox)))
        if "harmony" in requested:
            results.append(run_with_error_capture("HarmonyScoreModel", lambda: run_harmony(args, device)))
        if "pctnet" in requested:
            results.append(run_with_error_capture("ImageHarmonizationModel.PCTNet", lambda: run_harmonization(args, device, "PCTNet")))
        if "lbm" in requested:
            results.append(run_with_error_capture("ImageHarmonizationModel.LBM", lambda: run_harmonization(args, device, "LBM")))

    output = {
        "ok": any(item.get("ok") for item in results),
        "models": sorted(requested),
        "results": results,
        "output_dir": args.output_dir,
        "total_time_sec": time.time() - start_all,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
