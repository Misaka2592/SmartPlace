import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2


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
    score = net(args.background, args.foreground, bbox, foreground_mask=args.foreground_mask)
    return {
        "model": "FOSScoreModel",
        "model_type": "FOS_D",
        "description": "Scores foreground/background compatibility from geometry and semantics.",
        "score": float(score),
    }


def run_harmony(args, device) -> Dict:
    from libcom import HarmonyScoreModel

    net = HarmonyScoreModel(device=device, model_type="BargainNet")
    score = net(args.composite, args.composite_mask)
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
    output = net(args.composite, args.composite_mask, **kwargs)
    out_path = os.path.join(args.output_dir, f"harmonized_{model_type}.png")
    cv2.imwrite(out_path, output)
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
