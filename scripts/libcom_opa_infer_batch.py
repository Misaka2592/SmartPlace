import argparse
import contextlib
import json
import os
import sys
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composites", nargs="+", required=True)
    parser.add_argument("--masks", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_type", default="SimOPA")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.composites) != len(args.masks):
        raise ValueError(f"composites 数量和 masks 数量不一致: {len(args.composites)} vs {len(args.masks)}")

    with contextlib.redirect_stdout(sys.stderr):
        from libcom import OPAScoreModel
        net = OPAScoreModel(device=args.device, model_type=args.model_type)

    results = []
    start_all = time.time()

    for idx, (comp_path, mask_path) in enumerate(zip(args.composites, args.masks), start=1):
        if not os.path.exists(comp_path):
            raise FileNotFoundError(f"composite not found: {comp_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"mask not found: {mask_path}")

        comp_img = cv2.imread(comp_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if comp_img is None:
            raise RuntimeError(f"cv2 failed to read composite: {comp_path}")
        if mask_img is None:
            raise RuntimeError(f"cv2 failed to read mask: {mask_path}")

        start_one = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            score = float(net(comp_img, mask_img))
        elapsed_one = time.time() - start_one

        results.append({
            "index": idx,
            "composite": comp_path,
            "mask": mask_path,
            "score": score,
            "inference_time": elapsed_one,
        })

    output = {
        "results": results,
        "batch_time": time.time() - start_all,
        "device": args.device,
        "model_type": args.model_type,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
