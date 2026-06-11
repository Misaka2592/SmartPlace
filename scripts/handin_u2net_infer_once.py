import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_rgba", required=True)
    parser.add_argument("--output_mask", required=True)
    parser.add_argument("--handin_root", required=True)
    parser.add_argument("--model_type", default="u2netp")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    handin_root = os.path.abspath(args.handin_root)
    if handin_root not in sys.path:
        sys.path.insert(0, handin_root)

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from u2net import U2NetMatting

        device = None if args.device == "auto" else torch.device(args.device)
        matting = U2NetMatting(
            model_type=args.model_type,
            weight_path=os.path.abspath(args.weight),
            device=device,
            threshold=args.threshold,
        )

    image = Image.open(args.input).convert("RGBA")
    start = time.time()
    mask = matting.predict(image.convert("RGB"))
    elapsed = time.time() - start

    rgba = image.copy()
    rgba.putalpha(mask)
    rgba.save(args.output_rgba)
    mask.save(args.output_mask)

    foreground_pixel_ratio = float((np.array(mask, dtype=np.uint8) > 20).mean())
    result = {
        "input": args.input,
        "output_rgba": args.output_rgba,
        "output_mask": args.output_mask,
        "model_type": args.model_type,
        "weight": os.path.abspath(args.weight),
        "device": str(matting.device),
        "threshold": args.threshold,
        "input_size": image.size,
        "foreground_pixel_ratio": foreground_pixel_ratio,
        "inference_time": elapsed,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
