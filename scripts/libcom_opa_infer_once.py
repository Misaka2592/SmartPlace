import argparse
import json
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_type", default="SimOPA")
    args = parser.parse_args()

    from libcom import OPAScoreModel

    image = cv2.imread(args.composite, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read composite image: {args.composite}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {args.mask}")

    start = time.time()

    net = OPAScoreModel(device=args.device, model_type=args.model_type)
    score = float(net(image, mask))

    elapsed = time.time() - start

    result = {
        "score": score,
        "device": args.device,
        "model_type": args.model_type,
        "composite": args.composite,
        "mask": args.mask,
        "inference_time": elapsed,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()