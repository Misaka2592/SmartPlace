import argparse
import contextlib
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--handin_root", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--model_name", default="resnet")
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--width_factor", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_scorer(args):
    handin_root = os.path.abspath(args.handin_root)
    if handin_root not in sys.path:
        sys.path.insert(0, handin_root)

    with contextlib.redirect_stdout(sys.stderr):
        from opa import OPAScorer
        from opa import config as opa_config
        from opa.enums import ModelType

        model_type = {
            "resnet": ModelType.RESNET,
            "tiny_cnn": ModelType.TINY_CNN,
            "mobilenet_v2": ModelType.MOBILENET_V2,
        }[args.model_name.lower()]

        if model_type == ModelType.RESNET:
            cfg = opa_config.Config(model_type, layers=args.layers, width_factor=args.width_factor)
            scorer = OPAScorer(
                weight_path=os.path.abspath(args.weight),
                config_inst=cfg,
                layers=args.layers,
                model_type=model_type,
                device=None if args.device == "auto" else __import__("torch").device(args.device),
            )
        else:
            cfg = opa_config.Config(model_type)
            scorer = OPAScorer(
                weight_path=os.path.abspath(args.weight),
                config_inst=cfg,
                model_type=model_type,
                device=None if args.device == "auto" else __import__("torch").device(args.device),
            )
    return scorer


def main():
    args = parse_args()
    if not os.path.exists(args.composite):
        raise FileNotFoundError(f"Cannot read composite image: {args.composite}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Cannot read mask image: {args.mask}")

    start = time.time()
    scorer = build_scorer(args)
    score = float(scorer.score_from_path(args.composite, args.mask))
    elapsed = time.time() - start

    result = {
        "score": score,
        "device": str(scorer.device),
        "weight": os.path.abspath(args.weight),
        "model_name": args.model_name,
        "layers": args.layers,
        "width_factor": args.width_factor,
        "composite": args.composite,
        "mask": args.mask,
        "inference_time": elapsed,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
