import argparse
import contextlib
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composites", nargs="+", required=True)
    parser.add_argument("--masks", nargs="+", required=True)
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
        import torch
        from opa import OPAScorer
        from opa import config as opa_config
        from opa.enums import ModelType

        model_type = {
            "resnet": ModelType.RESNET,
            "tiny_cnn": ModelType.TINY_CNN,
            "mobilenet_v2": ModelType.MOBILENET_V2,
        }[args.model_name.lower()]

        device = None if args.device == "auto" else torch.device(args.device)
        if model_type == ModelType.RESNET:
            cfg = opa_config.Config(model_type, layers=args.layers, width_factor=args.width_factor)
            scorer = OPAScorer(
                weight_path=os.path.abspath(args.weight),
                config_inst=cfg,
                layers=args.layers,
                model_type=model_type,
                device=device,
            )
        else:
            cfg = opa_config.Config(model_type)
            scorer = OPAScorer(
                weight_path=os.path.abspath(args.weight),
                config_inst=cfg,
                model_type=model_type,
                device=device,
            )
    return scorer


def main():
    args = parse_args()
    if len(args.composites) != len(args.masks):
        raise ValueError(f"composites count != masks count: {len(args.composites)} vs {len(args.masks)}")

    scorer = build_scorer(args)
    results = []
    start_all = time.time()

    for idx, (comp_path, mask_path) in enumerate(zip(args.composites, args.masks), start=1):
        if not os.path.exists(comp_path):
            raise FileNotFoundError(f"composite not found: {comp_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"mask not found: {mask_path}")

        start_one = time.time()
        score = float(scorer.score_from_path(comp_path, mask_path))
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
        "device": str(scorer.device),
        "weight": os.path.abspath(args.weight),
        "model_name": args.model_name,
        "layers": args.layers,
        "width_factor": args.width_factor,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
