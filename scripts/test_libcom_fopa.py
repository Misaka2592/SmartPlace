import argparse
import os
import time

import numpy as np
from PIL import Image


def rgba_to_rgb_and_mask(foreground_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(foreground_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # 注意：FOPA 内部会按 "_" 解析文件名，所以这里不要使用 foreground_rgb 这种带下划线的名字
    fg_path = os.path.join(output_dir, "fg.png")
    mask_path = os.path.join(output_dir, "mask.png")

    Image.fromarray(rgb).save(fg_path)
    Image.fromarray(alpha).save(mask_path)

    return fg_path, mask_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--foreground", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="outputs/libcom_fopa_test")
    parser.add_argument("--fg_scale_num", type=int, default=8)
    parser.add_argument("--composite_num_choose", type=int, default=3)
    parser.add_argument("--composite_num", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = os.path.join(args.output_dir, "cache")
    heatmap_dir = os.path.join(args.output_dir, "heatmap")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    from libcom import FOPAHeatMapModel

    fg_rgb_path, fg_mask_path = rgba_to_rgb_and_mask(
        foreground_path=args.foreground,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("[libcom FOPA test]")
    print(f"[Input] background={args.background}")
    print(f"[Input] foreground={fg_rgb_path}")
    print(f"[Input] foreground_mask={fg_mask_path}")
    print(f"[Model] FOPAHeatMapModel device={args.device}")
    print(f"[Param] fg_scale_num={args.fg_scale_num}")
    print(f"[Param] composite_num_choose={args.composite_num_choose}")
    print(f"[Param] composite_num={args.composite_num}")
    print("=" * 80)

    start = time.time()

    net = FOPAHeatMapModel(device=args.device)

    bboxes, heatmaps = net(
        fg_rgb_path,
        fg_mask_path,
        args.background,
        cache_dir=cache_dir,
        heatmap_dir=heatmap_dir,
        fg_scale_num=args.fg_scale_num,
        composite_num_choose=args.composite_num_choose,
        composite_num=args.composite_num,
    )

    elapsed = time.time() - start

    print("[Output] bboxes:")
    print(bboxes)
    print("[Output] heatmaps:")
    print(heatmaps)
    print(f"[Output] cache_dir={cache_dir}")
    print(f"[Output] heatmap_dir={heatmap_dir}")
    print(f"[Time] elapsed={elapsed:.4f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()