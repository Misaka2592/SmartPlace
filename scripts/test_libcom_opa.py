import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image


def rgba_to_rgb_and_mask(foreground_path: str, output_dir: str):
    """
    如果前景是 RGBA，则拆成 RGB foreground 和 alpha mask。
    如果前景不是 RGBA，则生成全白 mask。
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(foreground_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg_path = os.path.join(output_dir, "foreground_rgb.png")
    mask_path = os.path.join(output_dir, "foreground_mask.png")

    Image.fromarray(rgb).save(fg_path)
    Image.fromarray(alpha).save(mask_path)

    return fg_path, mask_path


def build_bbox_from_xy_scale(bg_path, fg_path, x, y, scale):
    bg = Image.open(bg_path).convert("RGB")
    fg = Image.open(fg_path).convert("RGB")

    bg_w, bg_h = bg.size
    fg_w, fg_h = fg.size

    target_w = int(bg_w * scale)
    ratio = target_w / max(1, fg_w)
    target_h = int(fg_h * ratio)

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + target_w)
    y2 = int(y + target_h)

    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, help="背景图路径")
    parser.add_argument("--foreground", required=True, help="前景图路径，建议 RGBA PNG")
    parser.add_argument("--x", type=int, default=100)
    parser.add_argument("--y", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0；部分 libcom 版本也接受 0")
    parser.add_argument("--output_dir", default="outputs/libcom_opa_test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from libcom import get_composite_image, OPAScoreModel

    fg_rgb_path, fg_mask_path = rgba_to_rgb_and_mask(
        foreground_path=args.foreground,
        output_dir=args.output_dir,
    )

    bbox = build_bbox_from_xy_scale(
        bg_path=args.background,
        fg_path=fg_rgb_path,
        x=args.x,
        y=args.y,
        scale=args.scale,
    )

    print("=" * 80)
    print("[libcom OPA test]")
    print(f"[Input] background={args.background}")
    print(f"[Input] foreground={fg_rgb_path}")
    print(f"[Input] foreground_mask={fg_mask_path}")
    print(f"[Input] bbox={bbox}")
    print(f"[Model] OPAScoreModel device={args.device}")
    print("=" * 80)

    start = time.time()

    comp_img, comp_mask = get_composite_image(
        fg_rgb_path,
        fg_mask_path,
        args.background,
        bbox,
        option="none",
    )

    comp_path = os.path.join(args.output_dir, "opa_test_composite.png")
    comp_mask_path = os.path.join(args.output_dir, "opa_test_composite_mask.png")

    cv2.imwrite(comp_path, comp_img)
    cv2.imwrite(comp_mask_path, comp_mask)

    net = OPAScoreModel(device=args.device, model_type="SimOPA")

    score = net(comp_img, comp_mask)

    elapsed = time.time() - start

    print(f"[Output] composite={comp_path}")
    print(f"[Output] composite_mask={comp_mask_path}")
    print(f"[Output] opa_score={float(score):.6f}")
    print(f"[Time] elapsed={elapsed:.4f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()