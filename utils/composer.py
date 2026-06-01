from typing import Dict, Tuple
from PIL import Image


def resize_foreground(
    foreground: Image.Image,
    scale: float,
    bg_width: int,
    bg_height: int,
) -> Image.Image:
    """
    根据背景短边和 scale 缩放前景。
    scale 表示前景宽度约占背景宽度的比例。
    """
    foreground = foreground.convert("RGBA")

    fg_w, fg_h = foreground.size

    target_w = int(bg_width * scale)
    target_w = max(8, target_w)

    ratio = target_w / fg_w
    target_h = int(fg_h * ratio)
    target_h = max(8, target_h)

    # 防止前景过大
    max_w = int(bg_width * 0.9)
    max_h = int(bg_height * 0.9)

    if target_w > max_w:
        ratio = max_w / target_w
        target_w = max_w
        target_h = int(target_h * ratio)

    if target_h > max_h:
        ratio = max_h / target_h
        target_h = max_h
        target_w = int(target_w * ratio)

    return foreground.resize((target_w, target_h), Image.LANCZOS)


def check_out_of_bounds(
    x: int,
    y: int,
    fg_width: int,
    fg_height: int,
    bg_width: int,
    bg_height: int,
) -> bool:
    """
    检查前景是否越界。
    """
    if x < 0 or y < 0:
        return True
    if x + fg_width > bg_width:
        return True
    if y + fg_height > bg_height:
        return True
    return False


def compose_image(
    background: Image.Image,
    foreground: Image.Image,
    x: int,
    y: int,
    scale: float = 0.4,
    allow_out_of_bounds: bool = False,
) -> Tuple[Image.Image, Dict]:
    """
    将 foreground 按 scale 缩放后粘贴到 background 的 (x, y) 位置。

    参数:
        background: PIL Image，背景图
        foreground: PIL Image，前景图，建议 RGBA
        x, y: 前景左上角坐标
        scale: 前景缩放比例，表示前景宽度约占背景宽度比例
        allow_out_of_bounds: 是否允许越界

    返回:
        composite: 合成图
        info: 合成信息
    """
    background = background.convert("RGB")
    foreground = foreground.convert("RGBA")

    bg_w, bg_h = background.size

    resized_fg = resize_foreground(
        foreground=foreground,
        scale=scale,
        bg_width=bg_w,
        bg_height=bg_h,
    )

    fg_w, fg_h = resized_fg.size

    out_of_bounds = check_out_of_bounds(
        x=x,
        y=y,
        fg_width=fg_w,
        fg_height=fg_h,
        bg_width=bg_w,
        bg_height=bg_h,
    )

    composite = background.convert("RGBA")

    if out_of_bounds and not allow_out_of_bounds:
        info = {
            "x": x,
            "y": y,
            "fg_width": fg_w,
            "fg_height": fg_h,
            "bg_width": bg_w,
            "bg_height": bg_h,
            "out_of_bounds": True,
            "composed": False,
        }
        return composite.convert("RGB"), info

    # PIL paste 支持部分越界，不会报错
    composite.alpha_composite(resized_fg, dest=(x, y))

    info = {
        "x": x,
        "y": y,
        "fg_width": fg_w,
        "fg_height": fg_h,
        "bg_width": bg_w,
        "bg_height": bg_h,
        "out_of_bounds": out_of_bounds,
        "composed": True,
    }

    return composite.convert("RGB"), info