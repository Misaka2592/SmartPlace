import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def has_useful_alpha(image: Image.Image) -> bool:
    """
    判断图片是否有有效 alpha 通道。
    如果 alpha 全是 255，说明虽然是 RGBA，但并没有真正透明区域。
    """
    if image.mode != "RGBA":
        return False

    alpha = np.array(image.getchannel("A"))
    return alpha.min() < 250


def build_mask_preview(mask: np.ndarray) -> Image.Image:
    """
    mask: 0~255 uint8
    返回可视化灰度 mask。
    """
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L").convert("RGB")


def estimate_background_color_from_edges(
    rgb: np.ndarray,
    edge_sample_ratio: float = 0.08,
) -> np.ndarray:
    """
    从图像边缘估计背景颜色。
    适用于白底、浅色底、纯色底前景图。
    """
    h, w = rgb.shape[:2]

    edge_h = max(1, int(h * edge_sample_ratio))
    edge_w = max(1, int(w * edge_sample_ratio))

    top = rgb[:edge_h, :, :]
    bottom = rgb[-edge_h:, :, :]
    left = rgb[:, :edge_w, :]
    right = rgb[:, -edge_w:, :]

    edge_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    bg_color = np.median(edge_pixels, axis=0)
    return bg_color.astype(np.float32)


def remove_light_or_solid_background(
    image: Image.Image,
    threshold: int = 38,
    edge_sample_ratio: float = 0.08,
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    根据边缘背景颜色估计生成 alpha mask。

    思路：
    1. 假设图像边缘多数属于背景；
    2. 估计边缘背景颜色；
    3. 计算每个像素与背景颜色的距离；
    4. 距离大于阈值的区域视为前景；
    5. 形态学处理 + 高斯平滑，得到更自然的 alpha。
    """
    image = image.convert("RGB")
    rgb = np.array(image).astype(np.float32)

    bg_color = estimate_background_color_from_edges(
        rgb=rgb,
        edge_sample_ratio=edge_sample_ratio,
    )

    diff = np.linalg.norm(rgb - bg_color.reshape(1, 1, 3), axis=2)

    raw_mask = (diff > threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    # 去掉小噪点，填补小孔洞
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 轻微膨胀，避免边缘被吃掉
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 边缘平滑，alpha 更自然
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    rgba = image.convert("RGBA")
    rgba_arr = np.array(rgba)
    rgba_arr[:, :, 3] = mask

    foreground_rgba = Image.fromarray(rgba_arr, mode="RGBA")
    mask_preview = build_mask_preview(mask)

    info = {
        "mode_used": "白底/浅色背景去除",
        "has_alpha": False,
        "estimated_background_color": [
            float(bg_color[0]),
            float(bg_color[1]),
            float(bg_color[2]),
        ],
        "threshold": threshold,
        "edge_sample_ratio": edge_sample_ratio,
        "foreground_pixel_ratio": float((mask > 20).mean()),
    }

    return foreground_rgba, mask_preview, info


def use_alpha_channel(image: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    使用透明 PNG 自带 alpha 通道。
    """
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A")).astype(np.uint8)

    mask_preview = build_mask_preview(alpha)

    info = {
        "mode_used": "透明 PNG Alpha",
        "has_alpha": True,
        "foreground_pixel_ratio": float((alpha > 20).mean()),
    }

    return rgba, mask_preview, info


def use_full_image_as_foreground(image: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    不做抠图，整张图片作为前景。
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    arr[:, :, 3] = 255

    rgba = Image.fromarray(arr, mode="RGBA")
    mask = np.ones(arr.shape[:2], dtype=np.uint8) * 255
    mask_preview = build_mask_preview(mask)

    info = {
        "mode_used": "不处理，整图作为前景",
        "has_alpha": False,
        "foreground_pixel_ratio": 1.0,
    }

    return rgba, mask_preview, info


def process_foreground_for_composition(
    image: Image.Image,
    mode: str = "自动判断",
    white_bg_threshold: int = 38,
    edge_sample_ratio: float = 0.08,
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    前景处理统一入口。

    mode 可选：
    1. 自动判断
    2. 透明 PNG Alpha
    3. 白底/浅色背景去除
    4. 不处理
    """
    image = image.convert("RGBA")

    if mode == "透明 PNG Alpha":
        foreground_rgba, mask_preview, info = use_alpha_channel(image)

    elif mode == "白底/浅色背景去除":
        foreground_rgba, mask_preview, info = remove_light_or_solid_background(
            image=image,
            threshold=white_bg_threshold,
            edge_sample_ratio=edge_sample_ratio,
        )

    elif mode == "不处理":
        foreground_rgba, mask_preview, info = use_full_image_as_foreground(image)

    else:
        # 自动判断：有真实 alpha 就用 alpha，否则尝试白底去除
        if has_useful_alpha(image):
            foreground_rgba, mask_preview, info = use_alpha_channel(image)
            info["auto_decision"] = "detected_useful_alpha"
        else:
            foreground_rgba, mask_preview, info = remove_light_or_solid_background(
                image=image,
                threshold=white_bg_threshold,
                edge_sample_ratio=edge_sample_ratio,
            )
            info["auto_decision"] = "no_alpha_use_light_bg_removal"

    info["requested_mode"] = mode
    info["input_size"] = image.size
    info["output_size"] = foreground_rgba.size

    return foreground_rgba, mask_preview, info


def save_processed_foreground(
    foreground_rgba: Image.Image,
    mask_preview: Image.Image,
    output_dir: str = "outputs/masks",
) -> Tuple[str, str]:
    """
    保存处理后的前景 RGBA 和 mask 预览。
    """
    ensure_dir(output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    fg_path = os.path.join(output_dir, f"processed_foreground_{timestamp}.png")
    mask_path = os.path.join(output_dir, f"foreground_mask_{timestamp}.png")

    foreground_rgba.save(fg_path)
    mask_preview.save(mask_path)

    return fg_path, mask_path