"""
image_warnings — 图像级警告检测

在评分/搜索流程中对输入图像进行预检查，生成面向用户的友好提示。
检测项：
1. 前景物过大或过小
2. 背景图纯色/近纯色/无纹理（加载画布前即拦截）
3. 自动搜索无有效候选位置
"""



import numpy as np
import os.path

from PIL import Image
from typing import Dict, List, Optional

from scipy.cluster.hierarchy import average

from utils.logger import InferenceLogger


OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ---------------------------------------------------------------------------
# 阈值常量
# ---------------------------------------------------------------------------

_FOREGROUND_AREA_RATIO_TOO_LARGE = 0.45
_FOREGROUND_AREA_RATIO_TOO_SMALL = 0.02

# 背景纯色/近纯色检测
_BG_GLOBAL_STD_THRESHOLD = 15.0     # 整图 RGB std 低于此值 → 纯色/近纯色
_BG_PATCH_STD_THRESHOLD = 10.0     # 分块后单块 std 低于此值视为均匀色块
_BG_UNIFORM_PATCH_RATIO = 0.85     # 超过此比例的块均匀 → 整图视为近纯色
_BG_PATCH_SIZE = 64                # 分块尺寸（px）


# ---------------------------------------------------------------------------
# 前景物大小检测
# ---------------------------------------------------------------------------

def check_foreground_size(
        fg_width: int,
        fg_height: int,
        bg_width: int,
        bg_height: int,
) -> Optional[str]:
    """
    检测前景物相对背景的大小是否合理。

    Returns
    -------
    str or None
        如果有警告，返回提示文字；否则返回 None。
    """
    bg_area = max(1, bg_width * bg_height)
    fg_area = fg_width * fg_height
    area_ratio = fg_area / bg_area

    if area_ratio > _FOREGROUND_AREA_RATIO_TOO_LARGE:
        return (
            f"前景物大小超过预期阈值（面积占比 {area_ratio:.1%}），"
            f"过于影响画面表现，建议缩小前景或选择更小的缩放比例。"
        )
    if area_ratio < _FOREGROUND_AREA_RATIO_TOO_SMALL:
        return (
            f"前景物大小过小（面积占比 {area_ratio:.1%}），"
            f"存在感偏弱，建议增大前景或选择更大的缩放比例。"
        )
    return None


# ---------------------------------------------------------------------------
# 背景图质量检测（纯色/近纯色）
# ---------------------------------------------------------------------------

def check_background_quality(
        background: Image.Image,
) -> Optional[str]:
    """
    检测背景图是否为纯色/近纯色/无纹理平面。

    改进的三级检测：
    1. 全局 RGB 标准差极低 → 直接判定为纯色。
    2. 分块分析：将图片分成小网格，计算每个块的 RGB std，
       若绝大部分块的 std 都极低，说明整图几乎无纹理变化 → 近纯色。
    3. 渐变检测：如果图片为缓慢渐变色（如纯色渐变），
       全局 std 可能略高但块间差异极小，仍判定为近纯色。

    Parameters
    ----------
    background : PIL.Image
        背景图（RGB）。

    Returns
    -------
    str or None
        如果背景不适合放置，返回错误提示；否则返回 None。
    """
    arr = np.asarray(background.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    logger = InferenceLogger(log_dir=LOG_DIR)

    # --- 检测 1：全局 RGB 标准差 ---
    global_std = np.max([np.std(arr[:, :, c]) for c in range(arr.shape[2])])
    logger.log(f"[DEBUG]Global RGB standard deviation: {global_std:.2f}")
    if global_std < _BG_GLOBAL_STD_THRESHOLD:
        # 进一步区分"完全纯色"和"有轻微噪声的近纯色"
        if global_std < 3.0:
            return (
                "当前背景为纯色图片，缺少可放置参考信息，无法进行放置。"
                "请选择具有清晰平面或区域的图片。"
            )
        return (
            "当前背景近似纯色（色彩变化极小），缺少可放置参考信息，无法进行放置。"
            "推荐选择具有清晰平面或区域的图片。"
        )

    # --- 检测 2：分块均匀度分析 ---
    patch_size = _BG_PATCH_SIZE
    n_rows = max(1, h // patch_size)
    n_cols = max(1, w // patch_size)
    total_patches = n_rows * n_cols

    uniform_count = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            y0 = ri * patch_size
            y1 = min(h, y0 + patch_size)
            x0 = ci * patch_size
            x1 = min(w, x0 + patch_size)
            patch = arr[y0:y1, x0:x1]
            patch_std = float(patch.std())
            if patch_std < _BG_PATCH_STD_THRESHOLD:
                uniform_count += 1

    uniform_ratio = uniform_count / max(1, total_patches)
    if uniform_ratio > _BG_UNIFORM_PATCH_RATIO:
        if uniform_ratio > 0.95:
            return (
                "当前背景缺少可放置参考信息（各区域色彩几乎无变化），无法进行放置。"
                "请选择具有清晰平面或区域的图片。"
            )
        return (
            "当前背景大部分区域缺少纹理变化，可能影响放置效果，无法进行放置。"
            "推荐选择具有更多视觉参考信息的图片。"
        )

    # --- 检测 3：渐变色检测 ---
    # 如果全局 std 不算低但四角颜色接近，说明是渐变
    corner_size = min(32, h // 4, w // 4)
    if corner_size >= 8:
        corners = [
            arr[:corner_size, :corner_size],                     # 左上
            arr[:corner_size, -corner_size:],                    # 右上
            arr[-corner_size:, :corner_size],                    # 左下
            arr[-corner_size:, -corner_size:],                   # 右下
        ]
        # 每个角取 RGB 三通道均值 → shape (3,)
        corner_means = [c.mean(axis=(0, 1)) for c in corners]            # 计算四个角平均色之间的最大欧式距离
        max_corner_diff = 0.0
        for i in range(len(corner_means)):
            for j in range(i + 1, len(corner_means)):
                diff = float(np.linalg.norm(corner_means[i] - corner_means[j]))
                max_corner_diff = max(max_corner_diff, diff)
        # 如果四个角的均值差异也很小，说明是渐变
        if max_corner_diff < 20.0 and uniform_ratio > 0.6:
            return (
                "当前背景为渐变色平面，缺少可放置参考信息，无法进行放置。"
                "推荐选择具有清晰平面或区域的图片。"
            )
    return None

# ---------------------------------------------------------------------------
# 空候选检测
# ---------------------------------------------------------------------------

def check_no_candidates(
        candidates: List[Dict],
) -> Optional[str]:
    """
    检测是否没有任何有效候选位置。
    """
    if not candidates:
        return (
            "未找到合适放置位置，请尝试其他背景或移动物体位置。"
        )
    return None


# ---------------------------------------------------------------------------
# 汇总检测（仅前景大小 + 空候选，不含背景质量）
# ---------------------------------------------------------------------------

def collect_run_warnings(
        fg_width: int,
        fg_height: int,
        bg_width: int,
        bg_height: int,
        candidates: Optional[List[Dict]] = None,
) -> List[str]:
    """
    收集评分阶段的警告（前景大小 + 空候选）。

    注意：背景质量检测在加载画布时已完成拦截，此处不再重复。

    Returns
    -------
    list[str]
        警告文字列表，无警告时为空列表。
    """
    warnings: List[str] = []

    fg_warn = check_foreground_size(fg_width, fg_height, bg_width, bg_height)
    if fg_warn:
        warnings.append(fg_warn)

    if candidates is not None:
        no_cand_warn = check_no_candidates(candidates)
        if no_cand_warn:
            warnings.append(no_cand_warn)

    return warnings