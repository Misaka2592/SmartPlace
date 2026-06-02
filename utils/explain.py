import os
import time
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def occlude_patch(
    image: Image.Image,
    x: int,
    y: int,
    patch_size: int,
    fill_value: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    对图像指定区域进行灰色遮挡。
    """
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)

    w, h = image.size
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + patch_size)
    y2 = min(h, y + patch_size)

    draw.rectangle([x1, y1, x2, y2], fill=fill_value)

    return image


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    将热力图归一化到 0~1。
    """
    heatmap = heatmap.astype(np.float32)

    min_v = float(heatmap.min())
    max_v = float(heatmap.max())

    if max_v - min_v < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)

    return (heatmap - min_v) / (max_v - min_v)


def resize_heatmap_to_image(
    heatmap: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    将低分辨率 heatmap 放大到原图尺寸。
    """
    heatmap_norm = normalize_heatmap(heatmap)
    heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image_size, Image.BILINEAR)
    return np.array(heatmap_img).astype(np.float32) / 255.0


def apply_heatmap_overlay(
    image: Image.Image,
    heatmap_resized: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    将热力图叠加到原图上。

    注意：
    matplotlib 默认 colormap 生成 RGB，
    不需要额外指定颜色风格。
    """
    image = image.convert("RGB")
    image_arr = np.array(image).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap_resized)[:, :, :3]

    overlay = (1 - alpha) * image_arr + alpha * heatmap_color
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def generate_gradient_saliency_map(
    image: Image.Image,
    output_dir: str = "outputs/explanations",
    prefix: str = "saliency",
) -> Dict:
    """
    Generate a lightweight image-gradient saliency map.

    This is not Grad-CAM. It is a model-agnostic saliency baseline that marks
    high-contrast regions likely to influence placement calibration, especially
    object boundaries and contact areas.
    """
    ensure_dir(output_dir)
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    gy, gx = np.gradient(gray)
    saliency = np.sqrt(gx * gx + gy * gy)
    saliency = normalize_heatmap(saliency)

    overlay = apply_heatmap_overlay(image, saliency, alpha=0.40)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saliency_path = os.path.join(output_dir, f"{prefix}_gradient_saliency_{timestamp}.png")
    overlay_path = os.path.join(output_dir, f"{prefix}_gradient_overlay_{timestamp}.png")

    Image.fromarray((saliency * 255).astype(np.uint8)).save(saliency_path)
    overlay.save(overlay_path)

    return {
        "saliency_path": saliency_path,
        "overlay_path": overlay_path,
        "max_saliency": float(saliency.max()),
        "mean_saliency": float(saliency.mean()),
    }


def generate_calibration_feature_plot(
    candidate_info: Dict,
    output_dir: str = "outputs/explanations",
    prefix: str = "features",
) -> Dict:
    """
    Export a bar chart for the intermediate features used by
    SmartPlaceOPACalibratedScorer.
    """
    ensure_dir(output_dir)
    features = dict(candidate_info.get("calibration_features") or {})
    keep = ["geometry_score", "contact_score", "support_score"]
    names = [name for name in keep if name in features]
    values = [float(features[name]) for name in names]

    if not names:
        names = ["geometry_score", "contact_score", "support_score"]
        values = [0.0, 0.0, 0.0]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"{prefix}_calibration_features_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    colors = ["#2563eb", "#14b8a6", "#f59e0b"][: len(values)]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("SmartPlace calibration intermediate features")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, min(1.02, value + 0.03), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "feature_plot_path": plot_path,
        "features": {name: value for name, value in zip(names, values)},
    }


def generate_occlusion_heatmap(
    scorer,
    image: Image.Image,
    candidate_info: Dict,
    patch_size: int = 48,
    stride: int = 24,
    output_dir: str = "outputs/explanations",
    prefix: str = "occlusion",
) -> Dict:
    """
    生成遮挡实验热力图。

    思路：
    1. 先计算原图 score；
    2. 用灰色块依次遮挡图像不同区域；
    3. 每次重新调用 scorer 得到 new_score；
    4. importance = original_score - new_score；
    5. importance 越大，说明该区域被遮挡后模型分数下降越多，该区域越重要。
    """
    ensure_dir(output_dir)

    image = image.convert("RGB")
    w, h = image.size

    score_fn = getattr(scorer, "explain_score", scorer.score)
    original_score = score_fn(image, candidate_info)

    xs = list(range(0, max(1, w - patch_size + 1), stride))
    ys = list(range(0, max(1, h - patch_size + 1), stride))

    if not xs or xs[-1] != max(0, w - patch_size):
        xs.append(max(0, w - patch_size))
    if not ys or ys[-1] != max(0, h - patch_size):
        ys.append(max(0, h - patch_size))

    heatmap = np.zeros((len(ys), len(xs)), dtype=np.float32)

    records = []

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            occluded = occlude_patch(
                image=image,
                x=x,
                y=y,
                patch_size=patch_size,
            )

            new_score = score_fn(occluded, candidate_info)
            importance = original_score - new_score

            # 如果遮挡后分数上升，说明该区域可能是负贡献。
            # 热力图主要显示正贡献区域，因此这里截断到 0。
            heatmap[row, col] = max(0.0, importance)

            records.append(
                {
                    "x": x,
                    "y": y,
                    "new_score": float(new_score),
                    "importance": float(importance),
                }
            )

    heatmap_resized = resize_heatmap_to_image(heatmap, image_size=(w, h))
    overlay = apply_heatmap_overlay(image, heatmap_resized)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap_{timestamp}.png")
    overlay_path = os.path.join(output_dir, f"{prefix}_overlay_{timestamp}.png")

    heatmap_img = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
    heatmap_img.save(heatmap_path)
    overlay.save(overlay_path)

    max_importance = float(heatmap.max())
    mean_importance = float(heatmap.mean())

    explanation = build_occlusion_explanation_text(
        original_score=original_score,
        max_importance=max_importance,
        mean_importance=mean_importance,
        patch_size=patch_size,
        stride=stride,
    )

    return {
        "original_score": float(original_score),
        "heatmap": heatmap,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "records": records,
        "max_importance": max_importance,
        "mean_importance": mean_importance,
        "explanation": explanation,
    }


def build_occlusion_explanation_text(
    original_score: float,
    max_importance: float,
    mean_importance: float,
    patch_size: int,
    stride: int,
) -> str:
    """
    自动生成遮挡实验解释文字。
    """
    lines = []

    lines.append("【遮挡实验解释】")
    lines.append(f"原始候选分数：{original_score:.4f}")
    lines.append(f"遮挡块大小：{patch_size} × {patch_size}")
    lines.append(f"滑动步长：{stride}")
    lines.append(f"最大重要性分数：{max_importance:.6f}")
    lines.append(f"平均重要性分数：{mean_importance:.6f}")
    lines.append("")

    if max_importance < 1e-5:
        lines.append(
            "解释结果：当前模型对局部遮挡不敏感，热力图变化较弱。"
            "如果使用真实 OPA/FOPA 模型，通常可以得到更有区分度的关注区域。"
        )
    else:
        lines.append(
            "解释结果：热力图中高亮区域表示该区域被遮挡后模型评分下降较明显，"
            "说明模型判断较依赖这些区域。"
        )
        lines.append(
            "在物体放置任务中，可以重点观察高亮区域是否集中在前景物体、"
            "物体与背景的接触区域、地面区域或明显不合理区域。"
        )

    return "\n".join(lines)


def export_explanation_markdown(
    explanation_result: Dict,
    candidate_id: int,
    output_dir: str = "report/results",
) -> str:
    """
    导出解释模块的 Markdown 文件。
    """
    ensure_dir(output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(
        output_dir,
        f"explanation_candidate_{candidate_id}_{timestamp}.md",
    )

    lines = []

    lines.append(f"# 候选 {candidate_id} 模型解释结果\n")
    lines.append("## 1. 方法说明\n")
    lines.append(
        "本项目采用遮挡实验生成模型解释图。具体做法是使用固定大小的灰色遮挡块在图像上滑动，"
        "每遮挡一个区域，就重新调用评分模型得到新的合理性分数。"
        "如果遮挡某一区域后模型分数明显下降，说明该区域对模型判断较为重要。"
    )
    lines.append("")

    lines.append("## 2. 实验结果\n")
    lines.append(f"- 原始分数：{explanation_result['original_score']:.4f}")
    lines.append(f"- 最大重要性：{explanation_result['max_importance']:.6f}")
    lines.append(f"- 平均重要性：{explanation_result['mean_importance']:.6f}")
    lines.append(f"- 热力图路径：`{explanation_result['heatmap_path']}`")
    lines.append(f"- 叠加图路径：`{explanation_result['overlay_path']}`")
    lines.append("")

    lines.append("## 3. 自动解释\n")
    lines.append(explanation_result["explanation"])
    lines.append("")

    lines.append("## 4. 报告可用表述\n")
    lines.append(
        "遮挡实验结果表明，模型评分会受到局部图像区域的影响。"
        "在解释图中，高亮区域代表遮挡后分数下降较明显的区域，"
        "可以用于观察模型是否关注前景物体边界、接触区域以及背景语义区域。"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path
