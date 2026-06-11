"""
auto_search_helpers — 自动候选区域搜索的辅助函数集合

提供网格划分、随机采样、批量合成评分、候选结果构建等可复用工具，
供 auto_candidate_area_search 调用，也可独立使用。
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from models.base_scorer import BaseScorer
from utils.composer import compose_image_with_mask
from utils.scoring import analyze_candidate


# ---------------------------------------------------------------------------
# 网格划分
# ---------------------------------------------------------------------------

def compute_grid_layout(
        bg_width: int,
        bg_height: int,
        rows: int,
        cols: int,
        margin_ratio: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    将背景图按 rows×cols 划分为等分网格，返回每个网格的边界与中心信息。

    Parameters
    ----------
    bg_width, bg_height : int
        背景图宽高。
    rows, cols : int
        网格行数、列数。
    margin_ratio : float
        四周留白比例。

    Returns
    -------
    list[dict]
        每项包含: row, col, cell_id, x_start, y_start, x_end, y_end,
        cx, cy, cell_w, cell_h
    """
    margin_x = int(bg_width * margin_ratio)
    margin_y = int(bg_height * margin_ratio)

    usable_w = bg_width - 2 * margin_x
    usable_h = bg_height - 2 * margin_y
    cell_w = usable_w / max(1, cols)
    cell_h = usable_h / max(1, rows)

    cells = []
    for row in range(rows):
        for col in range(cols):
            x_start = margin_x + col * cell_w
            y_start = margin_y + row * cell_h
            cells.append({
                "row": row,
                "col": col,
                "cell_id": row * cols + col + 1,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_start + cell_w,
                "y_end": y_start + cell_h,
                "cx": x_start + cell_w / 2,
                "cy": y_start + cell_h / 2,
                "cell_w": cell_w,
                "cell_h": cell_h,
            })
    return cells


# ---------------------------------------------------------------------------
# 粗搜索：大网格内随机采样
# ---------------------------------------------------------------------------

def sample_random_points_in_cell(
        cell: Dict[str, Any],
        fg_width: int,
        fg_height: int,
        r: int,
        bg_width: int,
        bg_height: int,
        seed: Optional[int] = None,
) -> List[Dict[str, int]]:
    """
    在单个网格内随机采样 r 个前景放置中心，返回对应的左上角坐标 (x, y)。

    Parameters
    ----------
    cell : dict
        由 compute_grid_layout 返回的单个网格信息。
    fg_width, fg_height : int
        缩放后的前景尺寸。
    r : int
        每个网格的采样点数。
    bg_width, bg_height : int
        背景图宽高。
    seed : int or None
        局部随机种子，为 None 时不固定。

    Returns
    -------
    list[dict]
        每项包含 x, y（前景左上角坐标）。
    """
    rng = random.Random(seed)

    x_start = cell["x_start"]
    y_start = cell["y_start"]
    x_end = cell["x_end"]
    y_end = cell["y_end"]

    # 前景中心范围：确保中心落在网格内部即可
    center_x_min = x_start
    center_x_max = x_end
    center_y_min = y_start
    center_y_max = y_end

    # 防止前景完全超出背景
    center_x_min = max(center_x_min, fg_width / 2)
    center_x_max = min(center_x_max, bg_width - fg_width / 2)
    center_y_min = max(center_y_min, fg_height / 2)
    center_y_max = min(center_y_max, bg_height - fg_height / 2)

    if center_x_max <= center_x_min:
        center_x_min = cell["cx"] - 1
        center_x_max = cell["cx"] + 1
    if center_y_max <= center_y_min:
        center_y_min = cell["cy"] - 1
        center_y_max = cell["cy"] + 1

    points = []
    for _ in range(r):
        cx = rng.uniform(center_x_min, center_x_max)
        cy = rng.uniform(center_y_min, center_y_max)
        x = int(round(cx - fg_width / 2))
        y = int(round(cy - fg_height / 2))
        points.append({"x": x, "y": y})

    return points


def generate_coarse_candidates(
        bg_width: int,
        bg_height: int,
        fg_width: int,
        fg_height: int,
        n: int,
        m: int,
        r: int,
        margin_ratio: float = 0.08,
        seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    生成粗搜索阶段的候选放置点。

    Returns
    -------
    candidates : list[dict]
        每项包含 id, cell_row, cell_col, x, y。
    cells : list[dict]
        网格信息列表，供后续细搜索使用。
    """
    cells = compute_grid_layout(bg_width, bg_height, n, m, margin_ratio)

    rng = random.Random(seed)
    candidates = []
    cid = 1

    for cell in cells:
        # 每个网格用不同子种子，保证可复现又不会全部同点
        cell_seed = rng.randint(0, 2 ** 31 - 1) if seed is not None else None
        points = sample_random_points_in_cell(
            cell=cell,
            fg_width=fg_width,
            fg_height=fg_height,
            r=r,
            bg_width=bg_width,
            bg_height=bg_height,
            seed=cell_seed,
        )
        for pt in points:
            candidates.append({
                "id": cid,
                "cell_row": cell["row"],
                "cell_col": cell["col"],
                "x": pt["x"],
                "y": pt["y"],
            })
            cid += 1

    return candidates, cells


# ---------------------------------------------------------------------------
# 细搜索：小网格中心点
# ---------------------------------------------------------------------------

def generate_fine_candidates_for_cell(
        cell: Dict[str, Any],
        a: int,
        b: int,
        fg_width: int,
        fg_height: int,
        id_offset: int,
) -> List[Dict[str, Any]]:
    """
    在单个大网格内，细分为 a×b 小网格，每个小网格中心作为放置点。

    Parameters
    ----------
    cell : dict
        大网格信息。
    a, b : int
        细分行数、列数。
    fg_width, fg_height : int
        缩放后的前景尺寸。
    id_offset : int
        候选 id 起始偏移。

    Returns
    -------
    list[dict]
        每项包含 id, cell_row, cell_col, fine_row, fine_col, x, y。
    """
    cell_w = cell["cell_w"]
    cell_h = cell["cell_h"]
    fine_w = cell_w / max(1, b)
    fine_h = cell_h / max(1, a)

    candidates = []
    cid = id_offset + 1

    for fine_row in range(a):
        for fine_col in range(b):
            # 小网格中心
            center_x = cell["x_start"] + (fine_col + 0.5) * fine_w
            center_y = cell["y_start"] + (fine_row + 0.5) * fine_h

            x = int(round(center_x - fg_width / 2))
            y = int(round(center_y - fg_height / 2))

            candidates.append({
                "id": cid,
                "cell_row": cell["row"],
                "cell_col": cell["col"],
                "fine_row": fine_row,
                "fine_col": fine_col,
                "x": x,
                "y": y,
            })
            cid += 1

    return candidates


# ---------------------------------------------------------------------------
# 批量合成 + 评分
# ---------------------------------------------------------------------------

def compose_and_score_batch(
        background: Image.Image,
        foreground: Image.Image,
        candidates: List[Dict[str, Any]],
        scale: float,
        scorer: BaseScorer,
        allow_out_of_bounds: bool = False,
        batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    批量合成 + 分批评分，返回带完整分析的结果列表。

    Parameters
    ----------
    background : PIL.Image
        背景图（RGB）。
    foreground : PIL.Image
        前景图（RGBA）。
    candidates : list[dict]
        每项至少包含 id, x, y。
    scale : float
        前景缩放比例。
    scorer : BaseScorer
        评分器。
    allow_out_of_bounds : bool
        是否允许越界合成。
    batch_size : int
        每批评分数量上限。

    Returns
    -------
    list[dict]
        每项包含原始候选字段 + score, label, reason, conclusion, problems,
        strengths, area_ratio, x_center_ratio, y_center_ratio,
        out_of_bounds, fg_width, fg_height, candidate_info, image。
    """
    if not candidates:
        return []

    composites: List[Image.Image] = []
    infos: List[Dict] = []

    for cand in candidates:
        composite, composite_mask, info = compose_image_with_mask(
            background=background,
            foreground=foreground,
            x=cand["x"],
            y=cand["y"],
            scale=scale,
            allow_out_of_bounds=allow_out_of_bounds,
        )
        info["composite_mask"] = composite_mask
        info["candidate_id"] = cand["id"]
        cand["out_of_bounds"] = info.get("out_of_bounds", False)

        composites.append(composite)
        infos.append(info)

    # 分批评分
    all_scores: List[float] = []
    for i in range(0, len(composites), batch_size):
        batch_imgs = composites[i: i + batch_size]
        batch_infos = infos[i: i + batch_size]
        batch_scores = scorer.batch_score(batch_imgs, batch_infos)
        all_scores.extend(batch_scores)

    # 组装结果
    results = []
    for cand, composite, info, score in zip(candidates, composites, infos, all_scores):
        score = float(score)
        analysis = analyze_candidate(info, score)
        results.append({
            **cand,
            "scale": scale,
            "score": score,
            "label": analysis["label"],
            "reason": analysis["reason"],
            "conclusion": analysis["conclusion"],
            "problems": analysis["problems"],
            "strengths": analysis["strengths"],
            "area_ratio": analysis["area_ratio"],
            "x_center_ratio": analysis["x_center_ratio"],
            "y_center_ratio": analysis["y_center_ratio"],
            "out_of_bounds": info.get("out_of_bounds", False),
            "fg_width": info.get("fg_width"),
            "fg_height": info.get("fg_height"),
            "bg_width": info.get("bg_width"),
            "bg_height": info.get("bg_height"),
            "candidate_info": info,
            "image": composite,
        })

    return results


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def compute_cell_average_scores(
        results: List[Dict[str, Any]],
        n: int,
        m: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    按大网格聚合评分，计算每个网格的平均分、最高分、最低分、有效采样数。

    Returns
    -------
    dict
        key = (row, col), value = {avg_score, max_score, min_score, count, scores}
    """
    from collections import defaultdict

    cell_scores: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for r in results:
        key = (r["cell_row"], r["cell_col"])
        cell_scores[key].append(r["score"])

    cell_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in range(n):
        for col in range(m):
            key = (row, col)
            scores = cell_scores.get(key, [])
            cell_stats[key] = {
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "count": len(scores),
                "scores": scores,
            }
    return cell_stats


def select_top_cells(
        cell_stats: Dict[Tuple[int, int], Dict[str, Any]],
        determine_coeff: int,
) -> List[Tuple[int, int]]:
    """
    按平均合理度从高到低选出 determine_coeff 个大网格的 (row, col)。

    determine_coeff 为 1 时只取最高，为 2 时取最高和第二高。
    """
    sorted_cells = sorted(
        cell_stats.items(),
        key=lambda item: item[1]["avg_score"],
        reverse=True,
    )
    determine_coeff = max(1, min(determine_coeff, len(sorted_cells)))
    return [key for key, _ in sorted_cells[:determine_coeff]]