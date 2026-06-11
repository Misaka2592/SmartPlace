"""
auto_candidate_area_search — 自动候选区域搜索主入口

采用粗细两级搜索策略，自动寻找前景在背景图中最合理的放置区域：

1. 粗搜索：将后景图划分为 n×m 大网格，每个大网格内随机采样 r 个点作为放置中心，
   合成并评分，取平均值作为该大网格的合理度。
2. 细搜索：选出合理度最高的 determine_coeff 个大网格，将每个大网格细分为 a×b 小网格，
   逐格计算合理度。
3. 最终：在所有小网格中统一排序，取合理度最高的位置。

determine_coeff 仅决定粗搜索阶段选出多少个大网格进入细搜索，
不影响第二轮——第二轮是所有小网格全局比较取最优。

参数优先级：显式传入 > config/default.yaml 中 auto_search 节 > 代码硬编码默认值
"""

import os
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image

from models.base_scorer import BaseScorer
from utils.composer import resize_foreground
from utils.logger import InferenceLogger
from utils.mask_processor import process_foreground_for_composition
from utils.auto_candidate_area_search_helper import (
    compute_grid_layout,
    generate_coarse_candidates,
    generate_fine_candidates_for_cell,
    compose_and_score_batch,
    compute_cell_average_scores,
    select_top_cells,
)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")

# 代码层面的硬编码默认值（最低优先级）
_HARD_DEFAULTS = {
    "coarse_n": 4,
    "coarse_m": 4,
    "samples_per_cell": 3,
    "fine_a": 5,
    "fine_b": 5,
    "determine_coeff": 1,
    "scale": 0.25,
    "margin_ratio": 0.08,
    "allow_out_of_bounds": False,
    "filter_out_of_bounds": True,
    "batch_size": 32,
    "seed": 42,
    "mask_mode": "自动判断",
    "white_bg_threshold": 38,
}


def load_auto_search_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    从 YAML 配置文件中读取 auto_search 节的参数。

    如果配置文件不存在或没有 auto_search 节，则返回空字典，
    后续会回退到 _HARD_DEFAULTS。

    Parameters
    ----------
    config_path : str
        配置文件路径。

    Returns
    -------
    dict
        auto_search 节的配置字典。
    """
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("auto_search", {}) or {}
    except Exception:
        return {}


def _resolve(
        explicit: Any,
        config: Dict[str, Any],
        key: str,
):
    """
    参数解析优先级：显式传入（不为 None） > 配置文件值 > 硬编码默认值。
    """
    if explicit is not None:
        return explicit
    if key in config:
        return config[key]
    return _HARD_DEFAULTS.get(key)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def auto_candidate_area_search(
        background: Image.Image,
        foreground: Image.Image,
        scorer: BaseScorer,
        # 粗搜索参数
        n: Optional[int] = None,
        m: Optional[int] = None,
        r: Optional[int] = None,
        # 细搜索参数
        a: Optional[int] = None,
        b: Optional[int] = None,
        # 搜索控制
        determine_coeff: Optional[int] = None,
        scale: Optional[float] = None,
        margin_ratio: Optional[float] = None,
        allow_out_of_bounds: Optional[bool] = None,
        filter_out_of_bounds: Optional[bool] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        # 前景处理参数
        mask_mode: Optional[str] = None,
        white_bg_threshold: Optional[int] = None,
        # 配置文件路径
        config_path: str = _DEFAULT_CONFIG_PATH,
        # 日志
        logger: Optional[InferenceLogger] = None,
) -> Dict[str, Any]:
    """
    自动搜索前景在背景图中最合理的放置区域。

    参数解析优先级
    --------------
    显式传入（不为 None） > config/default.yaml 中 auto_search 节 > 代码硬编码默认值

    算法流程
    --------
    1. 粗搜索：将后景划分为 n×m 大网格，每个网格随机采样 r 个放置点，
       合成 + 评分后取平均，得到每个大网格的平均合理度。
    2. 细搜索：选出平均合理度最高的 determine_coeff 个大网格，
       将每个大网格细分为 a×b 小网格，在每个小网格中心放置并评分。
    3. 最终：在所有细搜索结果中统一排序，取合理度最高的位置。
       determine_coeff 仅影响粗搜索选出几个大网格进入细搜索，
       不影响细搜索的最终排序。

    Parameters
    ----------
    background : PIL.Image
        背景图。
    foreground : PIL.Image
        前景图。
    scorer : BaseScorer
        评分模型实例。
    n, m : int or None
        粗搜索行数、列数（大网格数量）。
    r : int or None
        每个大网格内随机采样点数。
    a, b : int or None
        细搜索行数、列数（每个大网格内的小网格数量，要求 a < n, b < m）。
    determine_coeff : int or None
        粗搜索阶段选出的最高合理度大网格数量，缺省 1。
        例如设为 2 时，会选出合理度最高和第二高的大网格进入细搜索，
        但细搜索仍然在所有小网格中统一排序取最优。
    scale : float or None
        前景缩放比例。
    margin_ratio : float or None
        边距比例。
    allow_out_of_bounds : bool or None
        是否允许越界合成。
    filter_out_of_bounds : bool or None
        是否过滤越界候选。
    batch_size : int or None
        批量评分大小。
    seed : int or None
        随机种子，设值后粗搜索采样可复现。
        注意：显式传 None 表示"不固定种子"，若要从配置文件读取，
        请不要传此参数（使用默认值）。
    mask_mode : str or None
        前景处理模式。
    white_bg_threshold : int or None
        白底去除阈值。
    config_path : str
        配置文件路径，默认 configs/default.yaml。
    logger : InferenceLogger or None
        日志记录器。

    Returns
    -------
    dict
        best : dict | None
            最优区域 {x, y, scale, score, label, ...}。
        top_k : list[dict]
            细搜索 Top-K 推荐结果。
        all_results : list[dict]
            全部细搜索评分结果，按分数降序。
        coarse_results : list[dict]
            粗搜索全部评分结果。
        coarse_cell_summary : list[dict]
            每个大网格的平均合理度摘要。
        selected_cells : list[dict]
            被选入细搜索的大网格及其平均分。
        mask_info : dict
            前景处理信息。
        search_summary : dict
            搜索统计信息。
        resolved_params : dict
            最终生效的参数（含来源：显式 / 配置 / 默认）。
    """
    if logger is None:
        logger = InferenceLogger()

    # 加载配置
    file_cfg = load_auto_search_config(config_path)

    # 解析参数：显式 > 配置文件 > 硬编码默认
    n = _resolve(n, file_cfg, "coarse_n")
    m = _resolve(m, file_cfg, "coarse_m")
    r = _resolve(r, file_cfg, "samples_per_cell")
    a = _resolve(a, file_cfg, "fine_a")
    b = _resolve(b, file_cfg, "fine_b")
    determine_coeff = _resolve(determine_coeff, file_cfg, "determine_coeff")
    scale = _resolve(scale, file_cfg, "scale")
    margin_ratio = _resolve(margin_ratio, file_cfg, "margin_ratio")
    allow_out_of_bounds = _resolve(allow_out_of_bounds, file_cfg, "allow_out_of_bounds")
    filter_out_of_bounds = _resolve(filter_out_of_bounds, file_cfg, "filter_out_of_bounds")
    batch_size = _resolve(batch_size, file_cfg, "batch_size")
    seed = _resolve(seed, file_cfg, "seed")
    mask_mode = _resolve(mask_mode, file_cfg, "mask_mode")
    white_bg_threshold = _resolve(white_bg_threshold, file_cfg, "white_bg_threshold")

    # 类型确保
    n = int(n)
    m = int(m)
    r = int(r)
    a = int(a)
    b = int(b)
    determine_coeff = int(determine_coeff)
    scale = float(scale)
    margin_ratio = float(margin_ratio)
    batch_size = int(batch_size)
    if seed is not None:
        seed = int(seed)
    white_bg_threshold = int(white_bg_threshold)

    # 记录最终生效参数
    resolved_params = {
        "n": n, "m": m, "r": r,
        "a": a, "b": b,
        "determine_coeff": determine_coeff,
        "scale": scale,
        "margin_ratio": margin_ratio,
        "allow_out_of_bounds": allow_out_of_bounds,
        "filter_out_of_bounds": filter_out_of_bounds,
        "batch_size": batch_size,
        "seed": seed,
        "mask_mode": mask_mode,
        "white_bg_threshold": white_bg_threshold,
        "config_path": config_path,
        "config_loaded": bool(file_cfg),
    }

    logger.section("[AutoCandidateAreaSearch] Start")
    logger.log(f"[Input] background_size={background.size}")
    logger.log(f"[Input] foreground_size={foreground.size}")
    logger.log(f"[Config] config_path={config_path}, config_loaded={bool(file_cfg)}")
    logger.log(f"[Param] coarse_grid={n}x{m}, samples_per_cell={r}")
    logger.log(f"[Param] fine_grid={a}x{b}, determine_coeff={determine_coeff}")
    logger.log(f"[Param] scale={scale}, margin_ratio={margin_ratio}")
    logger.log(f"[Param] filter_out_of_bounds={filter_out_of_bounds}, batch_size={batch_size}")
    logger.log(f"[Param] resolved_params={resolved_params}")

    # ------------------------------------------------------------------
    # Step 1: 前景预处理
    # ------------------------------------------------------------------
    foreground_rgba, mask_preview, mask_info = process_foreground_for_composition(
        image=foreground,
        mode=mask_mode,
        white_bg_threshold=white_bg_threshold,
    )
    logger.log(f"[Step1] mask_mode_used={mask_info.get('mode_used')}")
    logger.log(f"[Step1] foreground_output_size={mask_info.get('output_size')}")

    background = background.convert("RGB")
    bg_w, bg_h = background.size

    # ------------------------------------------------------------------
    # Step 2: 计算缩放后前景尺寸
    # ------------------------------------------------------------------
    resized_fg = resize_foreground(foreground_rgba, scale=scale, bg_width=bg_w, bg_height=bg_h)
    fg_w, fg_h = resized_fg.size
    logger.log(f"[Step2] resized_foreground_size=({fg_w}, {fg_h})")

    # ------------------------------------------------------------------
    # Step 3: 粗搜索 — 大网格随机采样 + 评分
    # ------------------------------------------------------------------
    logger.section("[AutoCandidateAreaSearch] Phase 1 — Coarse search")

    coarse_candidates, cells = generate_coarse_candidates(
        bg_width=bg_w,
        bg_height=bg_h,
        fg_width=fg_w,
        fg_height=fg_h,
        n=n,
        m=m,
        r=r,
        margin_ratio=margin_ratio,
        seed=seed,
    )
    logger.log(f"[Coarse] total_samples={len(coarse_candidates)}")

    coarse_results = compose_and_score_batch(
        background=background,
        foreground=foreground_rgba,
        candidates=coarse_candidates,
        scale=scale,
        scorer=scorer,
        allow_out_of_bounds=allow_out_of_bounds,
        batch_size=batch_size,
    )

    # 越界过滤
    if filter_out_of_bounds:
        coarse_valid = [r_ for r_ in coarse_results if not r_["out_of_bounds"]]
        logger.log(f"[Coarse] after_oob_filter: {len(coarse_valid)}/{len(coarse_results)}")
    else:
        coarse_valid = coarse_results

    # 计算每个大网格的平均合理度
    cell_stats = compute_cell_average_scores(coarse_valid, n, m)

    # 日志输出
    sorted_for_log = sorted(cell_stats.items(), key=lambda item: item[1]["avg_score"], reverse=True)
    for (row, col), stats in sorted_for_log:
        logger.log(
            f"[Coarse] Cell({row},{col}): "
            f"avg={stats['avg_score']:.6f}, max={stats['max_score']:.6f}, "
            f"min={stats['min_score']:.6f}, samples={stats['count']}"
        )

    # ------------------------------------------------------------------
    # Step 4: 选出 determine_coeff 个大网格进入细搜索
    # ------------------------------------------------------------------
    selected_keys = select_top_cells(cell_stats, determine_coeff)
    logger.log(f"[Coarse] Selected top {len(selected_keys)} cells for fine search: {selected_keys}")

    # ------------------------------------------------------------------
    # Step 5: 细搜索 — 对每个被选中的大网格细分 a×b 小网格
    # ------------------------------------------------------------------
    logger.section("[AutoCandidateAreaSearch] Phase 2 — Fine search")

    # 找到选中网格对应的 cell 信息
    cell_lookup = {(c["row"], c["col"]): c for c in cells}
    fine_candidates_all: List[Dict[str, Any]] = []
    id_offset = 0

    for key in selected_keys:
        cell = cell_lookup.get(key)
        if cell is None:
            continue
        fine_cands = generate_fine_candidates_for_cell(
            cell=cell,
            a=a,
            b=b,
            fg_width=fg_w,
            fg_height=fg_h,
            id_offset=id_offset,
        )
        fine_candidates_all.extend(fine_cands)
        id_offset += len(fine_cands)

    logger.log(f"[Fine] total_fine_candidates={len(fine_candidates_all)}")

    if not fine_candidates_all:
        logger.log("[Fine] WARNING: No fine candidates generated, falling back to coarse best")
        coarse_sorted = sorted(coarse_valid, key=lambda x: x["score"], reverse=True)
        best = coarse_sorted[0] if coarse_sorted else None
        result = _build_result(
            best=best,
            top_k=coarse_sorted[:max(1, determine_coeff)],
            all_results=coarse_sorted,
            coarse_results=coarse_results,
            cell_stats=cell_stats,
            selected_keys=selected_keys,
            mask_info=mask_info,
            n=n, m=m, r=r, a=a, b=b,
            determine_coeff=determine_coeff,
            coarse_count=len(coarse_candidates),
            fine_count=0,
            fallback=True,
        )
        result["resolved_params"] = resolved_params
        return result

    fine_results = compose_and_score_batch(
        background=background,
        foreground=foreground_rgba,
        candidates=fine_candidates_all,
        scale=scale,
        scorer=scorer,
        allow_out_of_bounds=allow_out_of_bounds,
        batch_size=batch_size,
    )

    # 越界过滤
    if filter_out_of_bounds:
        fine_valid = [r_ for r_ in fine_results if not r_["out_of_bounds"]]
        logger.log(f"[Fine] after_oob_filter: {len(fine_valid)}/{len(fine_results)}")
    else:
        fine_valid = fine_results

    # 统一排序 — determine_coeff 不影响此排名，所有小网格全局比较
    all_results_sorted = sorted(fine_valid, key=lambda x: x["score"], reverse=True)

    if not all_results_sorted:
        all_results_sorted = sorted(fine_results, key=lambda x: x["score"], reverse=True)
        logger.log("[Fine] WARNING: all fine results were out-of-bounds, using unfiltered results")

    best = all_results_sorted[0] if all_results_sorted else None
    top_k = all_results_sorted[:max(1, determine_coeff)]

    # ------------------------------------------------------------------
    # Step 6: 日志输出与结果组装
    # ------------------------------------------------------------------
    logger.section("[AutoCandidateAreaSearch] Result")
    if best:
        logger.log(
            f"[Best] x={best['x']}, y={best['y']}, "
            f"scale={scale}, score={best['score']:.6f}"
        )
        logger.log(f"[Best] label={best['label']}, conclusion={best.get('conclusion', '')}")
    else:
        logger.log("[Best] No valid candidate found.")

    for rank, item in enumerate(top_k, 1):
        logger.log(
            f"[Top{rank}] id={item['id']} score={item['score']:.6f} "
            f"pos=({item['x']},{item['y']}) label={item['label']}"
        )

    search_summary = {
        "coarse_grid": f"{n}x{m}",
        "coarse_samples_per_cell": r,
        "total_coarse_samples": len(coarse_candidates),
        "fine_grid": f"{a}x{b}",
        "determine_coeff": determine_coeff,
        "selected_cells_count": len(selected_keys),
        "total_fine_candidates": len(fine_candidates_all),
        "total_fine_scored": len(fine_valid) if filter_out_of_bounds else len(fine_results),
        "best_score": best["score"] if best else None,
        "best_position": {"x": best["x"], "y": best["y"]} if best else None,
    }
    logger.log(f"[Summary] {search_summary}")

    result = _build_result(
        best=best,
        top_k=top_k,
        all_results=all_results_sorted,
        coarse_results=coarse_results,
        cell_stats=cell_stats,
        selected_keys=selected_keys,
        mask_info=mask_info,
        n=n, m=m, r=r, a=a, b=b,
        determine_coeff=determine_coeff,
        coarse_count=len(coarse_candidates),
        fine_count=len(fine_candidates_all),
        fallback=False,
    )
    result["resolved_params"] = resolved_params
    return result


# ---------------------------------------------------------------------------
# 内部结果组装
# ---------------------------------------------------------------------------

def _build_result(
        best: Optional[Dict[str, Any]],
        top_k: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]],
        coarse_results: List[Dict[str, Any]],
        cell_stats: Dict,
        selected_keys: List,
        mask_info: Dict,
        n: int,
        m: int,
        r: int,
        a: int,
        b: int,
        determine_coeff: int,
        coarse_count: int,
        fine_count: int,
        fallback: bool,
) -> Dict[str, Any]:
    """统一构建返回字典。"""
    # 粗搜索网格摘要
    cell_summary = []
    for (row, col), stats in sorted(cell_stats.items()):
        cell_summary.append({
            "cell_row": row,
            "cell_col": col,
            "avg_score": stats["avg_score"],
            "max_score": stats["max_score"],
            "min_score": stats["min_score"],
            "valid_sample_count": stats["count"],
            "selected": (row, col) in selected_keys,
        })

    # 被选中进入细搜索的大网格
    selected_cells = []
    for key in selected_keys:
        stats = cell_stats.get(key, {})
        selected_cells.append({
            "cell_row": key[0],
            "cell_col": key[1],
            "avg_score": stats.get("avg_score", 0.0),
        })

    search_summary = {
        "coarse_grid": f"{n}x{m}",
        "coarse_samples_per_cell": r,
        "total_coarse_samples": coarse_count,
        "fine_grid": f"{a}x{b}",
        "determine_coeff": determine_coeff,
        "selected_cells_count": len(selected_keys),
        "total_fine_candidates": fine_count,
        "best_score": best["score"] if best else None,
        "best_position": {"x": best["x"], "y": best["y"]} if best else None,
        "fallback": fallback,
    }

    return {
        "best": best,
        "top_k": top_k,
        "all_results": all_results,
        "coarse_results": coarse_results,
        "coarse_cell_summary": cell_summary,
        "selected_cells": selected_cells,
        "mask_info": mask_info,
        "search_summary": search_summary,
    }