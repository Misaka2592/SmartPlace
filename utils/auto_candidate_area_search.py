"""
auto_candidate_area_search — 自动候选区域搜索主入口

采用粗细两级搜索策略，自动寻找前景在背景图中最合理的放置区域：

1. 粗搜索：将后景图划分为 n×m 大网格，每个大网格内随机采样 r 个点作为放置中心，
   使用 DummyScorer（启发式评分器）快速评分，取平均值作为该大网格的合理度。
2. 细搜索：选出合理度最高的 determine_coeff 个大网格，将每个大网格细分为 a×b 小网格，
   使用 OPA 模型（scorer）精确评分。
3. 最终：在所有细搜索结果中按 OPA 分数统一排序，取合理度最高的位置。

参数优先级：显式传入 > config/default.yaml 中 auto_search 节 > 代码硬编码默认值
"""

import os
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image

from models.base_scorer import BaseScorer
from models.dummy_scorer import DummyScorer
from utils.composer import resize_foreground, compose_image_with_mask
from utils.logger import InferenceLogger
from utils.mask_processor import process_foreground_for_composition
from utils.scoring import analyze_candidate
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
# 内部工具：对候选列表使用指定评分器重新评分
# ---------------------------------------------------------------------------

def _rescore_candidates(
        candidates: List[Dict[str, Any]],
        background: Image.Image,
        foreground_rgba: Image.Image,
        scale: float,
        scorer: BaseScorer,
        allow_out_of_bounds: bool,
        batch_size: int,
        logger: InferenceLogger,
) -> List[Dict[str, Any]]:
    """
    对已有的候选列表重新合成并使用指定评分器评分。

    仅在回退路径中使用（粗搜索结果需要 OPA 重新评分时）。
    """
    if not candidates:
        return []

    composites: List[Image.Image] = []
    infos: List[Dict] = []

    for cand in candidates:
        composite, composite_mask, info = compose_image_with_mask(
            background=background,
            foreground=foreground_rgba,
            x=cand["x"],
            y=cand["y"],
            scale=scale,
            allow_out_of_bounds=allow_out_of_bounds,
        )
        info["composite_mask"] = composite_mask
        info["candidate_id"] = cand["id"]
        composites.append(composite)
        infos.append(info)

    all_scores: List[float] = []
    for i in range(0, len(composites), batch_size):
        batch_imgs = composites[i: i + batch_size]
        batch_infos = infos[i: i + batch_size]
        batch_scores = scorer.batch_score(batch_imgs, batch_infos)
        all_scores.extend(batch_scores)

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

    算法流程
    --------
    1. 粗搜索：将后景划分为 n×m 大网格，每个网格随机采样 r 个放置点，
       使用 DummyScorer（启发式）快速评分后取平均，
       得到每个大网格的平均合理度。
    2. 细搜索：选出平均合理度最高的 determine_coeff 个大网格，
       将每个大网格细分为 a×b 小网格，使用 OPA 模型精确评分。
    3. 最终：在所有细搜索结果中按 OPA 分数统一排序，
       取合理度最高的位置。determine_coeff > 1 时排序同样基于 OPA 分数。
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

    # 创建 dummy_scorer 仅用于粗搜索
    dummy_scorer = DummyScorer(logger=logger)

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
        "coarse_scorer": dummy_scorer.get_model_info().get("model_name", "unknown"),
        "fine_scorer": scorer.get_model_info().get("model_name", "unknown"),
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
    logger.log(f"[Param] coarse_scorer=dummy (fast), fine_scorer=OPA (precise)")
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
    # Step 3: 粗搜索 — 大网格随机采样 + DummyScorer 快速评分
    # ------------------------------------------------------------------
    logger.section("[AutoCandidateAreaSearch] Phase 1 — Coarse search (dummy scorer)")

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
        scorer=dummy_scorer,
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
    # Step 5: 细搜索 — 对每个被选中的大网格细分 a×b 小网格 + OPA 精确评分
    # ------------------------------------------------------------------
    logger.section("[AutoCandidateAreaSearch] Phase 2 — Fine search (OPA model)")

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
        # 回退：从粗搜索结果中选 top_k，用 OPA 重新评分
        coarse_sorted_by_dummy = sorted(coarse_valid, key=lambda x: x["score"], reverse=True)
        fallback_top_k = coarse_sorted_by_dummy[:max(1, determine_coeff)]
        # 用 OPA 重新评分（因为粗搜索分数是 dummy 分数）
        opa_scored = _rescore_candidates(
            candidates=fallback_top_k,
            background=background,
            foreground_rgba=foreground_rgba,
            scale=scale,
            scorer=scorer,
            allow_out_of_bounds=allow_out_of_bounds,
            batch_size=batch_size,
            logger=logger,
        )
        opa_scored_sorted = sorted(opa_scored, key=lambda x: x["score"], reverse=True)
        best = opa_scored_sorted[0] if opa_scored_sorted else None
        top_k = opa_scored_sorted[:max(1, determine_coeff)]
        result = _build_result(
            best=best,
            top_k=top_k,
            all_results=opa_scored_sorted,
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

    # 细搜索直接使用 OPA 模型评分
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

    # 按 OPA 分数统一排序 — determine_coeff 不影响此排名，所有小网格全局比较
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
            f"scale={scale}, opa_score={best['score']:.6f}"
        )
        logger.log(f"[Best] label={best['label']}, conclusion={best.get('conclusion', '')}")
    else:
        logger.log("[Best] No valid candidate found.")

    for rank, item in enumerate(top_k, 1):
        logger.log(
            f"[Top{rank}] id={item['id']} opa_score={item['score']:.6f} "
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
        "scoring_strategy": "coarse=dummy, fine=OPA",
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