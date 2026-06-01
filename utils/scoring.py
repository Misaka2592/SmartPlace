from typing import Dict, List


def score_to_label(score: float) -> str:
    """
    将 0~1 分数映射成三档评价。
    """
    if score >= 0.75:
        return "推荐"
    if score >= 0.45:
        return "可接受"
    return "不推荐"


def build_reason(score: float, info: Dict) -> str:
    """
    根据分数和规则生成推荐/失败原因。
    """
    reasons = []

    if info.get("out_of_bounds", False):
        reasons.append("物体存在越界")
    else:
        reasons.append("物体未越界")

    bg_w = info.get("bg_width", 1)
    bg_h = info.get("bg_height", 1)
    fg_w = info.get("fg_width", 1)
    fg_h = info.get("fg_height", 1)

    area_ratio = (fg_w * fg_h) / max(1, bg_w * bg_h)

    if area_ratio > 0.45:
        reasons.append("物体尺度偏大")
    elif area_ratio < 0.02:
        reasons.append("物体尺度偏小")
    else:
        reasons.append("物体尺度较合理")

    if score >= 0.75:
        reasons.append("模型评分较高")
    elif score >= 0.45:
        reasons.append("模型评分中等")
    else:
        reasons.append("模型评分偏低")

    return "；".join(reasons)


def rank_candidates(results: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    按 score 降序排序，返回 Top-K。
    """
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)
    return sorted_results[:top_k]


def format_score(score: float) -> str:
    return f"{score:.4f}"