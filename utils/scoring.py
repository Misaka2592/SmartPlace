from typing import Dict, List

def assign_relative_labels(results):
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)

    n = len(sorted_results)

    for rank, item in enumerate(sorted_results):
        if rank < max(1, n // 3):
            item["label"] = "推荐"
        elif rank < max(2, 2 * n // 3):
            item["label"] = "可接受"
        else:
            item["label"] = "不推荐"

    return sorted_results


def score_to_label(score: float) -> str:
    """
    将 0~1 分数映射成三档评价。
    这是基础模型小改动的核心：把模型原始输出转成用户可理解的应用标签。
    """
    if score >= 0.75:
        return "推荐"
    if score >= 0.45:
        return "可接受"
    return "不推荐"


def analyze_candidate(info: Dict, score: float) -> Dict:
    """
    对单个候选进行规则分析，生成更细致的失败提示和推荐理由。

    分析维度：
    1. 是否越界
    2. 前景尺度是否合理
    3. 是否过于靠边
    4. 是否位于图像上方
    5. 模型评分是否偏低
    """
    bg_w = info.get("bg_width", 1)
    bg_h = info.get("bg_height", 1)
    fg_w = info.get("fg_width", 1)
    fg_h = info.get("fg_height", 1)
    x = info.get("x", 0)
    y = info.get("y", 0)

    out_of_bounds = info.get("out_of_bounds", False)

    cx = x + fg_w / 2
    cy = y + fg_h / 2

    area_ratio = (fg_w * fg_h) / max(1, bg_w * bg_h)
    x_center_ratio = cx / max(1, bg_w)
    y_center_ratio = cy / max(1, bg_h)

    problems = []
    strengths = []

    # 1. 越界分析
    if out_of_bounds:
        problems.append("物体存在越界，合成结果可能不完整")
    else:
        strengths.append("物体未越界")

    # 2. 尺度分析
    if area_ratio > 0.45:
        problems.append("物体占据画面比例过大，尺度不自然")
    elif area_ratio < 0.02:
        problems.append("物体占据画面比例过小，视觉存在感不足")
    else:
        strengths.append("物体尺度较合理")

    # 3. 边缘分析
    near_left = x_center_ratio < 0.15
    near_right = x_center_ratio > 0.85
    near_top = y_center_ratio < 0.15
    near_bottom = y_center_ratio > 0.95

    if near_left or near_right:
        problems.append("物体过于靠近左右边缘")
    else:
        strengths.append("水平位置较稳定")

    if near_top:
        problems.append("物体位于画面上方，可能不符合常见地面放置关系")
    elif near_bottom:
        problems.append("物体过于靠近底部边缘")
    else:
        strengths.append("垂直位置较合理")

    # 4. 模型分数分析
    if score >= 0.75:
        strengths.append("模型评分较高")
    elif score >= 0.45:
        strengths.append("模型评分中等")
    else:
        problems.append("模型评分偏低")

    label = score_to_label(score)

    if label == "推荐":
        conclusion = "该候选位置整体较自然，适合作为优先推荐结果。"
    elif label == "可接受":
        conclusion = "该候选位置基本可用，但仍存在一定不自然因素。"
    else:
        conclusion = "该候选位置不建议使用，存在明显失败风险。"

    if problems:
        reason = "；".join(problems)
    else:
        reason = "；".join(strengths)

    return {
        "label": label,
        "area_ratio": area_ratio,
        "x_center_ratio": x_center_ratio,
        "y_center_ratio": y_center_ratio,
        "problems": problems,
        "strengths": strengths,
        "reason": reason,
        "conclusion": conclusion,
    }


def build_reason(score: float, info: Dict) -> str:
    analysis = analyze_candidate(info, score)
    return analysis["reason"]


def rank_candidates(results: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    按 score 降序排序，返回 Top-K。
    这是进阶模型改造的核心：把单图评分扩展成多候选排序推荐。
    """
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)
    return sorted_results[:top_k]


def format_score(score: float) -> str:
    return f"{score:.4f}"


def summarize_run(results: List[Dict], top_k: int) -> Dict:
    """
    对一次运行进行总体统计，方便写报告和展示。
    """
    if not results:
        return {
            "total_candidates": 0,
            "recommend_count": 0,
            "acceptable_count": 0,
            "not_recommend_count": 0,
            "best_candidate_id": None,
            "best_score": None,
            "average_score": None,
        }

    recommend_count = sum(1 for item in results if item["label"] == "推荐")
    acceptable_count = sum(1 for item in results if item["label"] == "可接受")
    not_recommend_count = sum(1 for item in results if item["label"] == "不推荐")

    best_item = max(results, key=lambda item: item["score"])
    avg_score = sum(item["score"] for item in results) / len(results)

    return {
        "total_candidates": len(results),
        "top_k": top_k,
        "recommend_count": recommend_count,
        "acceptable_count": acceptable_count,
        "not_recommend_count": not_recommend_count,
        "best_candidate_id": best_item["id"],
        "best_score": best_item["score"],
        "average_score": avg_score,
    }