import json
import os
import time
from typing import Dict, List

import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_manual_label(label: str) -> str:
    if label is None:
        return "未填写"

    label = str(label).strip()

    if label in ["推荐", "合理", "好", "成功"]:
        return "推荐"
    if label in ["可接受", "一般", "基本合理"]:
        return "可接受"
    if label in ["不推荐", "不合理", "差", "失败"]:
        return "不推荐"

    return label if label else "未填写"


def judge_consistency(model_label: str, manual_label: str) -> str:
    """
    判断模型 Top-1 标签和人工判断是否一致。
    """
    manual_label = normalize_manual_label(manual_label)

    if manual_label == "未填写":
        return "未填写人工判断"

    if model_label == manual_label:
        return "一致"

    # 宽松一致：推荐 vs 可接受，可以认为部分一致
    soft_pairs = {
        ("推荐", "可接受"),
        ("可接受", "推荐"),
        ("可接受", "不推荐"),
        ("不推荐", "可接受"),
    }

    if (model_label, manual_label) in soft_pairs:
        return "部分一致"

    return "不一致"


def build_case_record(
    case_name: str,
    background_note: str,
    foreground_note: str,
    manual_label: str,
    manual_reason: str,
    mask_info: Dict,
    summary: Dict,
    ranked: List[Dict],
    results: List[Dict],
    files: Dict,
) -> Dict:
    """
    构建单次案例记录。
    """
    case_name = case_name.strip() if case_name else "未命名案例"
    manual_label = normalize_manual_label(manual_label)

    top1 = ranked[0] if ranked else None

    if top1:
        model_top1_label = top1["label"]
        model_top1_score = float(top1["score"])
        model_top1_id = top1["id"]
        consistency = judge_consistency(model_top1_label, manual_label)
    else:
        model_top1_label = "无"
        model_top1_score = None
        model_top1_id = None
        consistency = "无模型结果"

    serializable_results = []
    for item in results:
        copied = {}
        for k, v in item.items():
            if k in ["image", "candidate_info"]:
                continue
            copied[k] = v
        serializable_results.append(copied)

    serializable_ranked = []
    for item in ranked:
        copied = {}
        for k, v in item.items():
            if k in ["image", "candidate_info"]:
                continue
            copied[k] = v
        serializable_ranked.append(copied)

    record = {
        "case_name": case_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "background_note": background_note or "",
        "foreground_note": foreground_note or "",
        "manual_label": manual_label,
        "manual_reason": manual_reason or "",
        "model_top1_id": model_top1_id,
        "model_top1_label": model_top1_label,
        "model_top1_score": model_top1_score,
        "consistency": consistency,
        "summary": summary,
        "mask_info": mask_info,
        "topk_results": serializable_ranked,
        "all_results": serializable_results,
        "files": files,
    }

    return record


def save_case_record(
    record: Dict,
    output_dir: str = "report/cases",
) -> str:
    """
    保存单次案例 JSON。
    """
    ensure_dir(output_dir)

    safe_name = record["case_name"].replace(" ", "_").replace("/", "_").replace("\\", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    path = os.path.join(output_dir, f"{timestamp}_{safe_name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return path


def load_case_records(case_dir: str = "report/cases") -> List[Dict]:
    """
    读取所有案例 JSON。
    """
    ensure_dir(case_dir)

    records = []

    for name in os.listdir(case_dir):
        if not name.endswith(".json"):
            continue

        path = os.path.join(case_dir, name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["_file_path"] = path
            records.append(record)
        except Exception:
            continue

    records.sort(key=lambda x: x.get("created_at", ""))
    return records


def summarize_case_records(records: List[Dict]) -> pd.DataFrame:
    """
    将多个案例汇总为 DataFrame。
    """
    rows = []

    for idx, record in enumerate(records, start=1):
        summary = record.get("summary", {})

        rows.append(
            {
                "序号": idx,
                "案例名称": record.get("case_name", ""),
                "创建时间": record.get("created_at", ""),
                "背景说明": record.get("background_note", ""),
                "前景说明": record.get("foreground_note", ""),
                "人工判断": record.get("manual_label", ""),
                "模型Top1候选": record.get("model_top1_id", ""),
                "模型Top1评价": record.get("model_top1_label", ""),
                "模型Top1分数": record.get("model_top1_score", ""),
                "一致性": record.get("consistency", ""),
                "候选总数": summary.get("total_candidates", ""),
                "推荐数量": summary.get("recommend_count", ""),
                "可接受数量": summary.get("acceptable_count", ""),
                "不推荐数量": summary.get("not_recommend_count", ""),
                "平均分数": summary.get("average_score", ""),
                "人工原因": record.get("manual_reason", ""),
            }
        )

    return pd.DataFrame(rows)


def export_case_summary_csv(
    records: List[Dict],
    output_dir: str = "report/results",
) -> str:
    ensure_dir(output_dir)

    df = summarize_case_records(records)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"case_summary_{timestamp}.csv")

    df.to_csv(path, index=False, encoding="utf-8-sig")

    return path


def export_case_summary_markdown(
    records: List[Dict],
    output_dir: str = "report/results",
) -> str:
    ensure_dir(output_dir)

    df = summarize_case_records(records)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"case_summary_{timestamp}.md")

    total = len(records)
    exact = sum(1 for r in records if r.get("consistency") == "一致")
    partial = sum(1 for r in records if r.get("consistency") == "部分一致")
    mismatch = sum(1 for r in records if r.get("consistency") == "不一致")

    lines = []
    lines.append("# SmartPlace 测试案例汇总报告\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 总体统计\n")
    lines.append(f"- 案例总数：{total}")
    lines.append(f"- 完全一致：{exact}")
    lines.append(f"- 部分一致：{partial}")
    lines.append(f"- 不一致：{mismatch}")

    if total > 0:
        lines.append(f"- 完全一致率：{exact / total:.2%}")
        lines.append(f"- 宽松一致率：{(exact + partial) / total:.2%}")
    else:
        lines.append("- 完全一致率：无")
        lines.append("- 宽松一致率：无")

    lines.append("\n## 2. 案例结果表\n")
    lines.append("| 序号 | 案例名称 | 人工判断 | 模型Top1评价 | 模型Top1分数 | 一致性 | 人工原因 |")
    lines.append("|---:|---|---|---|---:|---|---|")

    for _, row in df.iterrows():
        score = row["模型Top1分数"]
        if isinstance(score, float):
            score_text = f"{score:.4f}"
        else:
            score_text = str(score)

        lines.append(
            f"| {row['序号']} | {row['案例名称']} | {row['人工判断']} | "
            f"{row['模型Top1评价']} | {score_text} | {row['一致性']} | {row['人工原因']} |"
        )

    lines.append("\n## 3. 报告可用结论\n")
    lines.append(
        "通过多组背景图与前景图测试，系统能够自动生成候选位置，"
        "并输出合理性分数、三档评价和 Top-K 推荐结果。"
        "人工判断与模型判断的一致性用于评估系统推荐是否符合直观视觉常识。"
        "对于不一致案例，可进一步结合失败提示和遮挡解释图分析原因。"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path