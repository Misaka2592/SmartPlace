import json
import os
import time
from typing import Dict, List

import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_json(data: Dict, output_dir: str, prefix: str = "analysis") -> str:
    """
    导出结构化 JSON。
    """
    ensure_dir(output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def export_markdown_report(
    results: List[Dict],
    ranked: List[Dict],
    summary: Dict,
    model_info: Dict,
    output_dir: str,
    csv_path: str,
    log_path: str,
    explanation_path: str = None,
    explanation_report_path: str = None,
) -> str:
    """
    导出一次运行的 Markdown 分析报告。

    这个报告可以直接放到 report/results/ 或 README 中，
    作为“模型小改动、批量排序、Top-K 推荐、本地推理证据”的材料。
    """
    ensure_dir(output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"smartplace_run_report_{timestamp}.md")

    lines = []

    lines.append("# SmartPlace 单次运行分析报告\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 模型信息\n")
    lines.append(f"- 模型名称：{model_info.get('model_name')}")
    lines.append(f"- 权重路径：{model_info.get('weight_path')}")
    lines.append(f"- 运行设备：{model_info.get('device')}")
    lines.append(f"- 输入尺寸：{model_info.get('input_size')}")
    lines.append(f"- 加载状态：{model_info.get('is_loaded')}\n")

    lines.append("## 2. 本次候选统计\n")
    lines.append(f"- 候选总数：{summary.get('total_candidates')}")
    lines.append(f"- Top-K：{summary.get('top_k')}")
    lines.append(f"- 推荐数量：{summary.get('recommend_count')}")
    lines.append(f"- 可接受数量：{summary.get('acceptable_count')}")
    lines.append(f"- 不推荐数量：{summary.get('not_recommend_count')}")
    lines.append(f"- 最佳候选编号：{summary.get('best_candidate_id')}")
    lines.append(f"- 最佳候选分数：{summary.get('best_score'):.4f}")
    lines.append(f"- 平均分数：{summary.get('average_score'):.4f}\n")

    lines.append("## 3. 基础模型小改动说明\n")
    lines.append(
        "本项目将模型原始输出改造成应用友好的合理性分数和三档评价标签。"
        "原始分数经过阈值映射后，被划分为“推荐 / 可接受 / 不推荐”。"
        "该改动使模型输出能够直接服务于用户界面展示和放置位置决策。"
    )
    lines.append("")
    lines.append("当前阈值规则如下：")
    lines.append("- score >= 0.75：推荐")
    lines.append("- 0.45 <= score < 0.75：可接受")
    lines.append("- score < 0.45：不推荐\n")

    lines.append("## 4. 进阶模型改造说明\n")
    lines.append(
        "本项目进一步将单张合成图评分扩展为多候选批量评分与排序推荐模块。"
        "系统会一次性生成多个候选放置位置，对每个候选合成图进行评分，"
        "再按分数降序排序并输出 Top-K 推荐结果。"
        "该改动使评分模型从单图判断模块变成了可以支撑交互式应用的推荐模块。"
    )
    lines.append("")

    lines.append("## 5. Top-K 推荐结果\n")
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            f"### Top {rank}: 候选 {item['id']}\n"
            f"- 位置：({item['x']}, {item['y']})\n"
            f"- 缩放比例：{item['scale']}\n"
            f"- 分数：{item['score']:.4f}\n"
            f"- 评价：{item['label']}\n"
            f"- 推荐理由：{item['reason']}\n"
            f"- 结论：{item.get('conclusion', '')}\n"
        )

    lines.append("## 6. 全部候选结果表\n")
    lines.append("| 候选编号 | x | y | 分数 | 评价 | 是否越界 | 原因 |")
    lines.append("|---|---:|---:|---:|---|---|---|")

    for item in results:
        lines.append(
            f"| {item['id']} | {item['x']} | {item['y']} | "
            f"{item['score']:.4f} | {item['label']} | "
            f"{'是' if item['out_of_bounds'] else '否'} | {item['reason']} |"
        )

    lines.append("")

    lines.append("## 7. 本地推理证据\n")
    lines.append(f"- 评分表 CSV：`{csv_path}`")
    lines.append(f"- 推理日志文件：`{log_path}`")
    lines.append(
        "- 推理日志中包含模型权重路径、运行设备、输入 tensor shape、"
        "raw output、score 和 inference time，可用于课堂现场核验。"
    )

    lines.append("## 8. 模型解释结果\n")

    if explanation_path:
        lines.append(f"- 解释图路径：`{explanation_path}`")
    else:
        lines.append("- 本次运行未生成解释图。")

    if explanation_report_path:
        lines.append(f"- 解释说明文件：`{explanation_report_path}`")

    lines.append(
        "本项目采用遮挡实验作为模型解释方法。系统使用灰色遮挡块依次遮挡候选合成图的局部区域，"
        "并重新计算模型评分。若遮挡某一区域后分数明显下降，则说明该区域对模型判断更重要。"
    )
    lines.append("")


    lines.append("\n## 9. 当前版本说明\n")
    lines.append(
        "当前版本使用 DummyScorerV2 作为占位评分器。"
        "它不代表最终真实模型效果，但已经完成了真实模型接入前所需的工程接口、"
        "批量评分流程、日志格式和应用展示逻辑。"
        "后续可将 DummyScorerV2 替换为 OPA/FOPA 评分模型。"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path


def export_result_package_metadata(
    results: List[Dict],
    ranked: List[Dict],
    summary: Dict,
    model_info: Dict,
    output_dir: str,
    csv_path: str,
    log_path: str,
    report_path: str,
) -> str:
    """
    导出完整元信息 JSON。
    """
    serializable_results = []

    for item in results:
        copied = {}
        for k, v in item.items():
            if k == "image":
                continue
            copied[k] = v
        serializable_results.append(copied)

    serializable_ranked = []

    for item in ranked:
        copied = {}
        for k, v in item.items():
            if k == "image":
                continue
            copied[k] = v
        serializable_ranked.append(copied)

    data = {
        "model_info": model_info,
        "summary": summary,
        "results": serializable_results,
        "topk_results": serializable_ranked,
        "files": {
            "csv_path": csv_path,
            "log_path": log_path,
            "report_path": report_path,
        },
    }

    return export_json(data=data, output_dir=output_dir, prefix="smartplace_metadata")