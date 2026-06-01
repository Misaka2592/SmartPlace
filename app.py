import os
import time
from typing import Dict, List

import gradio as gr
import pandas as pd
import yaml
from PIL import Image

from utils.composer import compose_image, resize_foreground
from utils.candidate_generator import generate_grid_candidates
from utils.scoring import (
    score_to_label,
    rank_candidates,
    format_score,
    analyze_candidate,
    summarize_run,
)
from utils.logger import InferenceLogger
from utils.exporter import (
    export_markdown_report,
    export_result_package_metadata,
)
from utils.explain import (
    generate_occlusion_heatmap,
    export_explanation_markdown,
)
from utils.mask_processor import (
    process_foreground_for_composition,
    save_processed_foreground,
)
from models.dummy_scorer import DummyScorer


OUTPUT_DIR = "outputs"
COMPOSITE_DIR = os.path.join(OUTPUT_DIR, "composites")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explanations")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
REPORT_RESULT_DIR = os.path.join("report", "results")
CONFIG_PATH = "configs/default.yaml"

os.makedirs(COMPOSITE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(REPORT_RESULT_DIR, exist_ok=True)


def load_config(config_path: str = CONFIG_PATH) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


cfg = load_config()

logger = InferenceLogger(
    log_dir=LOG_DIR,
    enable_file_log=cfg.get("output", {}).get("save_log", True),
)

scorer_cfg = cfg.get("scorer", {})

scorer = DummyScorer(
    weight_path=scorer_cfg.get("weight_path", "weights/dummy_scorer_rule_based.pth"),
    device=scorer_cfg.get("device", "cpu"),
    input_size=scorer_cfg.get("input_size", 224),
    logger=logger,
)


def save_candidate_images(results: List[dict]) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for item in results:
        image = item["image"]
        cid = item["id"]
        score = item["score"]
        filename = f"{timestamp}_candidate_{cid}_score_{score:.4f}.png"
        path = os.path.join(COMPOSITE_DIR, filename)
        image.save(path)
        item["saved_path"] = path


def build_model_info_text() -> str:
    info = scorer.get_model_info()

    lines = [
        f"模型名称：{info['model_name']}",
        f"权重路径：{info['weight_path']}",
        f"运行设备：{info['device']}",
        f"输入尺寸：{info['input_size']}",
        f"加载状态：{'已加载' if info['is_loaded'] else '未加载'}",
        "",
        "说明：当前为 DummyScorerV4，占位规则模型。它用于模拟真实模型推理接口、日志和遮挡解释，后续可替换为 OPA/FOPA。",
    ]

    return "\n".join(lines)


def build_run_analysis_text(
    summary: Dict,
    ranked: List[Dict],
    mask_info: Dict,
    explanation_text: str = "",
) -> str:
    lines = []

    lines.append("【本次运行统计】")
    lines.append(f"候选总数：{summary['total_candidates']}")
    lines.append(f"推荐数量：{summary['recommend_count']}")
    lines.append(f"可接受数量：{summary['acceptable_count']}")
    lines.append(f"不推荐数量：{summary['not_recommend_count']}")
    lines.append(f"最佳候选编号：{summary['best_candidate_id']}")
    lines.append(f"最佳候选分数：{summary['best_score']:.4f}")
    lines.append(f"平均分数：{summary['average_score']:.4f}")

    lines.append("")
    lines.append("【前景处理说明】")
    lines.append(f"请求模式：{mask_info.get('requested_mode')}")
    lines.append(f"实际使用：{mask_info.get('mode_used')}")
    lines.append(f"输入尺寸：{mask_info.get('input_size')}")
    lines.append(f"输出尺寸：{mask_info.get('output_size')}")
    lines.append(f"前景像素占比：{mask_info.get('foreground_pixel_ratio', 0):.4f}")

    if "auto_decision" in mask_info:
        lines.append(f"自动判断结果：{mask_info.get('auto_decision')}")

    if "estimated_background_color" in mask_info:
        bg_color = mask_info.get("estimated_background_color")
        lines.append(f"估计背景颜色：{bg_color}")

    lines.append("")
    lines.append("【Top-K 推荐解释】")

    for rank, item in enumerate(ranked, start=1):
        lines.append(
            f"Top {rank}：候选 {item['id']}，"
            f"位置=({item['x']}, {item['y']})，"
            f"分数={item['score']:.4f}，"
            f"评价={item['label']}。"
        )
        lines.append(f"理由：{item['reason']}")
        lines.append(f"结论：{item.get('conclusion', '')}")
        lines.append("")

    lines.append("【模型改造说明】")
    lines.append(
        "基础模型小改动：将模型原始输出转换为 0~1 合理性分数，并映射为“推荐 / 可接受 / 不推荐”三档评价。"
    )
    lines.append(
        "进阶模型改造：将单图评分扩展为多候选批量评分、降序排序和 Top-K 推荐。"
    )

    lines.append("")
    lines.append("【多工具串联说明】")
    lines.append(
        "当前系统流程为：前景 mask 处理工具 → 候选位置生成 → 图像合成 → 评分模型 → 遮挡解释模块 → Web 展示与结果导出。"
    )

    if explanation_text:
        lines.append("")
        lines.append(explanation_text)

    return "\n".join(lines)


def run_smartplace(
    background_image,
    foreground_image,
    mask_mode,
    white_bg_threshold,
    candidate_count,
    scale,
    top_k,
    filter_out_of_bounds,
    enable_explanation,
    occlusion_patch_size,
    occlusion_stride,
):
    if background_image is None:
        raise gr.Error("请上传背景图。")

    if foreground_image is None:
        raise gr.Error("请上传前景图。透明 PNG 效果最好，白底图片也可以尝试自动去底。")

    background = Image.fromarray(background_image).convert("RGB")
    raw_foreground = Image.fromarray(foreground_image).convert("RGBA")

    mask_cfg = cfg.get("mask_processor", {})

    foreground, mask_preview, mask_info = process_foreground_for_composition(
        image=raw_foreground,
        mode=mask_mode,
        white_bg_threshold=int(white_bg_threshold),
        edge_sample_ratio=float(mask_cfg.get("edge_sample_ratio", 0.08)),
    )

    processed_fg_path = None
    mask_path = None

    if cfg.get("output", {}).get("save_mask", True):
        processed_fg_path, mask_path = save_processed_foreground(
            foreground_rgba=foreground,
            mask_preview=mask_preview,
            output_dir=MASK_DIR,
        )

    bg_w, bg_h = background.size

    resized_fg = resize_foreground(
        foreground=foreground,
        scale=float(scale),
        bg_width=bg_w,
        bg_height=bg_h,
    )
    fg_w, fg_h = resized_fg.size

    candidates = generate_grid_candidates(
        bg_width=bg_w,
        bg_height=bg_h,
        fg_width=fg_w,
        fg_height=fg_h,
        candidate_count=int(candidate_count),
    )

    logger.section("[SmartPlace] Start one demo inference")
    logger.log(f"[Input] background_size={background.size}")
    logger.log(f"[Input] raw_foreground_size={raw_foreground.size}")
    logger.log(f"[Mask] requested_mode={mask_info.get('requested_mode')}")
    logger.log(f"[Mask] mode_used={mask_info.get('mode_used')}")
    logger.log(f"[Mask] foreground_pixel_ratio={mask_info.get('foreground_pixel_ratio')}")
    logger.log(f"[Mask] processed_foreground_path={processed_fg_path}")
    logger.log(f"[Mask] mask_path={mask_path}")
    logger.log(f"[Input] resized_foreground_size={resized_fg.size}")
    logger.log(f"[Param] candidate_count={candidate_count}")
    logger.log(f"[Param] scale={scale}")
    logger.log(f"[Param] top_k={top_k}")
    logger.log(f"[Param] filter_out_of_bounds={filter_out_of_bounds}")
    logger.log(f"[Param] enable_explanation={enable_explanation}")
    logger.log(f"[Param] occlusion_patch_size={occlusion_patch_size}")
    logger.log(f"[Param] occlusion_stride={occlusion_stride}")

    composites = []
    candidate_infos = []

    for cand in candidates:
        composite, info = compose_image(
            background=background,
            foreground=foreground,
            x=cand["x"],
            y=cand["y"],
            scale=float(scale),
            allow_out_of_bounds=not bool(filter_out_of_bounds),
        )

        info["candidate_id"] = cand["id"]

        composites.append(composite)
        candidate_infos.append(info)

    scores = scorer.batch_score(composites, candidate_infos)

    results = []

    for cand, composite, info, score in zip(candidates, composites, candidate_infos, scores):
        label = score_to_label(score)
        analysis = analyze_candidate(info, score)
        reason = analysis["reason"]

        result = {
            "id": cand["id"],
            "x": info["x"],
            "y": info["y"],
            "scale": float(scale),
            "score": score,
            "label": label,
            "reason": reason,
            "conclusion": analysis["conclusion"],
            "problems": analysis["problems"],
            "strengths": analysis["strengths"],
            "area_ratio": analysis["area_ratio"],
            "x_center_ratio": analysis["x_center_ratio"],
            "y_center_ratio": analysis["y_center_ratio"],
            "out_of_bounds": info["out_of_bounds"],
            "fg_width": info["fg_width"],
            "fg_height": info["fg_height"],
            "candidate_info": info,
            "image": composite,
        }

        results.append(result)

    ranked = rank_candidates(results, top_k=int(top_k))
    summary = summarize_run(results, top_k=int(top_k))

    if cfg.get("output", {}).get("save_images", True):
        save_candidate_images(results)

    explanation_gallery = []
    explanation_text = ""
    explanation_overlay_path = None
    explanation_report_path = None

    if enable_explanation and ranked:
        top1 = ranked[0]

        logger.section("[SmartPlace] Start explanation for Top-1 candidate")
        logger.log(f"[Explain] candidate_id={top1['id']}")

        explanation_result = generate_occlusion_heatmap(
            scorer=scorer,
            image=top1["image"],
            candidate_info=top1["candidate_info"],
            patch_size=int(occlusion_patch_size),
            stride=int(occlusion_stride),
            output_dir=EXPLAIN_DIR,
            prefix=f"candidate_{top1['id']}",
        )

        explanation_overlay_path = explanation_result["overlay_path"]

        explanation_report_path = export_explanation_markdown(
            explanation_result=explanation_result,
            candidate_id=top1["id"],
            output_dir=REPORT_RESULT_DIR,
        )

        explanation_text = explanation_result["explanation"]

        explanation_gallery.append(
            (
                explanation_result["overlay_path"],
                f"候选 {top1['id']} 遮挡实验热力图叠加结果",
            )
        )

        logger.log(f"[Explain] heatmap_path={explanation_result['heatmap_path']}")
        logger.log(f"[Explain] overlay_path={explanation_result['overlay_path']}")
        logger.log(f"[Explain] report_path={explanation_report_path}")

    table_rows = []

    for item in results:
        table_rows.append(
            {
                "候选编号": item["id"],
                "x": item["x"],
                "y": item["y"],
                "缩放比例": item["scale"],
                "分数": format_score(item["score"]),
                "评价": item["label"],
                "是否越界": "是" if item["out_of_bounds"] else "否",
                "面积占比": f"{item['area_ratio']:.4f}",
                "推荐理由/失败提示": item["reason"],
                "结论": item["conclusion"],
            }
        )

    df = pd.DataFrame(table_rows)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(TABLE_DIR, f"{timestamp}_scores.csv")

    if cfg.get("output", {}).get("save_csv", True):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    gallery_items = []

    for item in results:
        caption = (
            f"候选 {item['id']} | "
            f"score={item['score']:.4f} | "
            f"{item['label']}"
        )
        gallery_items.append((item["image"], caption))

    topk_gallery = []

    for rank, item in enumerate(ranked, start=1):
        caption = (
            f"Top {rank} - 候选 {item['id']} | "
            f"score={item['score']:.4f} | "
            f"{item['label']}"
        )
        topk_gallery.append((item["image"], caption))

    run_analysis_text = build_run_analysis_text(
        summary=summary,
        ranked=ranked,
        mask_info=mask_info,
        explanation_text=explanation_text,
    )

    model_info = scorer.get_model_info()
    log_path = logger.get_log_path()

    report_path = export_markdown_report(
        results=results,
        ranked=ranked,
        summary=summary,
        model_info=model_info,
        output_dir=REPORT_RESULT_DIR,
        csv_path=csv_path,
        log_path=log_path,
        explanation_path=explanation_overlay_path,
        explanation_report_path=explanation_report_path,
    )

    metadata_path = export_result_package_metadata(
        results=results,
        ranked=ranked,
        summary=summary,
        model_info=model_info,
        output_dir=REPORT_RESULT_DIR,
        csv_path=csv_path,
        log_path=log_path,
        report_path=report_path,
    )

    logger.log("[SmartPlace] Inference finished.")
    logger.log(f"[Output] processed_foreground={processed_fg_path}")
    logger.log(f"[Output] mask_preview={mask_path}")
    logger.log(f"[Output] score_table_saved={csv_path}")
    logger.log(f"[Output] report_saved={report_path}")
    logger.log(f"[Output] metadata_saved={metadata_path}")
    logger.log(f"[Output] log_file={log_path}")

    return (
        mask_preview,
        foreground,
        gallery_items,
        topk_gallery,
        df,
        run_analysis_text,
        explanation_gallery,
        processed_fg_path,
        mask_path,
        csv_path,
        log_path,
        report_path,
        metadata_path,
        explanation_report_path,
    )


with gr.Blocks(title="SmartPlace 智能物体放置推荐系统") as demo:
    gr.Markdown(
        """
        # SmartPlace：智能物体放置与合成图质量评价系统

        当前版本：v0.5 前景 mask / 多工具串联版本。

        本版本新增：
        - 透明 PNG alpha mask 读取
        - 白底 / 浅色背景自动去除
        - 前景 mask 预览
        - 处理后 RGBA 前景导出
        - 前景处理工具 → 候选生成 → 合成 → 评分 → 解释 → 展示 的多工具串联流程
        """
    )

    with gr.Row():
        with gr.Column():
            background_input = gr.Image(
                label="背景图",
                type="numpy",
            )
            foreground_input = gr.Image(
                label="前景图，透明 PNG 最佳；白底图也可尝试自动去底",
                type="numpy",
            )

            mask_mode_input = gr.Radio(
                choices=[
                    "自动判断",
                    "透明 PNG Alpha",
                    "白底/浅色背景去除",
                    "不处理",
                ],
                value=cfg.get("mask_processor", {}).get("default_mode", "自动判断"),
                label="前景处理模式",
            )

            white_bg_threshold_input = gr.Slider(
                minimum=10,
                maximum=100,
                value=cfg.get("mask_processor", {}).get("white_bg_threshold", 38),
                step=2,
                label="白底/浅色背景去除阈值",
            )

        with gr.Column():
            model_info = gr.Textbox(
                label="当前模型信息",
                value=build_model_info_text(),
                lines=8,
                interactive=False,
            )

            candidate_count_input = gr.Slider(
                minimum=4,
                maximum=16,
                value=9,
                step=1,
                label="候选数量",
            )

            scale_input = gr.Slider(
                minimum=0.1,
                maximum=0.8,
                value=0.35,
                step=0.05,
                label="前景缩放比例",
            )

            top_k_input = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Top-K 推荐数量",
            )

            filter_out_of_bounds_input = gr.Checkbox(
                value=True,
                label="过滤明显越界候选",
            )

            enable_explanation_input = gr.Checkbox(
                value=True,
                label="生成模型解释图",
            )

            occlusion_patch_size_input = gr.Slider(
                minimum=24,
                maximum=96,
                value=48,
                step=8,
                label="遮挡块大小",
            )

            occlusion_stride_input = gr.Slider(
                minimum=16,
                maximum=64,
                value=32,
                step=8,
                label="遮挡滑动步长",
            )

            run_button = gr.Button(
                value="处理前景、生成候选、评分并解释",
                variant="primary",
            )

    gr.Markdown("## 前景 mask 处理结果")
    with gr.Row():
        mask_preview_output = gr.Image(
            label="前景 Mask 预览",
            type="pil",
        )
        processed_foreground_output = gr.Image(
            label="处理后 RGBA 前景",
            type="pil",
        )

    gr.Markdown("## 全部候选结果")
    candidate_gallery = gr.Gallery(
        label="候选合成图",
        columns=3,
        height="auto",
    )

    gr.Markdown("## Top-K 推荐结果")
    topk_gallery = gr.Gallery(
        label="Top-K 推荐图",
        columns=3,
        height="auto",
    )

    gr.Markdown("## 评分表")
    score_table = gr.Dataframe(
        label="候选评分表",
        wrap=True,
    )

    gr.Markdown("## 本次实验分析说明")
    run_analysis_text = gr.Textbox(
        label="自动生成的分析说明",
        lines=22,
    )

    gr.Markdown("## 模型解释图")
    explanation_gallery = gr.Gallery(
        label="遮挡实验热力图",
        columns=1,
        height="auto",
    )

    gr.Markdown("## 导出文件")
    with gr.Row():
        processed_fg_file = gr.File(label="处理后前景 PNG")
        mask_file = gr.File(label="Mask 预览 PNG")
        csv_file = gr.File(label="评分 CSV")

    with gr.Row():
        log_file = gr.File(label="推理日志")
        report_file = gr.File(label="Markdown 运行报告")
        metadata_file = gr.File(label="JSON 元信息")

    explanation_report_file = gr.File(label="模型解释 Markdown 报告")

    run_button.click(
        fn=run_smartplace,
        inputs=[
            background_input,
            foreground_input,
            mask_mode_input,
            white_bg_threshold_input,
            candidate_count_input,
            scale_input,
            top_k_input,
            filter_out_of_bounds_input,
            enable_explanation_input,
            occlusion_patch_size_input,
            occlusion_stride_input,
        ],
        outputs=[
            mask_preview_output,
            processed_foreground_output,
            candidate_gallery,
            topk_gallery,
            score_table,
            run_analysis_text,
            explanation_gallery,
            processed_fg_file,
            mask_file,
            csv_file,
            log_file,
            report_file,
            metadata_file,
            explanation_report_file,
        ],
    )


if __name__ == "__main__":
    demo.launch()