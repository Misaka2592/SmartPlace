import os
import time
from typing import Dict, List

import gradio as gr
import pandas as pd
import yaml
from PIL import Image

from utils.composer import compose_image, resize_foreground
from utils.candidate_generator import generate_grid_candidates
from utils.scoring import score_to_label, build_reason, rank_candidates, format_score
from utils.logger import InferenceLogger
from models.dummy_scorer import DummyScorer


OUTPUT_DIR = "outputs"
COMPOSITE_DIR = os.path.join(OUTPUT_DIR, "composites")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CONFIG_PATH = "configs/default.yaml"

os.makedirs(COMPOSITE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
        "说明：当前为 DummyScorerV2，占位规则模型。它用于模拟真实模型推理接口和日志，后续可替换为 OPA/FOPA。",
    ]

    return "\n".join(lines)


def run_smartplace(
    background_image,
    foreground_image,
    candidate_count,
    scale,
    top_k,
    filter_out_of_bounds,
):
    if background_image is None:
        raise gr.Error("请上传背景图。")

    if foreground_image is None:
        raise gr.Error("请上传前景图，建议使用透明 PNG。")

    background = Image.fromarray(background_image).convert("RGB")
    foreground = Image.fromarray(foreground_image).convert("RGBA")

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
    logger.log(f"[Input] original_foreground_size={foreground.size}")
    logger.log(f"[Input] resized_foreground_size={resized_fg.size}")
    logger.log(f"[Param] candidate_count={candidate_count}")
    logger.log(f"[Param] scale={scale}")
    logger.log(f"[Param] top_k={top_k}")
    logger.log(f"[Param] filter_out_of_bounds={filter_out_of_bounds}")

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
        reason = build_reason(score, info)

        result = {
            "id": cand["id"],
            "x": info["x"],
            "y": info["y"],
            "scale": float(scale),
            "score": score,
            "label": label,
            "reason": reason,
            "out_of_bounds": info["out_of_bounds"],
            "fg_width": info["fg_width"],
            "fg_height": info["fg_height"],
            "image": composite,
        }

        results.append(result)

    ranked = rank_candidates(results, top_k=int(top_k))

    if cfg.get("output", {}).get("save_images", True):
        save_candidate_images(results)

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
                "推荐理由/失败提示": item["reason"],
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
    topk_text_lines = []

    for rank, item in enumerate(ranked, start=1):
        caption = (
            f"Top {rank} - 候选 {item['id']} | "
            f"score={item['score']:.4f} | "
            f"{item['label']}"
        )
        topk_gallery.append((item["image"], caption))

        topk_text_lines.append(
            f"Top {rank}: 候选 {item['id']}，"
            f"位置=({item['x']}, {item['y']})，"
            f"分数={item['score']:.4f}，"
            f"评价={item['label']}，"
            f"原因：{item['reason']}"
        )

    topk_text = "\n\n".join(topk_text_lines)

    logger.log("[SmartPlace] Inference finished.")
    logger.log(f"[Output] score_table_saved={csv_path}")
    logger.log(f"[Output] log_file={logger.get_log_path()}")

    return gallery_items, topk_gallery, df, topk_text, csv_path, logger.get_log_path()


with gr.Blocks(title="SmartPlace 智能物体放置推荐系统") as demo:
    gr.Markdown(
        """
        # SmartPlace：智能物体放置与合成图质量评价系统

        当前版本：v0.2 推理接口规范版。

        本版本新增：
        - 模型统一接口 `BaseScorer`
        - 批量评分接口 `batch_score`
        - 推理日志 `InferenceLogger`
        - 模型信息展示
        - 终端打印模型加载、输入 tensor shape、输出 score、推理时间
        """
    )

    with gr.Row():
        with gr.Column():
            background_input = gr.Image(
                label="背景图",
                type="numpy",
            )
            foreground_input = gr.Image(
                label="前景图，建议透明 PNG",
                type="numpy",
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

            run_button = gr.Button(
                value="生成候选并评分",
                variant="primary",
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

    gr.Markdown("## 推荐说明")
    topk_text = gr.Textbox(
        label="Top-K 推荐说明",
        lines=8,
    )

    with gr.Row():
        csv_file = gr.File(
            label="导出的评分 CSV",
        )

        log_file = gr.File(
            label="导出的推理日志",
        )

    run_button.click(
        fn=run_smartplace,
        inputs=[
            background_input,
            foreground_input,
            candidate_count_input,
            scale_input,
            top_k_input,
            filter_out_of_bounds_input,
        ],
        outputs=[
            candidate_gallery,
            topk_gallery,
            score_table,
            topk_text,
            csv_file,
            log_file,
        ],
    )


if __name__ == "__main__":
    demo.launch()