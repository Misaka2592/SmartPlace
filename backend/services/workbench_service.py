from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

from utils.case_manager import (
    build_case_record,
    export_case_summary_csv,
    export_case_summary_markdown,
    load_case_records,
    save_case_record,
    summarize_case_records,
)
from utils.composer import compose_image_with_mask, resize_foreground
from utils.explain import (
    export_explanation_markdown,
    generate_calibration_feature_plot,
    generate_gradient_saliency_map,
    generate_occlusion_heatmap,
)
from utils.exporter import export_markdown_report, export_result_package_metadata
from utils.mask_processor import process_foreground_for_composition, save_processed_foreground
from utils.scoring import analyze_candidate, format_score, summarize_run

from .runtime import (
    CASE_DIR,
    COMPOSITE_DIR,
    EXPLAIN_DIR,
    MASK_DIR,
    REPORT_RESULT_DIR,
    RUN_ROOT,
    SESSION_ROOT,
    TABLE_DIR,
    UPLOAD_ROOT,
    cfg,
    get_runtime_scorer,
    libcom_multimodel,
    logger,
    u2net_runner,
)


def format_param_value(value: Any, digits: int = 0) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits <= 0:
        return str(int(round(num)))
    return f"{num:.{digits}f}"


def json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items() if k not in {"image", "candidate_info", "composite_mask"}}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def make_export_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    export_items: List[Dict[str, Any]] = []
    for item in items:
        copied: Dict[str, Any] = {}
        for key, value in item.items():
            if key in {"image", "candidate_info"}:
                continue
            copied[key] = json_safe_value(value)
        export_items.append(copied)
    return export_items


def assign_relative_labels_in_place(results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked_all = sorted(results, key=lambda item: float(item["score"]), reverse=True)
    n = len(ranked_all)
    top_k = max(1, min(int(top_k), n)) if n else 1
    for rank_idx, item in enumerate(ranked_all):
        item["rank"] = rank_idx + 1
        if rank_idx < top_k:
            item["label"] = "推荐"
            item["conclusion"] = "该候选在本组候选中 OPA 分数排名靠前，作为 Top-K 推荐结果。"
        elif rank_idx < max(top_k + 3, n // 2):
            item["label"] = "可接受"
            item["conclusion"] = "该候选在本组候选中 OPA 分数处于中等水平，可以作为备选结果。"
        else:
            item["label"] = "不推荐"
            item["conclusion"] = "该候选在本组候选中 OPA 分数排名较低，不建议优先使用。"
    return ranked_all


def build_run_analysis_text(summary: Dict[str, Any], ranked: List[Dict[str, Any]], mask_info: Dict[str, Any], drag_mode: str, explanation_text: str = "") -> str:
    lines = [
        "【本次运行统计】",
        f"候选总数：{summary['total_candidates']}",
        f"推荐数量：{summary['recommend_count']}",
        f"可接受数量：{summary['acceptable_count']}",
        f"不推荐数量：{summary['not_recommend_count']}",
        f"最佳候选编号：{summary['best_candidate_id']}",
        f"最佳候选分数：{summary['best_score']:.4f}" if summary.get("best_score") is not None else "最佳候选分数：-",
        f"平均分数：{summary['average_score']:.4f}" if summary.get("average_score") is not None else "平均分数：-",
        "",
        "【交互方式】",
        drag_mode,
        "",
        "【前景处理说明】",
        f"请求模式：{mask_info.get('requested_mode')}",
        f"实际使用：{mask_info.get('mode_used')}",
        f"输入尺寸：{mask_info.get('input_size')}",
        f"输出尺寸：{mask_info.get('output_size')}",
        f"前景像素占比：{mask_info.get('foreground_pixel_ratio', 0):.4f}",
    ]
    if "auto_decision" in mask_info:
        lines.append(f"自动判断结果：{mask_info.get('auto_decision')}")
    if "estimated_background_color" in mask_info:
        lines.append(f"估计背景颜色：{mask_info.get('estimated_background_color')}")
    lines.extend(["", "【Top-K 推荐解释】"])
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            f"Top {rank}：候选 {item['id']}，位置 ({item['x']}, {item['y']})，分数={item['score']:.4f}，评价={item['label']}。"
        )
        lines.append(f"理由：{item['reason']}")
        lines.append(f"结论：{item.get('conclusion', '')}")
        lines.append("")
    if explanation_text:
        lines.extend(["【解释结果】", explanation_text])
    return "\n".join(lines)


def rel_url(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[2]
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(project_root)
        return "/files/" + relative.as_posix()
    except ValueError:
        pass

    parts = list(resolved.parts)
    project_name = project_root.name
    if project_name in parts:
        idx = parts.index(project_name)
        rebuilt = project_root.joinpath(*parts[idx + 1 :])
        try:
            return "/files/" + rebuilt.relative_to(project_root).as_posix()
        except ValueError:
            pass

    for anchor in ("outputs", "report", "assets", "backend"):
        if anchor in parts:
            idx = parts.index(anchor)
            rebuilt = project_root.joinpath(*parts[idx:])
            return "/files/" + rebuilt.relative_to(project_root).as_posix()

    raise ValueError(f"Cannot convert path to project-relative URL: {resolved}")


def save_upload_image(image_bytes: bytes, kind: str, filename: str) -> Dict[str, Any]:
    suffix = Path(filename).suffix or ".png"
    asset_id = uuid.uuid4().hex
    target = (UPLOAD_ROOT / f"{kind}s" / f"{asset_id}{suffix}").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(image_bytes)
    image = Image.open(target)
    return {
        "asset_id": asset_id,
        "kind": kind,
        "filename": target.name,
        "path": str(target),
        "url": rel_url(target),
        "size": {"width": image.width, "height": image.height},
    }


def _load_image(path: str, mode: str) -> Image.Image:
    return Image.open(path).convert(mode)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compose_session(background_path: str, foreground_path: str, mask_mode: str, white_bg_threshold: int, scale: float) -> Dict[str, Any]:
    background = _load_image(background_path, "RGB")
    raw_foreground = _load_image(foreground_path, "RGBA")

    mask_cfg = cfg.get("mask_processor", {})
    foreground, mask_preview, mask_info = process_foreground_for_composition(
        image=raw_foreground,
        mode=mask_mode,
        white_bg_threshold=int(white_bg_threshold),
        edge_sample_ratio=float(mask_cfg.get("edge_sample_ratio", 0.08)),
        handin_u2net_runner=u2net_runner,
    )

    processed_fg_path = None
    mask_path = None
    if cfg.get("output", {}).get("save_mask", True):
        processed_fg_path, mask_path = save_processed_foreground(
            foreground_rgba=foreground,
            mask_preview=mask_preview,
            output_dir=str(MASK_DIR),
        )
        mask_info["processed_foreground_path"] = processed_fg_path
        mask_info["mask_path"] = mask_path

    bg_w, bg_h = background.size
    resized_fg = resize_foreground(foreground=foreground, scale=float(scale), bg_width=bg_w, bg_height=bg_h)
    fg_w, fg_h = resized_fg.size
    init_x = max(0, (bg_w - fg_w) // 2)
    init_y = max(0, bg_h - fg_h - int(bg_h * 0.08))

    session_id = uuid.uuid4().hex
    session_path = SESSION_ROOT / f"{session_id}.json"
    payload = {
        "session_id": session_id,
        "background_path": background_path,
        "foreground_path": foreground_path,
        "processed_foreground_path": processed_fg_path or foreground_path,
        "mask_preview_path": mask_path,
        "mask_info": json_safe_value(mask_info),
        "initial_state": {
            "x": init_x,
            "y": init_y,
            "scale": float(scale),
            "bg_width": bg_w,
            "bg_height": bg_h,
            "fg_width": fg_w,
            "fg_height": fg_h,
        },
    }
    _write_json(session_path, payload)

    return {
        "session_id": session_id,
        "background": {"url": rel_url(Path(background_path)), "width": bg_w, "height": bg_h},
        "foreground": {"url": rel_url(Path(processed_fg_path or foreground_path)), "width": fg_w, "height": fg_h},
        "mask_preview_url": rel_url(Path(mask_path)) if mask_path else None,
        "mask_info": payload["mask_info"],
        "initial_state": payload["initial_state"],
    }


def get_session(session_id: str) -> Dict[str, Any]:
    return _read_json(SESSION_ROOT / f"{session_id}.json")


def score_session(
    session_id: str,
    candidate_points: List[Dict[str, Any]],
    top_k: int,
    filter_out_of_bounds: bool,
    enable_explanation: bool,
    enable_saliency: bool,
    enable_feature_analysis: bool,
    occlusion_patch_size: int,
    occlusion_stride: int,
    enable_libcom_suite: bool,
    libcom_suite_models: List[str],
    lbm_steps: int,
    lbm_resolution: int,
    case_name: str,
    background_note: str,
    foreground_note: str,
    manual_label: str,
    manual_reason: str,
    drag_mode_state: str,
    score_backend: str,
) -> Dict[str, Any]:
    session = get_session(session_id)
    background = _load_image(session["background_path"], "RGB")
    foreground = _load_image(session["processed_foreground_path"], "RGBA")
    mask_info = dict(session.get("mask_info") or {})
    scorer = get_runtime_scorer(score_backend)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    mask_info["score_backend"] = score_backend

    composites = []
    candidate_infos = []
    candidates = []
    for p in candidate_points:
        composite, composite_mask, info = compose_image_with_mask(
            background=background,
            foreground=foreground,
            x=int(p["x"]),
            y=int(p["y"]),
            scale=float(p["scale"]),
            allow_out_of_bounds=not bool(filter_out_of_bounds),
        )
        info["composite_mask"] = composite_mask
        info["candidate_id"] = p["id"]
        composites.append(composite)
        candidate_infos.append(info)
        candidates.append(p)

    scores = scorer.batch_score(composites, candidate_infos)
    results: List[Dict[str, Any]] = []
    for cand, composite, info, score in zip(candidates, composites, candidate_infos, scores):
        analysis = analyze_candidate(info, float(score))
        results.append({
            "id": cand["id"],
            "x": info["x"],
            "y": info["y"],
            "scale": float(cand["scale"]),
            "score": float(score),
            "label": "未标定",
            "reason": analysis["reason"],
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
            "composite_path": None,
            "saved_path": None,
        })

    ranked_all = assign_relative_labels_in_place(results, top_k=top_k)
    ranked = ranked_all[:top_k]

    if cfg.get("output", {}).get("save_images", True):
        for item in results:
            filename = f"{run_id}_candidate_{item['id']}_score_{float(item['score']):.4f}.png"
            path = COMPOSITE_DIR / filename
            item["image"].save(path)
            item["composite_path"] = str(path)
            item["saved_path"] = str(path)

    summary = summarize_run(results, top_k=top_k)

    explanation_text = ""
    explanation_overlay_path = None
    explanation_saliency_path = None
    explanation_feature_plot_path = None
    explanation_report_path = None
    if (enable_explanation or enable_saliency or enable_feature_analysis) and ranked:
        top1 = ranked[0]
        if enable_explanation:
            explanation_result = generate_occlusion_heatmap(
                scorer=scorer,
                image=top1["image"],
                candidate_info=top1["candidate_info"],
                patch_size=int(occlusion_patch_size),
                stride=int(occlusion_stride),
                output_dir=str(EXPLAIN_DIR),
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_overlay_path = explanation_result["overlay_path"]
            explanation_report_path = export_explanation_markdown(
                explanation_result=explanation_result,
                candidate_id=top1["id"],
                output_dir=str(REPORT_RESULT_DIR),
            )
            explanation_text = explanation_result["explanation"]
        if enable_saliency:
            saliency_result = generate_gradient_saliency_map(
                image=top1["image"],
                output_dir=str(EXPLAIN_DIR),
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_saliency_path = saliency_result["overlay_path"]
        if enable_feature_analysis:
            feature_result = generate_calibration_feature_plot(
                candidate_info=top1["candidate_info"],
                output_dir=str(EXPLAIN_DIR),
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_feature_plot_path = feature_result["feature_plot_path"]

    libcom_suite_text = ""
    libcom_suite_gallery: List[Dict[str, Any]] = []
    if enable_libcom_suite and ranked:
        top1 = ranked[0]
        try:
            suite_output = libcom_multimodel.run(
                background=background,
                foreground=foreground,
                composite=top1["image"],
                composite_mask=top1["candidate_info"]["composite_mask"],
                candidate_info=top1["candidate_info"],
                models=list(libcom_suite_models or []),
                lbm_steps=int(lbm_steps),
                lbm_resolution=int(lbm_resolution),
                run_id=f"{run_id}_candidate_{top1['id']}",
            )
            text, gallery = libcom_multimodel.build_ui_payload(suite_output)
            libcom_suite_text = text
            for path, caption in gallery:
                libcom_suite_gallery.append({"url": rel_url(Path(path)), "caption": caption})
        except Exception as exc:
            libcom_suite_text = f"LibCom 增强模型运行失败：{repr(exc)}"

    table_rows = [{
        "candidate_id": item["id"],
        "rank": item.get("rank"),
        "x": item["x"],
        "y": item["y"],
        "scale": item["scale"],
        "score": format_score(item["score"]),
        "label": item["label"],
        "out_of_bounds": item["out_of_bounds"],
        "area_ratio": f"{item['area_ratio']:.4f}",
        "reason": item["reason"],
        "conclusion": item["conclusion"],
        "composite_path": item.get("composite_path"),
    } for item in results]
    df = pd.DataFrame(table_rows)
    csv_path = TABLE_DIR / f"{run_id}_drag_scores.csv"
    if cfg.get("output", {}).get("save_csv", True):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    run_analysis = build_run_analysis_text(
        summary=summary,
        ranked=ranked,
        mask_info=mask_info,
        drag_mode=drag_mode_state or "用户拖拽前景物体并记录候选位置。",
        explanation_text=explanation_text,
    )

    export_results = make_export_results(results)
    export_ranked = make_export_results(ranked)
    model_info = scorer.get_model_info()
    log_path = logger.get_log_path()

    report_path = export_markdown_report(
        results=export_results,
        ranked=export_ranked,
        summary=summary,
        model_info=model_info,
        output_dir=str(REPORT_RESULT_DIR),
        csv_path=str(csv_path),
        log_path=log_path,
        explanation_path=explanation_overlay_path,
        explanation_report_path=explanation_report_path,
    )
    metadata_path = export_result_package_metadata(
        results=export_results,
        ranked=export_ranked,
        summary=summary,
        model_info=model_info,
        output_dir=str(REPORT_RESULT_DIR),
        csv_path=str(csv_path),
        log_path=log_path,
        report_path=report_path,
    )

    files = {
        "csv_path": str(csv_path),
        "log_path": log_path,
        "run_report_path": report_path,
        "metadata_path": metadata_path,
        "explanation_overlay_path": explanation_overlay_path,
        "explanation_report_path": explanation_report_path,
    }
    case_record = build_case_record(
        case_name=case_name,
        background_note=background_note,
        foreground_note=foreground_note,
        manual_label=manual_label,
        manual_reason=manual_reason,
        mask_info=mask_info,
        summary=summary,
        ranked=export_ranked,
        results=export_results,
        files=files,
    )
    case_record_path = save_case_record(record=case_record, output_dir=str(CASE_DIR))
    all_case_records = load_case_records(str(CASE_DIR))
    case_summary_df = summarize_case_records(all_case_records)
    case_summary_csv_path = export_case_summary_csv(records=all_case_records, output_dir=str(REPORT_RESULT_DIR))
    case_summary_md_path = export_case_summary_markdown(records=all_case_records, output_dir=str(REPORT_RESULT_DIR))

    run_payload = {
        "run_id": run_id,
        "session_id": session_id,
        "summary": summary,
        "mask_info": json_safe_value(mask_info),
        "ranked": export_ranked,
        "results": export_results,
        "result_image_paths": {str(item["id"]): item.get("composite_path") for item in results},
        "files": {
            **files,
            "case_record_path": case_record_path,
            "case_summary_csv_path": case_summary_csv_path,
            "case_summary_md_path": case_summary_md_path,
            "explanation_saliency_path": explanation_saliency_path,
            "explanation_feature_plot_path": explanation_feature_plot_path,
        },
        "model_info": json_safe_value(model_info),
        "run_analysis_text": run_analysis,
        "libcom_suite_text": libcom_suite_text,
    }
    _write_json(RUN_ROOT / f"{run_id}.json", run_payload)

    return {
        "run_id": run_id,
        "summary": summary,
        "ranked": [
            {
                **item,
                "image_url": rel_url(Path(item["composite_path"])) if item.get("composite_path") else None,
            }
            for item in export_ranked
        ],
        "results": [
            {
                **item,
                "image_url": rel_url(Path(path)) if (path := next((r.get("composite_path") for r in export_results if r["id"] == item["id"]), None)) else None,
            }
            for item in export_results
        ],
        "score_table": table_rows,
        "run_analysis_text": run_analysis,
        "explanations": {
            "occlusion_url": rel_url(Path(explanation_overlay_path)) if explanation_overlay_path else None,
            "saliency_url": rel_url(Path(explanation_saliency_path)) if explanation_saliency_path else None,
            "feature_plot_url": rel_url(Path(explanation_feature_plot_path)) if explanation_feature_plot_path else None,
            "report_url": rel_url(Path(explanation_report_path)) if explanation_report_path else None,
        },
        "libcom_suite": {
            "text": libcom_suite_text,
            "gallery": libcom_suite_gallery,
        },
        "exports": {
            "csv_url": rel_url(Path(csv_path)),
            "log_url": rel_url(Path(log_path)) if log_path else None,
            "report_url": rel_url(Path(report_path)),
            "metadata_url": rel_url(Path(metadata_path)),
            "case_record_url": rel_url(Path(case_record_path)),
            "case_summary_csv_url": rel_url(Path(case_summary_csv_path)),
            "case_summary_md_url": rel_url(Path(case_summary_md_path)),
        },
        "case_summary": case_summary_df.fillna("").to_dict(orient="records"),
    }


def generate_heatmap(run_id: str, candidate_id: Optional[int] = None, patch_size: int = 96, stride: int = 96, score_backend: Optional[str] = None) -> Dict[str, Any]:
    run = _read_json(RUN_ROOT / f"{run_id}.json")
    session = get_session(run["session_id"])
    background = _load_image(session["background_path"], "RGB")
    foreground = _load_image(session["processed_foreground_path"], "RGBA")
    backend_key = score_backend or session.get("mask_info", {}).get("score_backend") or "handin_opa_subprocess"
    scorer = get_runtime_scorer(backend_key)
    target = next((item for item in run["results"] if item["id"] == (candidate_id or run["summary"]["best_candidate_id"])), None)
    if not target:
        raise ValueError("Candidate not found for heatmap generation.")
    composite, composite_mask, info = compose_image_with_mask(
        background=background,
        foreground=foreground,
        x=int(target["x"]),
        y=int(target["y"]),
        scale=float(target["scale"]),
        allow_out_of_bounds=True,
    )
    info["composite_mask"] = composite_mask
    result = generate_occlusion_heatmap(
        scorer=scorer,
        image=composite,
        candidate_info=info,
        patch_size=patch_size,
        stride=stride,
        output_dir=str(EXPLAIN_DIR),
        prefix=f"api_{run_id}_{target['id']}",
    )
    return {
        "run_id": run_id,
        "candidate_id": target["id"],
        "overlay_url": rel_url(Path(result["overlay_path"])),
        "explanation": result["explanation"],
    }


def export_report_bundle(run_id: str) -> Dict[str, Any]:
    run = _read_json(RUN_ROOT / f"{run_id}.json")
    files = run.get("files", {})
    return {
        "run_id": run_id,
        "exports": {
            key: rel_url(Path(value)) if value else None
            for key, value in files.items()
        },
    }
