import base64
import html as html_lib
import json
import os
import subprocess
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import yaml
from PIL import Image
from typing import Optional

from models.libcom_opa_subprocess_scorer import LibcomOPASubprocessScorer
from models.handin_opa_subprocess_scorer import HandinOPASubprocessScorer
from models.smartplace_opa_calibrated_scorer import SmartPlaceOPACalibratedScorer
from models.libcom_multimodel_subprocess import LibcomMultiModelSubprocess
from utils.scoring import format_score, analyze_candidate, summarize_run
from utils.logger import InferenceLogger
from utils.exporter import export_markdown_report, export_result_package_metadata
from utils.explain import (
    generate_occlusion_heatmap,
    generate_gradient_saliency_map,
    generate_calibration_feature_plot,
    export_explanation_markdown,
)
from utils.mask_processor import process_foreground_for_composition, save_processed_foreground
from utils.handin_u2net_subprocess import HandinU2NetSubprocessMatting
from utils.case_manager import (
    build_case_record,
    save_case_record,
    load_case_records,
    summarize_case_records,
    export_case_summary_csv,
    export_case_summary_markdown,
)
from utils.auto_candidate_area_search import auto_candidate_area_search
from utils.composer import compose_image_with_mask, resize_foreground
from utils.image_warnings import (
    check_foreground_size,
    check_background_quality,
    check_no_candidates,
    collect_run_warnings,
)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def collect_preset_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(exts)
    ]
    return files if files else [""]


def format_param_value(value: Any, digits: int = 0) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits <= 0:
        return str(int(round(num)))
    return f"{num:.{digits}f}"


CUSTOM_DRAG_JS = """
() => {
    window.smartplaceDragState = window.smartplaceDragState || {
        x: 0,
        y: 0,
        scale: 1.0
    };

    window.addEventListener("message", (event) => {
        const data = event.data || {};
        if (data.type !== "smartplace-drag-update") {
            return;
        }

        window.smartplaceDragState = {
            x: data.x,
            y: data.y,
            scale: data.scale
        };

        function setInputValue(elemId, value) {
            const root = document.getElementById(elemId);
            if (!root) return;

            const input = root.querySelector("textarea, input");
            if (!input) return;

            input.value = String(value);
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        }

        setInputValue("drag_x_input", data.x);
        setInputValue("drag_y_input", data.y);
        setInputValue("drag_scale_input", data.scale);
    });
}
"""

RECORD_CANDIDATE_JS = """
(candidate_points, x, y, scale) => {
    let latestX = x;
    let latestY = y;
    let latestScale = scale;

    try {
        const iframe = document.getElementById("smartplace_drag_iframe");
        if (iframe && iframe.contentWindow && iframe.contentWindow.getSmartPlaceDragState) {
            const s = iframe.contentWindow.getSmartPlaceDragState();
            if (s && s.x !== undefined && s.y !== undefined) {
                latestX = s.x;
                latestY = s.y;
                latestScale = s.scale;
            }
        } else if (window.smartplaceDragState) {
            const s = window.smartplaceDragState;
            latestX = s.x !== undefined ? s.x : latestX;
            latestY = s.y !== undefined ? s.y : latestY;
            latestScale = s.scale !== undefined ? s.scale : latestScale;
        }
    } catch (e) {
        if (window.smartplaceDragState) {
            const s = window.smartplaceDragState;
            latestX = s.x !== undefined ? s.x : latestX;
            latestY = s.y !== undefined ? s.y : latestY;
            latestScale = s.scale !== undefined ? s.scale : latestScale;
        }
    }

    return [
        candidate_points,
        String(latestX),
        String(latestY),
        String(latestScale)
    ];
}
"""

APP_CSS = r"""
:root {
  --sp-bg: #f7fbff;
  --sp-bg-2: #eef6ff;
  --sp-panel: rgba(255, 255, 255, 0.78);
  --sp-panel-strong: rgba(255, 255, 255, 0.94);
  --sp-border: rgba(160, 199, 237, 0.34);
  --sp-border-strong: rgba(95, 175, 255, 0.55);
  --sp-ink: #20304d;
  --sp-muted: #6f7e98;
  --sp-primary: #53b2ff;
  --sp-primary-2: #70ead8;
  --sp-accent: #9d8cff;
  --sp-success: #59d9ad;
  --sp-danger: #ff7c98;
  --sp-radius-lg: 34px;
  --sp-shadow-lg: 0 24px 60px rgba(114, 147, 206, 0.16);
  --sp-shadow-md: 0 14px 30px rgba(114, 147, 206, 0.12);
  --sp-ease: 220ms cubic-bezier(0.22, 1, 0.36, 1);
}
html, body {
  margin: 0;
  background:
    radial-gradient(circle at 10% 10%, rgba(83, 178, 255, 0.16), transparent 30%),
    radial-gradient(circle at 90% 12%, rgba(112, 234, 216, 0.16), transparent 24%),
    radial-gradient(circle at 54% 100%, rgba(157, 140, 255, 0.12), transparent 28%),
    linear-gradient(180deg, var(--sp-bg) 0%, var(--sp-bg-2) 48%, #fbfdff 100%) !important;
  color: var(--sp-ink) !important;
}
.gradio-container {
  min-height: 100vh !important;
  padding: 12px !important;
  color: var(--sp-ink) !important;
  font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
  background: transparent !important;
}
.gradio-container *, .gradio-container *::before, .gradio-container *::after { box-sizing: border-box; }
.gradio-container main.contain { max-width: 1680px !important; margin: 0 auto !important; }
#sp-app-frame {
  position: relative;
  margin: 0 auto 10px;
  border-radius: 30px;
  overflow: visible;
  border: 1px solid rgba(255,255,255,0.78);
  background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(244,249,255,0.64));
  box-shadow: var(--sp-shadow-lg);
}
#sp-app-frame::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(circle at top left, rgba(83, 178, 255, 0.12), transparent 34%),
    radial-gradient(circle at bottom right, rgba(112, 234, 216, 0.10), transparent 26%);
}
#sp-app-inner { position: relative; padding: 14px; overflow: visible; }
#sp-topbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px 18px;
  margin-bottom: 10px;
}
.sp-brand { display: grid; grid-template-columns: 56px minmax(0, 1fr); align-items: center; gap: 14px; }
.sp-logo {
  width: 50px; height: 50px; display: grid; place-items: center; border-radius: 16px;
  background: linear-gradient(135deg, #58b4ff, #77f0d9); color: #13253d; font-size: 19px; font-weight: 900;
  letter-spacing: -0.08em; box-shadow: 0 14px 28px rgba(88, 180, 255, 0.24);
}
.sp-brand h1 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.35rem); line-height: 1; letter-spacing: -0.05em; color: #233250; }
.sp-brand p { margin: 6px 0 0; color: var(--sp-muted); line-height: 1.45; font-size: 0.9rem; }
.sp-toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
}
.sp-status-strip { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-start; }
.sp-pill {
  display: inline-flex; align-items: center; gap: 8px; min-height: 38px; padding: 8px 12px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.9); background: rgba(255,255,255,0.56); color: #5a6f89;
  box-shadow: 0 10px 20px rgba(122, 144, 184, 0.08); backdrop-filter: blur(14px);
  font-size: 0.8rem;
}
.sp-mode-pill {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 8px 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(83, 178, 255, 0.15), rgba(112, 234, 216, 0.18));
  border: 1px solid rgba(124, 196, 255, 0.42);
  color: #2f5c86;
  font-size: 0.8rem;
  font-weight: 760;
}
.sp-toolbar-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 40px;
  padding: 8px 15px;
  border-radius: 999px;
  color: #183050;
  background: linear-gradient(135deg, rgba(99, 183, 255, 0.94), rgba(116, 234, 216, 0.9));
  box-shadow: 0 12px 24px rgba(91, 180, 255, 0.18);
  font-size: 0.82rem;
  font-weight: 760;
}
.sp-pill-dot { width: 10px; height: 10px; border-radius: 999px; background: var(--sp-success); box-shadow: 0 0 0 5px rgba(89, 217, 173, 0.14); }
#sp-hero-grid { display: grid; grid-template-columns: 1fr; gap: 0; }
.sp-workflow-card {
  position: relative;
  display: grid;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 24px;
  border: 1px solid rgba(255,255,255,0.84);
  background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(246,250,255,0.72));
  box-shadow: var(--sp-shadow-md);
}
.sp-workflow-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.sp-workflow-title {
  color: #24324f;
  font-size: 0.96rem;
  font-weight: 780;
  letter-spacing: -0.02em;
}
.sp-workflow-note {
  color: var(--sp-muted);
  font-size: 0.84rem;
}
.sp-workflow-rail {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.sp-work-step {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 8px 12px;
  border-radius: 999px;
  color: #6281a1;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(198, 218, 241, 0.72);
  font-size: 0.82rem;
  font-weight: 700;
}
.sp-work-step.is-active {
  color: #1f4467;
  background: linear-gradient(135deg, rgba(83, 178, 255, 0.16), rgba(112, 234, 216, 0.18));
  border-color: rgba(117, 193, 255, 0.44);
}
.sp-work-arrow {
  color: #91a3bb;
  font-size: 0.92rem;
  font-weight: 800;
}
.sp-hero-card, .sp-showcase-card, .sp-metric {
  position: relative; overflow: hidden; border-radius: var(--sp-radius-lg); border: 1px solid rgba(255,255,255,0.82);
  background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(247,251,255,0.68)); box-shadow: var(--sp-shadow-md); backdrop-filter: blur(18px);
}
.sp-card, .sp-card-tight {
  position: relative; overflow: visible; border-radius: var(--sp-radius-lg); border: 1px solid rgba(255,255,255,0.82);
  background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(247,251,255,0.68)); box-shadow: var(--sp-shadow-md); backdrop-filter: blur(18px);
}
.sp-hero-card { padding: 24px 26px; min-height: 260px; display: flex; flex-direction: column; justify-content: space-between; }
.sp-hero-card::after {
  content: ""; position: absolute; right: -120px; top: -140px; width: 360px; height: 360px; border-radius: 999px;
  background: radial-gradient(circle, rgba(83,178,255,0.22), rgba(112,234,216,0.05) 56%, transparent 72%); pointer-events: none;
}
.sp-kicker {
  display: inline-flex; align-items: center; min-height: 38px; padding: 0 14px; border-radius: 999px;
  color: #4794db; background: rgba(83, 178, 255, 0.1); border: 1px solid rgba(83, 178, 255, 0.18);
  font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em;
}
.sp-hero-card h2 { margin: 14px 0 0; max-width: 12ch; font-size: clamp(2rem, 5vw, 3.9rem); line-height: 0.98; letter-spacing: -0.08em; color: #233250; }
.sp-hero-card h2 span {
  display: block; color: transparent; background: linear-gradient(120deg, #49a8ff, #71dfd1 58%, #8d7dff);
  -webkit-background-clip: text; background-clip: text;
}
.sp-hero-card p { margin: 14px 0 0; max-width: 58ch; color: var(--sp-muted); font-size: 0.96rem; line-height: 1.72; }
.sp-hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.sp-soft-tag {
  min-height: 38px; display: inline-flex; align-items: center; padding: 0 13px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.82); color: #617590; background: rgba(255,255,255,0.52); box-shadow: 0 8px 16px rgba(122, 144, 184, 0.06);
  font-size: 0.84rem;
}
.sp-showcase-card { padding: 18px; min-height: 260px; display: grid; grid-template-rows: 132px auto auto; gap: 14px; }
.sp-score-orbit {
  position: relative; min-height: 132px; border-radius: 22px;
  background: radial-gradient(circle at 24% 28%, rgba(83, 178, 255, 0.24), transparent 0 18%), radial-gradient(circle at 70% 68%, rgba(112, 234, 216, 0.22), transparent 0 18%), linear-gradient(135deg, rgba(255,255,255,0.92), rgba(244,249,255,0.76));
  border: 1px solid rgba(255,255,255,0.86); overflow: hidden;
}
.sp-score-orbit::before, .sp-score-orbit::after {
  content: ""; position: absolute; inset: 18px; border-radius: 999px; border: 1px solid rgba(83, 178, 255, 0.16); animation: spOrbit 8s linear infinite;
}
.sp-score-orbit::after { inset: 36px; animation-duration: 12s; animation-direction: reverse; }
.sp-showcase-card h3 { margin: 0; font-size: 1.08rem; line-height: 1.35; color: #24324f; }
.sp-showcase-card p { margin: 0; color: var(--sp-muted); line-height: 1.72; font-size: 0.92rem; }
.sp-stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.sp-stat {
  border-radius: 18px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.62);
  border: 1px solid rgba(255,255,255,0.82);
}
.sp-stat span {
  display: block;
  font-size: 0.74rem;
  color: var(--sp-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.sp-stat strong {
  display: block;
  margin-top: 4px;
  font-size: 0.98rem;
  color: #24324f;
}
.sp-metric-row { display: none !important; }
.sp-metric { border-radius: 24px; padding: 18px; }
.sp-metric::before {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: linear-gradient(90deg, rgba(75,168,255,0.85), rgba(103,228,210,0.48), transparent 78%);
}
.sp-metric .label { color: var(--sp-muted); font-size: 0.84rem; margin-bottom: 10px; }
.sp-metric .value { font-size: 1.06rem; font-weight: 760; line-height: 1.35; color: #273654; }
.sp-card, .sp-card-tight { border-radius: 28px !important; }
.sp-card { padding: 16px !important; }
.sp-card-tight { padding: 12px !important; }
.sp-section-title { display: flex; align-items: center; gap: 10px; margin: 0 0 8px 0; font-size: 1rem; font-weight: 780; color: #24324f; }
.sp-section-title::before {
  content: ""; width: 11px; height: 11px; border-radius: 999px; background: linear-gradient(135deg, var(--sp-primary), var(--sp-primary-2)); box-shadow: 0 0 14px rgba(75, 168, 255, 0.24);
}
.sp-subtitle { margin: 0 0 14px 22px; color: var(--sp-muted); font-size: 0.92rem; line-height: 1.7; }
.tabs, .tabitem, .tab-nav, .tabitem > div { border-radius: 24px !important; }
.tab-nav {
  gap: 8px !important; margin: 4px 0 14px !important; padding: 6px !important; background: rgba(255,255,255,0.54) !important;
  border: 1px solid rgba(255,255,255,0.88) !important; box-shadow: 0 10px 20px rgba(122, 144, 184, 0.08);
}
.tab-nav button {
  min-height: 48px !important; border-radius: 999px !important; padding: 10px 16px !important; color: #5d6f8a !important;
  background: transparent !important; transition: transform var(--sp-ease), background var(--sp-ease), box-shadow var(--sp-ease), color var(--sp-ease) !important;
}
.tab-nav button:hover { transform: translateY(-1px); color: #27405f !important; background: rgba(255,255,255,0.62) !important; }
.tab-nav button.selected {
  color: #183050 !important; background: linear-gradient(135deg, rgba(75,168,255,0.28), rgba(103,228,210,0.24)) !important; box-shadow: 0 10px 24px rgba(86, 154, 231, 0.16) !important;
}
button, .gradio-button {
  min-height: 48px !important; border-radius: 16px !important; font-weight: 760 !important; letter-spacing: -0.01em !important;
  transition: transform var(--sp-ease), box-shadow var(--sp-ease), background var(--sp-ease), opacity var(--sp-ease) !important;
}
button:hover, .gradio-button:hover { transform: translate3d(0, -2px, 0); }
button:active, .gradio-button:active { transform: translate3d(0, 1px, 0) scale(0.985); }
button:focus-visible, .gradio-button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
  outline: 2px solid rgba(75, 168, 255, 0.75) !important; outline-offset: 2px !important;
}
button.primary, .gradio-button.primary, .sp-blue button, .sp-green button {
  color: #16314f !important; border: 0 !important; background: linear-gradient(135deg, #63b7ff, #74ead8) !important; box-shadow: 0 14px 28px rgba(91, 180, 255, 0.2) !important;
}
.sp-danger button { color: #fff !important; background: linear-gradient(135deg, rgba(255,127,150,0.78), rgba(255,154,168,0.7)) !important; border: 0 !important; }
input, textarea, select, .wrap, .dataframe, .table-wrap, .image-container, .gallery, .gallery > div, .file-preview, .download-button, .accordion { border-radius: 18px !important; }
input, textarea, select { min-height: 48px !important; color: #20314d !important; background: rgba(255,255,255,0.8) !important; border: 1px solid rgba(160, 196, 235, 0.4) !important; }
label, .label-wrap span { color: #4f6484 !important; font-weight: 680 !important; }
.dataframe, .table-wrap, .gallery, .file-preview, .accordion { background: rgba(255,255,255,0.66) !important; border: 1px solid rgba(255,255,255,0.88) !important; }
.dataframe th { background: linear-gradient(180deg, rgba(91, 180, 255, 0.14), rgba(103, 228, 210, 0.08)) !important; color: #28405f !important; font-weight: 760 !important; }
.dataframe td { color: #4d6280 !important; }
.dataframe table { border-collapse: separate !important; border-spacing: 0 8px !important; }
.dataframe tbody tr { background: rgba(255,255,255,0.9) !important; box-shadow: 0 8px 18px rgba(122, 144, 184, 0.08) !important; }
.dataframe tbody td:first-child { border-radius: 12px 0 0 12px !important; }
.dataframe tbody td:last-child { border-radius: 0 12px 12px 0 !important; }
.gradio-container .fullscreen,
.gradio-container [data-fullscreen="true"] {
  position: fixed !important;
  inset: 24px !important;
  width: auto !important;
  height: auto !important;
  max-width: calc(100vw - 48px) !important;
  max-height: calc(100vh - 48px) !important;
  margin: 0 !important;
  padding: 18px !important;
  overflow: auto !important;
  z-index: 9999 !important;
  background: rgba(247, 251, 255, 0.96) !important;
  border: 1px solid rgba(210, 224, 241, 0.95) !important;
  border-radius: 24px !important;
  box-shadow: 0 28px 60px rgba(73, 96, 140, 0.22) !important;
  backdrop-filter: blur(18px) !important;
}
.gradio-container .fullscreen::before,
.gradio-container [data-fullscreen="true"]::before {
  content: "" !important;
  position: fixed !important;
  inset: 0 !important;
  background: rgba(239, 246, 255, 0.72) !important;
  z-index: -1 !important;
}
.gradio-container .fullscreen .image-container,
.gradio-container .fullscreen [data-testid="image"],
.gradio-container .fullscreen img,
.gradio-container [data-fullscreen="true"] .image-container,
.gradio-container [data-fullscreen="true"] [data-testid="image"],
.gradio-container [data-fullscreen="true"] img {
  min-height: unset !important;
  height: auto !important;
  max-height: 82vh !important;
  max-width: 100% !important;
  object-fit: contain !important;
  border-radius: 18px !important;
  box-shadow: none !important;
}
.gradio-container .fullscreen .dataframe,
.gradio-container .fullscreen .table-wrap,
.gradio-container [data-fullscreen="true"] .dataframe,
.gradio-container [data-fullscreen="true"] .table-wrap {
  background: #ffffff !important;
  border: 1px solid rgba(210, 224, 241, 0.9) !important;
  box-shadow: none !important;
  width: 100% !important;
}
.gradio-container .fullscreen .dataframe table,
.gradio-container [data-fullscreen="true"] .dataframe table {
  border-spacing: 0 !important;
  border-collapse: collapse !important;
}
.gradio-container .fullscreen .dataframe tbody tr,
.gradio-container [data-fullscreen="true"] .dataframe tbody tr {
  background: transparent !important;
  box-shadow: none !important;
}
.gradio-container .fullscreen .dataframe tbody td:first-child,
.gradio-container .fullscreen .dataframe tbody td:last-child,
.gradio-container [data-fullscreen="true"] .dataframe tbody td:first-child,
.gradio-container [data-fullscreen="true"] .dataframe tbody td:last-child {
  border-radius: 0 !important;
}
.gradio-container .fullscreen button,
.gradio-container [data-fullscreen="true"] button {
  box-shadow: none !important;
}
#canvas-shell {
  padding: 12px;
  border-radius: 30px;
  background: linear-gradient(145deg, rgba(86,175,255,0.08), rgba(112,234,216,0.06), rgba(255,255,255,0.64));
  border: 1px solid rgba(255,255,255,0.92);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.82), 0 18px 38px rgba(108, 136, 190, 0.12);
}
#canvas-shell iframe { width: 100%; min-height: 500px; border-radius: 28px !important; border: 1px solid rgba(255,255,255,0.95) !important; box-shadow: 0 24px 48px rgba(108, 136, 190, 0.18), 0 0 0 8px rgba(83,178,255,0.06) !important; }
.sp-preview-card .image-container, .sp-preview-card [data-testid="image"], .sp-preview-card img { min-height: 210px !important; }
#workspace-main-row { display: grid !important; grid-template-columns: 1fr; gap: 16px !important; }
#sp-left-panel, #sp-center-panel, #sp-right-panel { min-width: 0 !important; }
#sp-left-panel .sp-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(249,252,255,0.64)) !important;
}
#sp-left-panel .image-container,
#sp-left-panel .gallery,
#sp-left-panel .gallery > div {
  border-color: rgba(255,255,255,0.7) !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-card {
  position: sticky;
  top: 16px;
  z-index: 1;
  background: linear-gradient(180deg, rgba(251,253,255,0.92), rgba(244,249,255,0.76)) !important;
}
#sp-right-panel .sp-subtitle { margin-bottom: 12px; }
#sp-right-panel .sp-param-title {
  margin: 14px 0 8px;
  color: #5a708d;
  font-size: 0.86rem;
  font-weight: 730;
}
#sp-right-panel .sp-flat-select {
  margin-bottom: 12px !important;
}
#sp-right-panel .sp-flat-select > .wrap {
  padding: 10px 12px !important;
  border-radius: 20px !important;
  background: rgba(255,255,255,0.94) !important;
  border: 1px solid rgba(205, 220, 240, 0.92) !important;
  box-shadow: 0 10px 24px rgba(121, 145, 183, 0.08) !important;
}
#sp-right-panel .sp-param-block {
  margin: 12px 0 16px !important;
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-param-row {
  margin: 0 0 8px !important;
  align-items: center !important;
  justify-content: space-between !important;
}
#sp-right-panel .sp-param-name {
  color: #435a77;
  font-size: 0.88rem;
  font-weight: 730;
  line-height: 1.35;
}
#sp-right-panel .sp-value-badge {
  max-width: 84px !important;
  min-width: 72px !important;
}
#sp-right-panel .sp-value-badge input,
#sp-right-panel .sp-value-badge textarea {
  height: 34px !important;
  min-height: 34px !important;
  padding: 6px 10px !important;
  text-align: center !important;
  border-radius: 999px !important;
  border: 1px solid rgba(191, 212, 238, 0.92) !important;
  background: linear-gradient(180deg, rgba(250,253,255,0.96), rgba(239,247,255,0.94)) !important;
  color: #2f5f88 !important;
  font-size: 0.82rem !important;
  font-weight: 760 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-flat-slider {
  margin: 0 !important;
}
#sp-right-panel .sp-flat-slider > .wrap {
  padding: 2px 0 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-flat-slider .block-label,
#sp-right-panel .sp-flat-slider .label-wrap,
#sp-right-panel .sp-flat-slider label,
#sp-right-panel .sp-flat-slider [data-testid="number-input"],
#sp-right-panel .sp-flat-slider input[type="number"],
#sp-right-panel .sp-flat-slider input[type="text"] {
  display: none !important;
}
#sp-right-panel .sp-flat-slider .wrap > div:last-child,
#sp-right-panel .sp-flat-slider .wrap > :last-child {
  min-width: 0 !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"],
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"] {
  width: 100% !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  accent-color: #49b0ff !important;
  height: 12px !important;
  border-radius: 999px !important;
  background: linear-gradient(90deg, rgba(216, 228, 242, 0.95), rgba(226, 235, 247, 0.92)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.92),
    inset 0 0 0 1px rgba(194, 212, 236, 0.62),
    0 8px 18px rgba(123, 147, 186, 0.08) !important;
  transition: transform var(--sp-ease), box-shadow var(--sp-ease), height var(--sp-ease) !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]:hover,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]:hover {
  height: 14px !important;
  transform: translateY(-1px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.95),
    inset 0 0 0 1px rgba(154, 203, 255, 0.78),
    0 12px 22px rgba(95, 163, 240, 0.12) !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]::-webkit-slider-runnable-track,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]::-webkit-slider-runnable-track {
  height: 12px !important;
  border-radius: 999px !important;
  background: transparent !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]::-moz-range-track,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]::-moz-range-track {
  height: 12px !important;
  border-radius: 999px !important;
  background: transparent !important;
  border: 0 !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]::-webkit-slider-thumb,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]::-webkit-slider-thumb {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 24px !important;
  height: 24px !important;
  margin-top: -6px !important;
  border-radius: 999px !important;
  background:
    radial-gradient(circle at 32% 30%, rgba(255,255,255,0.98), rgba(255,255,255,0.8) 52%, rgba(207, 230, 255, 0.88) 100%) !important;
  border: 1px solid rgba(191, 214, 242, 0.96) !important;
  box-shadow:
    0 10px 18px rgba(108, 138, 180, 0.18),
    0 0 0 6px rgba(94, 186, 255, 0.10) !important;
  transition: transform var(--sp-ease), box-shadow var(--sp-ease) !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]:hover::-webkit-slider-thumb,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]:hover::-webkit-slider-thumb {
  transform: scale(1.08) !important;
  box-shadow:
    0 12px 22px rgba(108, 138, 180, 0.22),
    0 0 0 9px rgba(94, 186, 255, 0.12) !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]:active::-webkit-slider-thumb,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]:active::-webkit-slider-thumb {
  transform: scale(1.14) !important;
}
#sp-right-panel .sp-flat-slider input[type=\"range\"]::-moz-range-thumb,
#sp-right-panel .sp-flat-slider .gradio-slider input[type=\"range\"]::-moz-range-thumb {
  width: 24px !important;
  height: 24px !important;
  border: 1px solid rgba(191, 214, 242, 0.96) !important;
  border-radius: 999px !important;
  background:
    radial-gradient(circle at 32% 30%, rgba(255,255,255,0.98), rgba(255,255,255,0.8) 52%, rgba(207, 230, 255, 0.88) 100%) !important;
  box-shadow:
    0 10px 18px rgba(108, 138, 180, 0.18),
    0 0 0 6px rgba(94, 186, 255, 0.10) !important;
}
#sp-right-panel .sp-switch-row {
  margin: 10px 0 !important;
  padding: 10px 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 12px !important;
}
#sp-right-panel .sp-single-switch {
  min-width: 68px !important;
}
#sp-right-panel .sp-single-switch > .wrap {
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-single-switch label {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: flex-end !important;
  gap: 0 !important;
  min-width: 68px !important;
  padding: 0 !important;
}
#sp-right-panel .sp-single-switch input[type="checkbox"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 52px !important;
  height: 30px !important;
  border-radius: 999px !important;
  background: rgba(211, 223, 239, 0.96) !important;
  border: 1px solid rgba(192, 209, 233, 0.92) !important;
  position: relative !important;
  cursor: pointer !important;
  box-shadow: inset 0 1px 2px rgba(111, 138, 178, 0.08) !important;
}
#sp-right-panel .sp-single-switch input[type="checkbox"]::before {
  content: "" !important;
  position: absolute !important;
  top: 3px !important;
  left: 3px !important;
  width: 22px !important;
  height: 22px !important;
  border-radius: 999px !important;
  background: #ffffff !important;
  box-shadow: 0 4px 10px rgba(113, 136, 172, 0.18) !important;
  transition: transform var(--sp-ease), background var(--sp-ease) !important;
}
#sp-right-panel .sp-single-switch input[type="checkbox"]:checked {
  background: linear-gradient(135deg, rgba(86,181,255,0.96), rgba(112,234,216,0.96)) !important;
  border-color: rgba(99, 183, 255, 0.84) !important;
}
#sp-right-panel .sp-single-switch input[type="checkbox"]:checked::before {
  transform: translateX(22px) !important;
}
#sp-right-panel .sp-chip-group {
  margin: 8px 0 14px !important;
}
#sp-right-panel .sp-chip-group > .wrap {
  padding: 10px !important;
  border-radius: 20px !important;
  background: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(205, 220, 240, 0.92) !important;
  box-shadow: 0 10px 24px rgba(121, 145, 183, 0.08) !important;
}
#sp-right-panel .sp-chip-group .wrap label,
#sp-right-panel .sp-chip-group .checkbox-group label {
  margin: 0 8px 8px 0 !important;
  padding: 8px 14px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(198, 217, 240, 0.92) !important;
  background: rgba(246,250,255,0.94) !important;
  color: #507095 !important;
  font-size: 0.84rem !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-chip-group input:checked + span,
#sp-right-panel .sp-chip-group label:has(input:checked) {
  color: #1f4467 !important;
  background: linear-gradient(135deg, rgba(83, 178, 255, 0.16), rgba(112, 234, 216, 0.18)) !important;
  border-color: rgba(117, 193, 255, 0.44) !important;
}
#sp-right-panel .sp-advanced-accordion {
  margin-top: 12px !important;
  border-radius: 22px !important;
  border: 1px solid rgba(206, 221, 240, 0.92) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(247,251,255,0.8)) !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-advanced-accordion .label-wrap,
#sp-right-panel .sp-advanced-accordion .accordion-header,
#sp-right-panel .sp-advanced-accordion button {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#sp-right-panel .sp-advanced-accordion button,
#sp-right-panel .sp-advanced-accordion summary {
  min-height: 46px !important;
  color: #425a79 !important;
  font-weight: 760 !important;
}
#sp-right-panel .accordion {
  background: rgba(255,255,255,0.44) !important;
  border-color: rgba(206, 221, 240, 0.92) !important;
  box-shadow: none !important;
}
#sp-right-panel .gradio-slider input,
#sp-right-panel input[type=\"range\"] {
  accent-color: #5bb8ff !important;
}
#sp-center-panel > .sp-card:first-child,
#sp-center-panel .sp-card:first-child {
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(243,249,255,0.78)) !important;
}
#run-analysis-text textarea { min-height: 260px !important; line-height: 1.7 !important; resize: vertical !important; }
.sp-card, .sp-card-tight, .sp-metric, .sp-hero-card, .sp-showcase-card, .tab-nav button, button, .gradio-button, #canvas-shell iframe {
  transition: transform var(--sp-ease), box-shadow var(--sp-ease), border-color var(--sp-ease), background var(--sp-ease) !important;
}
.sp-card:hover, .sp-card-tight:hover, .sp-metric:hover { transform: translate3d(0, -2px, 0); border-color: var(--sp-border-strong) !important; box-shadow: 0 18px 36px rgba(122, 144, 184, 0.16) !important; }
@keyframes spOrbit { from { transform: rotate(0deg) scale(1); } 50% { transform: rotate(180deg) scale(1.02); } to { transform: rotate(360deg) scale(1); } }
@media (min-width: 640px) {
  .gradio-container { padding: 18px !important; }
  #sp-app-inner { padding: 16px; }
  #sp-topbar { grid-template-columns: 1fr auto; align-items: center; }
  .sp-toolbar { justify-content: flex-end; }
}
@media (min-width: 960px) {
  #workspace-main-row { grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.55fr) minmax(300px, 0.9fr); align-items: start; }
}
@media (max-width: 959px) {
  #sp-topbar { grid-template-columns: 1fr; }
  .sp-toolbar { justify-content: flex-start; }
  .sp-workflow-head { align-items: flex-start; flex-direction: column; }
}
@media (max-width: 640px) {
  .gradio-container { padding: 8px !important; }
  #sp-app-inner { padding: 12px; }
  .sp-brand { grid-template-columns: 48px 1fr; gap: 12px; }
  .sp-logo { width: 44px; height: 44px; font-size: 17px; }
  .sp-workflow-rail { overflow-x: auto; flex-wrap: nowrap; padding-bottom: 2px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

OUTPUT_DIR = "outputs"
COMPOSITE_DIR = os.path.join(OUTPUT_DIR, "composites")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explanations")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
REPORT_RESULT_DIR = os.path.join("report", "results")
CASE_DIR = os.path.join("report", "cases")

CONFIG_PATH = "configs/default.yaml"

for path in [COMPOSITE_DIR, TABLE_DIR, LOG_DIR, EXPLAIN_DIR, MASK_DIR, REPORT_RESULT_DIR, CASE_DIR]:
    os.makedirs(path, exist_ok=True)


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
logger = InferenceLogger(log_dir=LOG_DIR, enable_file_log=cfg.get("output", {}).get("save_log", True))

scorer_cfg = cfg.get("scorer", {})
active_backend = scorer_cfg.get("active_backend", "handin_opa_subprocess")
calibration_cfg = scorer_cfg.get("smartplace_opa_calibrated", {})
libcom_cfg = scorer_cfg.get("libcom_opa_subprocess", {})
runtime_scorer_cache = {}


def _build_calibrated_scorer(backend_key: str) -> SmartPlaceOPACalibratedScorer:
    if backend_key == "handin_opa_subprocess":
        handin_cfg = scorer_cfg.get("handin_opa_subprocess", {})
        base_scorer = HandinOPASubprocessScorer(
            python_path=handin_cfg.get("python_path", "../handin/.venv/Scripts/python.exe"),
            script_path=handin_cfg.get("script_path", "scripts/handin_opa_infer_once.py"),
            batch_script_path=handin_cfg.get("batch_script_path", "scripts/handin_opa_infer_batch.py"),
            handin_root=handin_cfg.get("handin_root", "../handin"),
            weight_path=handin_cfg.get("weight_path", "../handin/experiments/ablation_study/resnet18_w05_20260609_161229/checkpoints/resnet18_w05_best-acc-0.718_epoch15_f1-0.614.pth"),
            device=handin_cfg.get("device", "cpu"),
            model_name=handin_cfg.get("model_name", "resnet"),
            layers=handin_cfg.get("layers", 18),
            width_factor=handin_cfg.get("width_factor", 0.5),
            temp_dir=handin_cfg.get("temp_dir", "outputs/handin_subprocess"),
            timeout_seconds=handin_cfg.get("timeout_seconds", 120),
            logger=logger,
        )
    else:
        base_scorer = LibcomOPASubprocessScorer(
            python_path=libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe"),
            script_path=libcom_cfg.get("script_path", "scripts/libcom_opa_infer_once.py"),
            batch_script_path=libcom_cfg.get("batch_script_path", "scripts/libcom_opa_infer_batch.py"),
            device=libcom_cfg.get("device", "cuda:0"),
            model_type=libcom_cfg.get("model_type", "SimOPA"),
            temp_dir=libcom_cfg.get("temp_dir", "outputs/libcom_subprocess"),
            timeout_seconds=libcom_cfg.get("timeout_seconds", 120),
            logger=logger,
        )

    return SmartPlaceOPACalibratedScorer(
        base_scorer=base_scorer,
        opa_weight=calibration_cfg.get("opa_weight", 0.72),
        geometry_weight=calibration_cfg.get("geometry_weight", 0.14),
        contact_weight=calibration_cfg.get("contact_weight", 0.08),
        support_weight=calibration_cfg.get("support_weight", 0.06),
        out_of_bounds_cap=calibration_cfg.get("out_of_bounds_cap", 0.20),
        logger=logger,
    )


def get_runtime_scorer(backend_key: str | None = None) -> SmartPlaceOPACalibratedScorer:
    key = backend_key or active_backend
    if key not in runtime_scorer_cache:
        runtime_scorer_cache[key] = _build_calibrated_scorer(key)
    return runtime_scorer_cache[key]


scorer = get_runtime_scorer(active_backend)

multi_cfg = scorer_cfg.get("libcom_multimodel", {})
libcom_multimodel = LibcomMultiModelSubprocess(
    python_path=multi_cfg.get("python_path", libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe")),
    script_path=multi_cfg.get("script_path", "scripts/libcom_multi_model_infer.py"),
    device=multi_cfg.get("device", libcom_cfg.get("device", "cuda:0")),
    temp_dir=multi_cfg.get("temp_dir", "outputs/libcom_multimodel"),
    logger=logger,
)

u2net_cfg = cfg.get("u2net", {})
u2net_runner = HandinU2NetSubprocessMatting(
    python_path=u2net_cfg.get("python_path", "../handin/.venv/Scripts/python.exe"),
    script_path=u2net_cfg.get("script_path", "scripts/handin_u2net_infer_once.py"),
    handin_root=u2net_cfg.get("handin_root", "../handin"),
    model_type=u2net_cfg.get("model_type", "u2netp"),
    weight_path=u2net_cfg.get("weight_path", "../handin/u2netp.pth"),
    device=u2net_cfg.get("device", "cpu"),
    threshold=u2net_cfg.get("threshold", 0.5),
    temp_dir=u2net_cfg.get("temp_dir", "outputs/handin_u2net"),
    timeout_seconds=u2net_cfg.get("timeout_seconds", 120),
)


def pil_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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
    export_items = []
    for item in items:
        copied = {}
        for key, value in item.items():
            if key in {"image", "candidate_info"}:
                continue
            copied[key] = json_safe_value(value)
        export_items.append(copied)
    return export_items


def save_candidate_images(results: List[Dict[str, Any]], run_id: str) -> None:
    for item in results:
        image = item.get("image")
        if image is None:
            continue
        cid = item["id"]
        score = float(item["score"])
        filename = f"{run_id}_candidate_{cid}_score_{score:.4f}.png"
        path = os.path.join(COMPOSITE_DIR, filename)
        image.save(path)
        item["composite_path"] = path
        item["saved_path"] = path


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


def build_model_info_text() -> str:
    info = get_runtime_scorer(active_backend).get_model_info()
    lines = [
        f"模型名称：{info.get('model_name')}",
        f"加载状态：{'已加载' if info.get('is_loaded') else '未加载'}",
    ]
    if "source" in info:
        lines.append(f"模型来源：{info.get('source')}")
    if "device" in info:
        lines.append(f"运行设备：{info.get('device')}")
    if "model_type" in info:
        lines.append(f"模型类型：{info.get('model_type')}")
    if "python_path" in info:
        lines.append(f"libcom Python：{info.get('python_path')}")
    if "script_path" in info:
        lines.append(f"单图脚本：{info.get('script_path')}")
    if "batch_script_path" in info:
        lines.append(f"批量脚本：{info.get('batch_script_path')}")
    lines.append("")
    lines.append("说明：当前评分后端为 libcom OPAScoreModel，Web 主环境通过子进程调用 .venv_libcom 中的真实参考模型。")
    return "\n".join(lines)


def build_run_analysis_text(summary: Dict[str, Any], ranked: List[Dict[str, Any]], mask_info: Dict[str, Any], drag_mode: str, explanation_text: str = "", run_warnings: List[str] = None) -> str:
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
    lines.append("【交互方式】")
    lines.append(drag_mode)
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
        lines.append(f"估计背景颜色：{mask_info.get('estimated_background_color')}")
    lines.append("")
    lines.append("【Top-K 推荐解释】")
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            f"Top {rank}：候选 {item['id']}，位置=({item['x']}, {item['y']})，"
            f"分数={item['score']:.4f}，评价={item['label']}。"
        )
        lines.append(f"理由：{item['reason']}")
        lines.append(f"结论：{item.get('conclusion', '')}")
        lines.append("")
    lines.append("【评分点说明】")
    lines.append(
        "复杂交互："
        "一、用户可以在浏览器画布中直接拖动前景物体，并将拖拽位置记录为候选位置。"
        "二、用户可以在画布中缩放前景物体，可以实时观察位置变化。"
        "三、用户可以选择自动候选搜索，系统会自动根据当前指定的参数搜索寻找近似最合理的候选位置。"
        "四、当用户的操作非法或用户提交的背景无可用位置时，系统会提示用户。"
    )
    lines.append("参考模型评分：系统将拖拽候选合成为 composite image + composite mask，并调用 libcom OPAScoreModel 批量评分。")
    lines.append("多工具串联：前景 mask 处理 → 拖拽候选记录 → 图像合成 → OPA 评分 → Top-K 推荐 → 导出结果。")
    if explanation_text:
        lines.append("")
        lines.append(explanation_text)
    if run_warnings:
        lines.append("")
        lines.append("【⚠ 输入警告】")
        for w in run_warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


def build_empty_outputs(message: str = "请先加载拖拽画布并记录候选位置。"):
    empty_df = pd.DataFrame()
    return (
        [],
        [],
        empty_df,
        message,
        [],
        empty_df,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def prepare_drag_canvas(background_image, foreground_image, mask_mode, white_bg_threshold, scale):
    if background_image is None:
        raise gr.Error("请先上传背景图。")
    if foreground_image is None:
        raise gr.Error("请先上传前景图。")

    background = Image.fromarray(background_image).convert("RGB")
    raw_foreground = Image.fromarray(foreground_image).convert("RGBA")

    bg_quality_warn = check_background_quality(background)
    if bg_quality_warn:
        gr.Warning(bg_quality_warn)

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
            output_dir=MASK_DIR,
        )
        mask_info["processed_foreground_path"] = processed_fg_path
        mask_info["mask_path"] = mask_path

    bg_w, bg_h = background.size

    resized_fg = resize_foreground(
        foreground=foreground,
        scale=float(scale),
        bg_width=bg_w,
        bg_height=bg_h,
    )
    fg_w, fg_h = resized_fg.size

    init_x = max(0, (bg_w - fg_w) // 2)
    init_y = max(0, bg_h - fg_h - int(bg_h * 0.08))

    iframe_html = _build_drag_canvas_html(background, foreground, scale, init_x, init_y)

    candidate_points = []
    candidate_df = pd.DataFrame(columns=["候选编号", "x", "y", "scale"])
    drag_mode = "用户通过鼠标在画布中拖动前景物体，并手动记录多个候选位置。"

    return (
        iframe_html,
        mask_preview,
        foreground,
        background,
        foreground,
        mask_info,
        candidate_points,
        candidate_df,
        str(init_x),
        str(init_y),
        str(float(scale)),
        drag_mode,
    )


def add_current_candidate(candidate_points, drag_x, drag_y, drag_scale):
    candidate_points = list(candidate_points or [])
    try:
        x = int(float(drag_x))
        y = int(float(drag_y))
        scale = float(drag_scale)
    except Exception:
        raise gr.Error("当前拖拽坐标无效，请先加载画布并拖动物体。")

    cid = len(candidate_points) + 1
    candidate_points.append({"id": cid, "x": x, "y": y, "scale": scale})
    df = pd.DataFrame([
        {"候选编号": p["id"], "x": p["x"], "y": p["y"], "scale": p["scale"]}
        for p in candidate_points
    ])
    return candidate_points, df


def clear_candidates():
    return [], pd.DataFrame(columns=["候选编号", "x", "y", "scale"])

def update_canvas_scale(bg_state, fg_state, drag_x, drag_y, old_scale, new_scale):
    """当前景缩放比例改变时，实时更新画布，并自动修正越界位置。"""
    if bg_state is None or fg_state is None:
        # 画布未加载，不更新
        return gr.update(), drag_x, drag_y, new_scale

    background = bg_state.convert("RGB")
    foreground = fg_state.convert("RGBA")
    new_scale = float(new_scale)
    old_scale = float(old_scale)

    try:
        cur_x = int(float(drag_x))
        cur_y = int(float(drag_y))
    except (ValueError, TypeError):
        cur_x = 0
        cur_y = 0

    bg_w, bg_h = background.size

    # 计算新缩放下的前景尺寸
    new_fg = resize_foreground(foreground, scale=new_scale, bg_width=bg_w, bg_height=bg_h)
    new_fg_w, new_fg_h = new_fg.size

    # 边界修正
    # 1. 若放大后右/下越界 → 向左/上平移到紧贴边界
    # 2. 若缩小后前景完全在画面外 → 平移到边界内
    adj_x = cur_x
    adj_y = cur_y

    # 右侧越界：x + fg_w > bg_w → 移到 x = bg_w - fg_w
    if adj_x + new_fg_w > bg_w:
        adj_x = bg_w - new_fg_w
    # 左侧越界（负值）
    if adj_x < 0:
        adj_x = 0
    # 下侧越界：y + fg_h > bg_h → 移到 y = bg_h - fg_h
    if adj_y + new_fg_h > bg_h:
        adj_y = bg_h - new_fg_h
    # 上侧越界（负值）
    if adj_y < 0:
        adj_y = 0

    # 如果前景比背景还大（极端情况），居中
    if new_fg_w >= bg_w:
        adj_x = max(0, (bg_w - new_fg_w) // 2)
    if new_fg_h >= bg_h:
        adj_y = max(0, (bg_h - new_fg_h) // 2)

    iframe_html = _build_drag_canvas_html(background, foreground, new_scale, adj_x, adj_y)

    return iframe_html, str(adj_x), str(adj_y), new_scale

def on_background_change(background_image):
    """
    当用户上传或选择背景图时，立即检测是否为纯色/近纯色。
    如果是，弹出错误提示，清空背景图，阻止后续加载画布。
    """
    logger.log(f"[DEBUG]Background image changed, checking quality...")
    if background_image is None:
        logger.log(f"[DEBUG]Background image is None, skip quality check.")
        return background_image

    background = Image.fromarray(background_image).convert("RGB")
    logger.log(f"[DEBUG]Background image quality check triggered.")
    bg_warn = check_background_quality(background)

    if bg_warn:
        logger.log(f"[DEBUG]Background image quality warning: {bg_warn}")
        gr.Warning(bg_warn)
        return None
    logger.log(f"[DEBUG]Background image quality passed.")
    return background_image


def run_auto_search(
        bg_state,
        fg_state,
        mask_info_state,
        candidate_points,
        scale,
        determine_coeff,
        auto_coarse_n,
        auto_coarse_m,
        auto_samples_per_cell,
        auto_fine_a,
        auto_fine_b,
        source_html
):
    """运行自动搜索，将最优位置呈现在画布上并加入候选列表。"""
    if bg_state is None or fg_state is None:
        raise gr.Error('请先点击"加载拖拽画布"。')

    background = bg_state.convert("RGB")
    foreground = fg_state.convert("RGBA")
    scale = float(scale)
    determine_coeff = int(determine_coeff)

    logger.section("[SmartPlace-AutoSearch] Start auto candidate area search")

        # 调用自动搜索接口（参数优先级：显式 > 配置文件 > 默认值）
    search_result = auto_candidate_area_search(
        background=background,
        foreground=foreground,
        scorer=scorer,
        n=int(auto_coarse_n),
        m=int(auto_coarse_m),
        r=int(auto_samples_per_cell),
        a=int(auto_fine_a),
        b=int(auto_fine_b),
        determine_coeff=determine_coeff,
        scale=scale,
        logger=logger,
    )

    best = search_result.get("best")
    top_k = search_result.get("top_k", [])
    summary = search_result.get("search_summary", {})

    if best is None:
        no_cand_warn = check_no_candidates([])
        raise gr.Error(no_cand_warn or "自动搜索未找到有效位置，请检查前景/背景是否已加载。")

    best_x = int(best["x"] or 0)
    best_y = int(best["y"] or 0)
    best_score = best["score"] or 0.0

    logger.log(f"[AutoSearch] Best position: x={best_x}, y={best_y}, score={best_score:.6f}")

        # 重建画布，将前景移到最优位置
    iframe_html = _build_drag_canvas_html(background, foreground, scale, best_x, best_y)
    if iframe_html is None:
        iframe_html = source_html

        # 将 top_k 全部加入候选列表（画布只呈现 #1 位置）
    candidate_points = list(candidate_points or [])
    existing_count = len(candidate_points)

    for idx, item in enumerate(top_k):
        cid = existing_count + idx + 1
        candidate_points.append({
            "id": cid,
            "x": int(item["x"]),
            "y": int(item["y"]),
            "scale": scale,
        })

    candidate_df = pd.DataFrame([
        {"候选编号": p["id"], "x": p["x"], "y": p["y"], "scale": p["scale"]}
        for p in candidate_points
    ])
    if candidate_df is None:
        candidate_df = pd.DataFrame([{"候选编号": 1, "x": 0, "y": 0, "scale": scale}])

    drag_mode = (
        f"自动搜索完成：粗搜索 {summary.get('coarse_grid', '?')} → "
        f"细搜索 {summary.get('fine_grid', '?')}，"
        f"determine_coeff={determine_coeff}，"
        f"最优位置=({best_x}, {best_y})，分数={best_score:.4f}。"
        f"已将 Top-{len(top_k)} 结果加入候选列表。"
    )

    return (
        iframe_html,
        candidate_points,
        candidate_df,
        str(best_x),
        str(best_y),
        str(scale),
        drag_mode,
    )


def _build_drag_canvas_html(background, foreground_rgba, scale, init_x, init_y, bg_warning:Optional[str]=None):
    """
    构建拖拽画布 iframe HTML，前景初始位置为 (init_x, init_y)。
    供 prepare_drag_canvas 和 run_auto_search 共用。
    """
    bg_w, bg_h = background.size

    resized_fg = resize_foreground(
        foreground=foreground_rgba,
        scale=float(scale),
        bg_width=bg_w,
        bg_height=bg_h,
    )
    fg_w, fg_h = resized_fg.size

    canvas_scale = min(980 / bg_w, 680 / bg_h, 1.0)
    canvas_w = int(bg_w * canvas_scale)
    canvas_h = int(bg_h * canvas_scale)
    display_fg_w = int(fg_w * canvas_scale)
    display_fg_h = int(fg_h * canvas_scale)

    bg_url = pil_to_data_url(background)
    fg_url = pil_to_data_url(resized_fg)

    srcdoc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --ink: #172033;
    --muted: #708099;
    --line: rgba(148, 163, 184, 0.26);
    --primary: #5b8def;
    --primary2: #77c8e8;
    --panel: rgba(255,255,255,0.78);
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background:
      radial-gradient(circle at 8% 0%, rgba(119, 200, 232, 0.22), transparent 32%),
      linear-gradient(180deg, #f8fbff, #f3f6fb);
    font-family: Inter, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
    color: var(--ink);
  }}
  .panel {{
    box-sizing: border-box;
    border: 1px solid rgba(255,255,255,0.82);
    padding: 22px;
    border-radius: 28px;
    background:
      radial-gradient(circle at top left, rgba(91,141,239,0.10), transparent 34%),
      linear-gradient(145deg, rgba(255,255,255,0.92), rgba(248,252,255,0.68));
    box-shadow: 0 20px 46px rgba(115,135,176,0.14);
    backdrop-filter: blur(16px) saturate(130%);
  }}
  .titlebar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
  }}
  .title {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 820;
    letter-spacing: -0.02em;
  }}
  .title::before {{
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--primary), var(--primary2));
    box-shadow: 0 0 0 6px rgba(91,141,239,0.10);
  }}
  .tips {{
    font-size: 14px;
    color: var(--muted);
    line-height: 1.65;
    margin-bottom: 18px;
  }}
  .chip {{
    display: inline-flex;
    align-items: center;
    padding: 7px 10px;
    border-radius: 999px;
    background: rgba(91,141,239,0.09);
    color: #3d69c6;
    font-size: 12px;
    font-weight: 760;
    white-space: nowrap;
  }}
  #stage {{
    position: relative;
    width: {canvas_w}px;
    height: {canvas_h}px;
    max-width: 100%;
    border: 1px solid rgba(163,198,235,0.42);
    border-radius: 24px;
    background:
      radial-gradient(circle at 12% 16%, rgba(91,141,239,0.10), transparent 26%),
      radial-gradient(circle at 84% 10%, rgba(119,200,232,0.08), transparent 22%),
      #ffffff;
    overflow: hidden;
    user-select: none;
    touch-action: none;
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,0.96),
      0 22px 42px rgba(113,135,181,0.16),
      0 0 0 10px rgba(91,141,239,0.05);
  }}
  #stage::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.42);
    border-radius: 24px;
  }}
  #bg {{
    position: absolute;
    left: 0;
    top: 0;
    width: {canvas_w}px;
    height: {canvas_h}px;
    pointer-events: none;
  }}
  #fg {{
    position: absolute;
    left: {int(init_x * canvas_scale)}px;
    top: {int(init_y * canvas_scale)}px;
    width: {display_fg_w}px;
    height: {display_fg_h}px;
    cursor: grab;
    touch-action: none;
    filter: drop-shadow(0 18px 24px rgba(15,23,42,0.16));
  }}
  #box {{
    position: absolute;
    left: {int(init_x * canvas_scale)}px;
    top: {int(init_y * canvas_scale)}px;
    width: {display_fg_w}px;
    height: {display_fg_h}px;
    border: 2px solid rgba(91,141,239,0.92);
    border-radius: 18px;
    box-sizing: border-box;
    pointer-events: none;
    box-shadow: 0 0 0 6px rgba(91,141,239,0.12), 0 0 28px rgba(119,200,232,0.24);
  }}
  #status {{
    font-size: 14px;
    color: #465772;
    margin-top: 12px;
    padding: 13px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(148,163,184,0.18);
  }}
</style>
</head>
<body>
<div class="panel">
  <div class="title">拖拽交互画布</div>
  <div class="tips">
    操作：鼠标按住物体拖动；也可以点击背景中的任意位置，将物体中心移动到该处。
    调整好位置后，点击页面下方“记录当前拖拽位置为候选”。
  </div>

      <div id="stage">
        <img id="bg" src="{bg_url}" />
        <img id="fg" src="{fg_url}" />
        <div id="box"></div>
      </div>

      <div id="status">当前位置：x={init_x}, y={init_y}, scale={float(scale):.3f}</div>
    </div>

    <script>
    (function() {{
      const stage = document.getElementById("stage");
      const fg = document.getElementById("fg");
      const box = document.getElementById("box");
      const status = document.getElementById("status");

      const canvasScale = {canvas_scale};
      const stageW = {canvas_w};
      const stageH = {canvas_h};
      const fgW = {display_fg_w};
      const fgH = {display_fg_h};
      const originalFgW = {fg_w};
      const originalFgH = {fg_h};
      const currentScale = {float(scale)};

      let x = {int(init_x * canvas_scale)};
      let y = {int(init_y * canvas_scale)};
      let dragging = false;
      let offsetX = 0;
      let offsetY = 0;

      window.smartplaceLocalDragState = {{
        x: Math.round(x / canvasScale),
        y: Math.round(y / canvasScale),
        scale: currentScale
      }};
      window.getSmartPlaceDragState = function() {{
        return window.smartplaceLocalDragState;
      }};

      function findInputInParent(elemId) {{
        try {{
          const root = parent.document.getElementById(elemId);
          if (!root) return null;
          return root.querySelector("textarea, input");
        }} catch (e) {{
          return null;
        }}
      }}

      function setParentValue(elemId, value) {{
        const input = findInputInParent(elemId);
        if (!input) return;
        input.value = String(value);
        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
      }}

      function clamp() {{
        x = Math.max(0, Math.min(stageW - fgW, x));
        y = Math.max(0, Math.min(stageH - fgH, y));
      }}
      
      function checkPlacementNaturality(ox, oy, fgW, fgH, bgW, bgH) {{
            const bottomY = oy + fgH;
            const bottomRatio = bottomY / bgH;
            const centerX = ox + fgW / 2;
            const centerXRatio = centerX / bgW;
            let hints = [];

            // 悬空检测：物体底部远高于画面底部
            if (bottomRatio < 0.45) {{
              hints.push("物体悬空，建议放在靠近地面的位置");
            }}

            // 过于靠近边缘
            if (centerXRatio < 0.12 || centerXRatio > 0.88) {{
              hints.push("物体过于靠边");
            }}

            // 越界检测
            if (ox < 0 || oy < 0 || ox + fgW > bgW || oy + fgH > bgH) {{
              hints.push("物体越出画面边界");
            }}

            return hints;
          }}

      function update() {{
        clamp();

        fg.style.left = x + "px";
        fg.style.top = y + "px";
        box.style.left = x + "px";
        box.style.top = y + "px";

        const ox = Math.round(x / canvasScale);
        const oy = Math.round(y / canvasScale);

        setParentValue("drag_x_input", ox);
        setParentValue("drag_y_input", oy);
        setParentValue("drag_scale_input", currentScale);

        window.smartplaceLocalDragState = {{
          x: ox,
          y: oy,
          scale: currentScale
        }};

        try {{
          parent.postMessage({{
            type: "smartplace-drag-update",
            x: ox,
            y: oy,
            scale: currentScale
          }}, "*");
        }} catch (e) {{}}
        
        const hints = checkPlacementNaturality(
              ox, oy, originalFgW, originalFgH, stageW / canvasScale, stageH / canvasScale
            );

            let statusText = "当前位置：x=" + ox + ", y=" + oy + ", scale=" + currentScale.toFixed(3);

            if (hints.length > 0) {{
              statusText += "  ⚠ " + hints.join("；");
              status.style.borderColor = "rgba(224,122,122,0.50)";
              status.style.background = "rgba(255,245,245,0.85)";
            }} else {{
              status.style.borderColor = "rgba(148,163,184,0.18)";
              status.style.background = "rgba(255,255,255,0.70)";
            }}

        status.innerText = statusText + "；物体尺寸≈" + originalFgW + "×" + originalFgH;
      }}
      
      function pointerPosition(evt) {{
        const rect = stage.getBoundingClientRect();
        const sx = stageW / rect.width;
        const sy = stageH / rect.height;
        return {{
          x: (evt.clientX - rect.left) * sx,
          y: (evt.clientY - rect.top) * sy
        }};
      }}

      stage.addEventListener("pointerdown", function(evt) {{
        const p = pointerPosition(evt);

        const inside =
          p.x >= x && p.x <= x + fgW &&
          p.y >= y && p.y <= y + fgH;

        if (inside) {{
          offsetX = p.x - x;
          offsetY = p.y - y;
        }} else {{
          x = p.x - fgW / 2;
          y = p.y - fgH / 2;
          offsetX = fgW / 2;
          offsetY = fgH / 2;
        }}

        dragging = true;
        stage.setPointerCapture(evt.pointerId);
        fg.style.cursor = "grabbing";
        update();
        evt.preventDefault();
      }});

      stage.addEventListener("pointermove", function(evt) {{
        if (!dragging) return;
        const p = pointerPosition(evt);
        x = p.x - offsetX;
        y = p.y - offsetY;
        update();
        evt.preventDefault();
      }});

      stage.addEventListener("pointerup", function(evt) {{
        dragging = false;
        fg.style.cursor = "grab";
        update();
        evt.preventDefault();
      }});

      stage.addEventListener("pointercancel", function(evt) {{
        dragging = false;
        fg.style.cursor = "grab";
        update();
      }});

      update();
    }})();
    </script>
    </body>
    </html>
    """

    iframe_html = f"""
    <iframe
      id="smartplace_drag_iframe"
      srcdoc="{html_lib.escape(srcdoc, quote=True)}"
      style="width:100%; height:{canvas_h + 190}px; border:0; border-radius:28px; background:transparent; box-shadow:0 24px 58px rgba(31,41,55,0.13);"
    ></iframe>
    """

    return iframe_html



def as_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"开启", "on", "true", "1", "yes"}


def score_drag_candidates(
    bg_state,
    fg_state,
    mask_info_state,
    candidate_points,
    top_k,
    filter_out_of_bounds,
    enable_explanation,
    enable_saliency,
    enable_feature_analysis,
    occlusion_patch_size,
    occlusion_stride,
    enable_libcom_suite,
    libcom_suite_models,
    lbm_steps,
    lbm_resolution,
    case_name,
    background_note,
    foreground_note,
    manual_label,
    manual_reason,
    drag_mode_state,
    score_backend,
):
    if bg_state is None or fg_state is None:
        raise gr.Error("请先点击“加载拖拽画布”。")

    candidate_points = list(candidate_points or [])
    if not candidate_points:
        raise gr.Error("请至少记录一个拖拽候选位置。")

    background = bg_state.convert("RGB")
    foreground = fg_state.convert("RGBA")
    mask_info = dict(mask_info_state or {})
    top_k = int(top_k)
    filter_out_of_bounds = as_enabled(filter_out_of_bounds)
    enable_explanation = as_enabled(enable_explanation)
    enable_saliency = as_enabled(enable_saliency)
    enable_feature_analysis = as_enabled(enable_feature_analysis)
    enable_libcom_suite = as_enabled(enable_libcom_suite)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    scorer = get_runtime_scorer(score_backend)
    mask_info["score_backend"] = score_backend

    logger.section("[SmartPlace-Drag] Start drag candidate scoring")
    logger.log(f"[Input] background_size={background.size}")
    logger.log(f"[Input] foreground_size={foreground.size}")
    logger.log(f"[Param] recorded_candidates={len(candidate_points)}")
    logger.log(f"[Param] top_k={top_k}")
    logger.log(f"[Param] filter_out_of_bounds={filter_out_of_bounds}")
    logger.log(f"[Param] enable_explanation={enable_explanation}")
    logger.log(f"[Param] enable_saliency={enable_saliency}")
    logger.log(f"[Param] enable_feature_analysis={enable_feature_analysis}")
    logger.log(f"[Param] enable_libcom_suite={enable_libcom_suite}")
    logger.log(f"[Param] libcom_suite_models={list(libcom_suite_models or [])}")

    composites = []
    candidate_infos = []
    candidates = []

    resized_fg_for_check = resize_foreground(
        foreground, scale=float(candidate_points[0]["scale"]),
        bg_width=background.size[0], bg_height=background.size[1],
    ) if candidate_points else None

    if resized_fg_for_check is not None:
        run_warnings = collect_run_warnings(
            fg_width=resized_fg_for_check.size[0],
            fg_height=resized_fg_for_check.size[1],
            bg_width=background.size[0],
            bg_height=background.size[1],
            candidates=None,  # 候选还没合成，暂不检测空候选
        )
    else:
        run_warnings = collect_run_warnings(
            fg_width=0, fg_height=0,
            bg_width=background.size[0], bg_height=background.size[1],
        )

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

    results = []
    for cand, composite, info, score in zip(candidates, composites, candidate_infos, scores):
        analysis = analyze_candidate(info, float(score))
        result = {
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
        }
        results.append(result)

    if not results:
        no_cand_warn = check_no_candidates([])
        if no_cand_warn and no_cand_warn not in run_warnings:
            run_warnings.append(no_cand_warn)
    if run_warnings:
        logger.section("[SmartPlace-Drag] Run Warnings")
        for w in run_warnings:
            logger.log(f"[Warning] {w}")

    ranked_all = assign_relative_labels_in_place(results, top_k=top_k)
    ranked = ranked_all[:top_k]

    if cfg.get("output", {}).get("save_images", True):
        save_candidate_images(results, run_id=run_id)

    summary = summarize_run(results, top_k=top_k)

    explanation_text = ""
    explanation_overlay_path = None
    explanation_saliency_path = None
    explanation_feature_plot_path = None
    explanation_report_path = None
    occlusion_explanation_update = gr.update(visible=False)
    saliency_explanation_update = gr.update(visible=False)
    feature_explanation_update = gr.update(visible=False)
    libcom_suite_text = ""
    libcom_suite_gallery = []
    if (enable_explanation or enable_saliency or enable_feature_analysis) and ranked:
        top1 = ranked[0]
        logger.section("[SmartPlace-Drag] Start explanation for Top-1 candidate")
        if enable_explanation:
            explanation_result = generate_occlusion_heatmap(
                scorer=scorer,
                image=top1["image"],
                candidate_info=top1["candidate_info"],
                patch_size=int(occlusion_patch_size),
                stride=int(occlusion_stride),
                output_dir=EXPLAIN_DIR,
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_overlay_path = explanation_result["overlay_path"]
            explanation_report_path = export_explanation_markdown(
                explanation_result=explanation_result,
                candidate_id=top1["id"],
                output_dir=REPORT_RESULT_DIR,
            )
            explanation_text = explanation_result["explanation"]
            occlusion_explanation_update = gr.update(value=explanation_overlay_path, visible=True)
        if enable_saliency:
            saliency_result = generate_gradient_saliency_map(
                image=top1["image"],
                output_dir=EXPLAIN_DIR,
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_saliency_path = saliency_result["overlay_path"]
            saliency_explanation_update = gr.update(value=explanation_saliency_path, visible=True)
        if enable_feature_analysis:
            feature_result = generate_calibration_feature_plot(
                candidate_info=top1["candidate_info"],
                output_dir=EXPLAIN_DIR,
                prefix=f"drag_candidate_{top1['id']}",
            )
            explanation_feature_plot_path = feature_result["feature_plot_path"]
            feature_explanation_update = gr.update(value=explanation_feature_plot_path, visible=True)

    if enable_libcom_suite and ranked:
        top1 = ranked[0]
        selected_models = list(libcom_suite_models or [])
        logger.section("[SmartPlace-Drag] Start LibCom multi-model suite for Top-1 candidate")
        try:
            suite_output = libcom_multimodel.run(
                background=background,
                foreground=foreground,
                composite=top1["image"],
                composite_mask=top1["candidate_info"]["composite_mask"],
                candidate_info=top1["candidate_info"],
                models=selected_models,
                lbm_steps=int(lbm_steps),
                lbm_resolution=int(lbm_resolution),
                run_id=f"{run_id}_candidate_{top1['id']}",
            )
            libcom_suite_text, libcom_suite_gallery = libcom_multimodel.build_ui_payload(suite_output)
        except Exception as exc:
            libcom_suite_text = f"LibCom 增强模型运行失败：{repr(exc)}"
            libcom_suite_gallery = []

    table_rows = []
    for item in results:
        table_rows.append({
            "候选编号": item["id"],
            "排名": item.get("rank"),
            "x": item["x"],
            "y": item["y"],
            "缩放比例": item["scale"],
            "OPA分数": format_score(item["score"]),
            "评价": item["label"],
            "是否越界": "是" if item["out_of_bounds"] else "否",
            "面积占比": f"{item['area_ratio']:.4f}",
            "推荐理由/失败提示": item["reason"],
            "结论": item["conclusion"],
            "合成图路径": item.get("composite_path"),
        })
    df = pd.DataFrame(table_rows)
    csv_path = os.path.join(TABLE_DIR, f"{run_id}_drag_scores.csv")
    if cfg.get("output", {}).get("save_csv", True):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    gallery_items = []
    for item in results:
        caption = f"候选 {item['id']} | rank={item.get('rank')} | score={item['score']:.4f} | {item['label']}"
        gallery_items.append((item["image"], caption))

    topk_gallery = []
    for rank, item in enumerate(ranked, start=1):
        caption = f"Top {rank} - 候选 {item['id']} | score={item['score']:.4f} | {item['label']}"
        topk_gallery.append((item["image"], caption))

    run_analysis_text = build_run_analysis_text(
        summary=summary,
        ranked=ranked,
        mask_info=mask_info,
        drag_mode=drag_mode_state or "用户拖拽前景物体并记录候选位置。",
        explanation_text=explanation_text,
        run_warnings=run_warnings if run_warnings else None,
    )

    model_info = scorer.get_model_info()
    log_path = logger.get_log_path()
    export_results = make_export_results(results)
    export_ranked = make_export_results(ranked)

    report_path = export_markdown_report(
        results=export_results,
        ranked=export_ranked,
        summary=summary,
        model_info=model_info,
        output_dir=REPORT_RESULT_DIR,
        csv_path=csv_path,
        log_path=log_path,
        explanation_path=explanation_overlay_path,
        explanation_report_path=explanation_report_path,
        run_warnings=run_warnings if run_warnings else None,
    )

    metadata_path = export_result_package_metadata(
        results=export_results,
        ranked=export_ranked,
        summary=summary,
        model_info=model_info,
        output_dir=REPORT_RESULT_DIR,
        csv_path=csv_path,
        log_path=log_path,
        report_path=report_path,
    )

    files = {
        "csv_path": csv_path,
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
    case_record_path = save_case_record(record=case_record, output_dir=CASE_DIR)

    all_case_records = load_case_records(CASE_DIR)
    case_summary_df = summarize_case_records(all_case_records)
    case_summary_csv_path = export_case_summary_csv(records=all_case_records, output_dir=REPORT_RESULT_DIR)
    case_summary_md_path = export_case_summary_markdown(records=all_case_records, output_dir=REPORT_RESULT_DIR)

    logger.log("[SmartPlace-Drag] Inference finished.")
    logger.log(f"[Output] score_table_saved={csv_path}")
    logger.log(f"[Output] report_saved={report_path}")
    logger.log(f"[Output] metadata_saved={metadata_path}")
    logger.log(f"[Output] case_record_saved={case_record_path}")
    logger.log(f"[Output] case_summary_csv={case_summary_csv_path}")
    logger.log(f"[Output] case_summary_md={case_summary_md_path}")
    logger.log(f"[Output] log_file={log_path}")

    return (
        gallery_items,
        topk_gallery,
        df,
        run_analysis_text,
        occlusion_explanation_update,
        saliency_explanation_update,
        feature_explanation_update,
        libcom_suite_text,
        libcom_suite_gallery,
        case_summary_df,
        csv_path,
        log_path,
        report_path,
        metadata_path,
        explanation_report_path,
        case_record_path,
        case_summary_csv_path,
        case_summary_md_path,
    )



APP_THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md",
)

with gr.Blocks(
    title="SmartPlace Studio · 智能物体放置展示平台",
    fill_width=True,
) as demo:
    preset_foregrounds = collect_preset_images(os.path.join("assets", "foregrounds"))
    preset_backgrounds = collect_preset_images(os.path.join("assets", "backgrounds"))
    bg_state = gr.State(value=None)
    fg_state = gr.State(value=None)
    mask_info_state = gr.State(value={})
    candidate_points_state = gr.State(value=[])
    drag_mode_state = gr.State(value="用户通过鼠标在画布中拖动前景物体，并记录多个候选位置。")

    gr.HTML(
        """
<header id="sp-app-frame" role="banner">
  <div id="sp-app-inner">
    <div id="sp-topbar">
      <div class="sp-brand">
        <div class="sp-logo" aria-hidden="true">SP</div>
        <div>
          <h1>SmartPlace Studio</h1>
          <p>Compact AI workbench for object placement scoring</p>
        </div>
      </div>
      <div class="sp-toolbar" aria-label="system toolbar">
        <span class="sp-mode-pill">Demo / Expert</span>
        <span class="sp-pill"><span class="sp-pill-dot"></span>Local Inference Ready</span>
        <span class="sp-pill">Libcom OPA</span>
        <span class="sp-toolbar-btn">Export Report</span>
      </div>
    </div>

    <section id="sp-hero-grid" aria-label="workflow status">
      <div class="sp-workflow-card">
        <div class="sp-workflow-head">
          <span class="sp-workflow-title">Workflow Status</span>
          <span class="sp-workflow-note">Upload, drag, score, explain, export.</span>
        </div>
        <div class="sp-workflow-rail">
          <span class="sp-work-step is-active">Upload Assets</span>
          <span class="sp-work-arrow">&rarr;</span>
          <span class="sp-work-step is-active">Drag Candidate</span>
          <span class="sp-work-arrow">&rarr;</span>
          <span class="sp-work-step">Score</span>
          <span class="sp-work-arrow">&rarr;</span>
          <span class="sp-work-step">Explain</span>
          <span class="sp-work-arrow">&rarr;</span>
          <span class="sp-work-step">Export</span>
        </div>
      </div>
    </section>
  </div>
</header>
        """
    )

    gr.HTML(
        """
<div class="sp-metric-row">
  <div class="sp-metric"><div class="label">核心交互</div><div class="value">拖拽生成候选</div></div>
  <div class="sp-metric"><div class="label">模型后端</div><div class="value">libcom OPA 评分</div></div>
  <div class="sp-metric"><div class="label">演示产物</div><div class="value">Top-K + 表格 + 解释</div></div>
  <div class="sp-metric"><div class="label">项目证据</div><div class="value">日志 / 报告 / 案例库</div></div>
</div>
        """
    )

    gr.HTML("")

    with gr.Tabs(selected="workspace"):
        with gr.Tab("01 · 创作工作台", id="workspace"):
            with gr.Row(equal_height=False, elem_id="workspace-main-row"):
                with gr.Column(scale=3, min_width=400, elem_id="sp-left-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>前景图片区域</div><div class='sp-subtitle'>预制前景图片 / 本地上传</div>")
                        foreground_input = gr.Image(label="前景物体 Foreground", type="numpy", height=240)
                        gr.Examples(
                            examples=preset_foregrounds,
                            inputs=foreground_input,
                            label="预制前景图片",
                        )
                        gr.HTML("<div class='sp-section-title'>背景图片区域</div><div class='sp-subtitle'>预制背景图片 / 本地上传</div>")
                        background_input = gr.Image(label="背景图 Background", type="numpy", height=240)
                        gr.Examples(
                            examples=preset_backgrounds,
                            inputs=background_input,
                            label="预制背景图片",
                        )


                    case_name_input = gr.State("SmartPlace 工作台案例")
                    background_note_input = gr.State("")
                    foreground_note_input = gr.State("")
                    manual_label_input = gr.State("未填写")
                    manual_reason_input = gr.State("")

                with gr.Column(scale=6, min_width=620, elem_id="sp-center-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>交互画布</div><div class='sp-subtitle'>在画布中拖动物体，记录多个候选位置后再统一评分。</div>")
                        drag_canvas_html = gr.HTML(label="拖拽画布", elem_id="canvas-shell")
                        with gr.Row():
                            load_canvas_button = gr.Button("加载 / 刷新拖拽画布", variant="primary", elem_classes=["sp-blue"])
                            add_candidate_button = gr.Button("记录当前位置为候选", variant="primary", elem_classes=["sp-blue"])
                        with gr.Row():
                            clear_candidate_button = gr.Button("清空候选", variant="primary", elem_classes=["sp-blue"])
                            auto_search_button = gr.Button("🔍 自动搜索最优位置", variant="primary", elem_classes=["sp-green"])
                        with gr.Row():
                            score_button = gr.Button("批量评分并生成结果", variant="primary", elem_classes=["sp-blue"])

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, elem_classes=["sp-card-tight", "sp-preview-card"]):
                            gr.HTML("<div class='sp-section-title'>Mask 预览</div><div class='sp-subtitle'>加载画布后自动显示前景分割结果。</div>")
                            mask_preview_output = gr.Image(label="Mask", type="pil", height=190)
                        with gr.Column(scale=1, elem_classes=["sp-card-tight", "sp-preview-card"]):
                            gr.HTML("<div class='sp-section-title'>处理后前景</div><div class='sp-subtitle'>去底后的前景物体会显示在这里。</div>")
                            processed_foreground_output = gr.Image(label="Foreground", type="pil", height=190)

                    with gr.Group(elem_classes=["sp-card-tight"]):
                        gr.HTML("<div class='sp-section-title'>已记录候选</div>")
                        candidate_points_table = gr.Dataframe(label="候选位置表", wrap=True)

                    with gr.Group(elem_classes=["sp-card-tight"]):
                        gr.HTML("<div class='sp-section-title'>调试坐标</div>")
                        with gr.Row():
                            drag_x_input = gr.Textbox(label="当前 x", value="0", elem_id="drag_x_input")
                            drag_y_input = gr.Textbox(label="当前 y", value="0", elem_id="drag_y_input")
                            drag_scale_input = gr.Textbox(label="当前 scale", value="0.25", elem_id="drag_scale_input")

                with gr.Column(scale=3, min_width=400, elem_id="sp-right-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>模型与参数</div><div class='sp-subtitle'>参数越少越适合演示，更多选项已收纳。</div>")
                        gr.HTML("<div class='sp-param-title'>评分后端</div>")
                        score_backend_input = gr.Dropdown(
                            choices=[
                                ("handin OPA + SmartPlace 校准", "handin_opa_subprocess"),
                                ("libcom OPA + SmartPlace 校准", "libcom_opa_subprocess"),
                            ],
                            value=active_backend,
                            show_label=False,
                            container=False,
                            elem_classes=["sp-flat-select"],
                        )
                        gr.HTML("<div class='sp-param-title'>前景处理模式</div>")
                        mask_mode_input = gr.Dropdown(
                            choices=["自动判断", "透明 PNG Alpha", "浅色/纯色背景去除", "U2Net 自动抠图", "不处理"],
                            value=cfg.get("mask_processor", {}).get("default_mode", "自动判断"),
                            show_label=False,
                            container=False,
                            elem_classes=["sp-flat-select"],
                        )
                        with gr.Group(elem_classes=["sp-param-block"]):
                            with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                gr.HTML("<div class='sp-param-name'>去底阈值</div>")
                                white_bg_threshold_display = gr.Textbox(value=format_param_value(cfg.get('mask_processor', {}).get('white_bg_threshold', 38)), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                            white_bg_threshold_input = gr.Slider(minimum=10, maximum=100, value=cfg.get("mask_processor", {}).get("white_bg_threshold", 38), step=2, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                        with gr.Group(elem_classes=["sp-param-block"]):
                            with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                gr.HTML("<div class='sp-param-name'>前景缩放比例</div>")
                                scale_display = gr.Textbox(value=format_param_value(0.25, 2), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                            scale_input = gr.Slider(minimum=0.05, maximum=0.8, value=0.25, step=0.05, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                        with gr.Group(elem_classes=["sp-param-block"]):
                            with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                gr.HTML("<div class='sp-param-name'>Top-K 数量</div>")
                                top_k_display = gr.Textbox(value=format_param_value(3), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                            top_k_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                        with gr.Group(elem_classes=["sp-switch-row"]):
                            gr.HTML("<div class='sp-param-name'>过滤越界候选</div>")
                            filter_out_of_bounds_input = gr.Checkbox(value=True, show_label=False, container=False, elem_classes=["sp-single-switch"])
                        with gr.Group(elem_classes=["sp-switch-row"]):
                            gr.HTML("<div class='sp-param-name'>实验热力图</div>")
                            enable_explanation_input = gr.Checkbox(value=False, show_label=False, container=False, elem_classes=["sp-single-switch"])
                        with gr.Group(elem_classes=["sp-switch-row"]):
                            gr.HTML("<div class='sp-param-name'>梯度显著图</div>")
                            enable_saliency_input = gr.Checkbox(value=False, show_label=False, container=False, elem_classes=["sp-single-switch"])
                        with gr.Group(elem_classes=["sp-switch-row"]):
                            gr.HTML("<div class='sp-param-name'>中间特征分析</div>")
                            enable_feature_analysis_input = gr.Checkbox(value=False, show_label=False, container=False, elem_classes=["sp-single-switch"])
                        with gr.Group(elem_classes=["sp-switch-row"]):
                            gr.HTML("<div class='sp-param-name'>启用 LibCom 增强模型</div>")
                            enable_libcom_suite_input = gr.Checkbox(value=False, show_label=False, container=False, elem_classes=["sp-single-switch"])
                        gr.HTML("<div class='sp-param-title'>增强模型</div>")
                        libcom_suite_models_input = gr.CheckboxGroup(
                            choices=["fopa", "fos", "harmony", "pctnet", "lbm"],
                            value=["fopa", "fos", "harmony"],
                            show_label=False,
                            container=False,
                            elem_classes=["sp-chip-group"],
                        )
                        with gr.Accordion("LibCom 高级参数", open=False, elem_classes=["sp-advanced-accordion"]):
                            with gr.Group(elem_classes=["sp-param-block"]):
                                with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                    gr.HTML("<div class='sp-param-name'>LBM 步数</div>")
                                    lbm_steps_display = gr.Textbox(value=format_param_value(4), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                                lbm_steps_input = gr.Slider(minimum=1, maximum=8, value=4, step=1, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                            with gr.Group(elem_classes=["sp-param-block"]):
                                with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                    gr.HTML("<div class='sp-param-name'>LBM 分辨率</div>")
                                    lbm_resolution_display = gr.Textbox(value=format_param_value(768), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                                lbm_resolution_input = gr.Slider(minimum=512, maximum=1024, value=768, step=128, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                        with gr.Accordion("高级解释图参数", open=False, elem_classes=["sp-advanced-accordion"]):
                            with gr.Group(elem_classes=["sp-param-block"]):
                                with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                    gr.HTML("<div class='sp-param-name'>遮挡块大小</div>")
                                    occlusion_patch_size_display = gr.Textbox(value=format_param_value(96), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                                occlusion_patch_size_input = gr.Slider(minimum=48, maximum=160, value=96, step=16, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])
                            with gr.Group(elem_classes=["sp-param-block"]):
                                with gr.Row(elem_classes=["sp-param-row"], equal_height=True):
                                    gr.HTML("<div class='sp-param-name'>遮挡滑动步长</div>")
                                    occlusion_stride_display = gr.Textbox(value=format_param_value(96), show_label=False, container=False, interactive=False, elem_classes=["sp-value-badge"])
                                occlusion_stride_input = gr.Slider(minimum=32, maximum=128, value=96, step=16, show_label=False, container=False, buttons=[], elem_classes=["sp-flat-slider"])

        with gr.Tab("02 · 结果仪表盘", id="results"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>Top-K 推荐结果</div><div class='sp-subtitle'>系统认为最适合放置的位置会显示在这里。</div>")
                    topk_gallery = gr.Gallery(label="Top-K 推荐图", columns=3, height="auto")
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>全部候选合成图</div><div class='sp-subtitle'>用于对比模型排序是否合理。</div>")
                    candidate_gallery = gr.Gallery(label="全部候选合成图", columns=3, height="auto")

            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>候选评分表</div><div class='sp-subtitle'>包含排名、OPA 分数、面积占比、越界状态和推荐理由。</div>")
                    score_table = gr.Dataframe(label="候选评分表", wrap=True)
                with gr.Column(scale=5, elem_classes=["sp-card"]):
                    run_analysis_text = gr.Textbox(label="分析说明", lines=14, max_lines=14, elem_id="run-analysis-text")

        with gr.Tab("03 · 解释与案例库", id="analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>模型解释图</div><div class='sp-subtitle'>开启后会基于 Top-1 候选生成遮挡实验热力图、梯度显著性图和中间特征分析图。</div>")
                    with gr.Row(equal_height=False):
                        occlusion_explanation_image = gr.Image(label="遮挡实验热力图", type="filepath", height=260, visible=False)
                        saliency_explanation_image = gr.Image(label="梯度显著性图", type="filepath", height=260, visible=False)
                    feature_explanation_image = gr.Image(label="中间特征分析图", type="filepath", height=300, visible=False)
                    gr.HTML("<div class='sp-section-title'>LibCom 增强模型</div><div class='sp-subtitle'>可选运行 FOPA、FOS、HarmonyScore、PCTNet、LBM，对 Top-1 候选做进一步评估与协调。</div>")
                    libcom_suite_gallery = gr.Gallery(label="FOPA 热力图 / 协调结果", columns=1, height="auto")
                    libcom_suite_text = gr.Textbox(label="多模型结果摘要", lines=12)
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>案例汇总</div><div class='sp-subtitle'>所有评分案例会自动沉淀，方便报告展示。</div>")
                    case_summary_table = gr.Dataframe(label="所有已记录案例汇总", wrap=True)

        with gr.Tab("04 · 导出与复现", id="exports"):
            with gr.Column(elem_classes=["sp-card", "sp-file-row"]):
                gr.HTML("<div class='sp-section-title'>导出文件中心</div><div class='sp-subtitle'>每次评分后自动生成评分表、日志、运行报告、元信息和案例记录。</div>")
                with gr.Row():
                    csv_file = gr.File(label="评分 CSV")
                    log_file = gr.File(label="推理日志")
                    report_file = gr.File(label="Markdown 运行报告")
                with gr.Row():
                    metadata_file = gr.File(label="JSON 元信息")
                    explanation_report_file = gr.File(label="模型解释 Markdown 报告")
                    case_record_file = gr.File(label="单次案例 JSON")
                with gr.Row():
                    case_summary_csv_file = gr.File(label="测试案例汇总 CSV")
                    case_summary_md_file = gr.File(label="测试案例汇总 Markdown")

    white_bg_threshold_input.change(lambda value: format_param_value(value), inputs=white_bg_threshold_input, outputs=white_bg_threshold_display)
    scale_input.change(lambda value: format_param_value(value, 2), inputs=scale_input, outputs=scale_display)
    top_k_input.change(lambda value: format_param_value(value), inputs=top_k_input, outputs=top_k_display)
    lbm_steps_input.change(lambda value: format_param_value(value), inputs=lbm_steps_input, outputs=lbm_steps_display)
    lbm_resolution_input.change(lambda value: format_param_value(value), inputs=lbm_resolution_input, outputs=lbm_resolution_display)
    occlusion_patch_size_input.change(lambda value: format_param_value(value), inputs=occlusion_patch_size_input, outputs=occlusion_patch_size_display)
    occlusion_stride_input.change(lambda value: format_param_value(value), inputs=occlusion_stride_input, outputs=occlusion_stride_display)

    load_canvas_button.click(
        fn=prepare_drag_canvas,
        inputs=[background_input, foreground_input, mask_mode_input, white_bg_threshold_input, scale_input],
        outputs=[
            drag_canvas_html,
            mask_preview_output,
            processed_foreground_output,
            bg_state,
            fg_state,
            mask_info_state,
            candidate_points_state,
            candidate_points_table,
            drag_x_input,
            drag_y_input,
            drag_scale_input,
            drag_mode_state,
        ],
    )

    add_candidate_button.click(
        fn=add_current_candidate,
        inputs=[candidate_points_state, drag_x_input, drag_y_input, drag_scale_input],
        outputs=[candidate_points_state, candidate_points_table],
        js=RECORD_CANDIDATE_JS,
    )

    clear_candidate_button.click(
        fn=clear_candidates,
        inputs=[],
        outputs=[candidate_points_state, candidate_points_table],
    )

    auto_search_button.click(
        fn=run_auto_search,
        inputs=[
            bg_state,
            fg_state,
            mask_info_state,
            candidate_points_state,
            scale_input,
            # determine_coeff_input,
            # auto_coarse_n_input,
            # auto_coarse_m_input,
            # auto_samples_per_cell_input,
            # auto_fine_a_input,
            # auto_fine_b_input,
            # drag_canvas_html,
        ],
        outputs=[
            drag_canvas_html,
            candidate_points_state,
            candidate_points_table,
            drag_x_input,
            drag_y_input,
            drag_scale_input,
            drag_mode_state,
        ],
    )

    score_button.click(
        fn=score_drag_candidates,
        inputs=[
            bg_state,
            fg_state,
            mask_info_state,
            candidate_points_state,
            top_k_input,
            filter_out_of_bounds_input,
            enable_explanation_input,
            enable_saliency_input,
            enable_feature_analysis_input,
            occlusion_patch_size_input,
            occlusion_stride_input,
            enable_libcom_suite_input,
            libcom_suite_models_input,
            lbm_steps_input,
            lbm_resolution_input,
            case_name_input,
            background_note_input,
            foreground_note_input,
            manual_label_input,
            manual_reason_input,
            drag_mode_state,
            score_backend_input,
        ],
        outputs=[
            candidate_gallery,
            topk_gallery,
            score_table,
            run_analysis_text,
            occlusion_explanation_image,
            saliency_explanation_image,
            feature_explanation_image,
            libcom_suite_text,
            libcom_suite_gallery,
            case_summary_table,
            csv_file,
            log_file,
            report_file,
            metadata_file,
            explanation_report_file,
            case_record_file,
            case_summary_csv_file,
            case_summary_md_file,
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css=APP_CSS, js=CUSTOM_DRAG_JS)
