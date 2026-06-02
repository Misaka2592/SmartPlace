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

from models.libcom_opa_subprocess_scorer import LibcomOPASubprocessScorer
from models.libcom_multimodel_subprocess import LibcomMultiModelSubprocess
from utils.composer import compose_image_with_mask, resize_foreground
from utils.scoring import format_score, analyze_candidate, summarize_run
from utils.logger import InferenceLogger
from utils.exporter import export_markdown_report, export_result_package_metadata
from utils.explain import generate_occlusion_heatmap, export_explanation_markdown
from utils.mask_processor import process_foreground_for_composition, save_processed_foreground
from utils.case_manager import (
    build_case_record,
    save_case_record,
    load_case_records,
    summarize_case_records,
    export_case_summary_csv,
    export_case_summary_markdown,
)


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
  --sp-bg: #f6f8fc;
  --sp-panel: rgba(255, 255, 255, 0.78);
  --sp-panel-strong: rgba(255, 255, 255, 0.94);
  --sp-ink: #172033;
  --sp-muted: #728098;
  --sp-primary: #5b8def;
  --sp-primary-2: #77c8e8;
  --sp-mint: #8fd8c6;
  --sp-danger: #e07a7a;
  --sp-success: #48b99f;
  --sp-radius-xl: 30px;
  --sp-radius-lg: 22px;
  --sp-radius-md: 16px;
  --sp-shadow-float: 0 24px 70px rgba(31, 41, 55, 0.10);
  --sp-shadow-card: 0 14px 38px rgba(31, 41, 55, 0.075);
}
html, body { background: var(--sp-bg) !important; }
.gradio-container {
  max-width: 1540px !important;
  margin: 0 auto !important;
  padding: 22px 24px 38px !important;
  color: var(--sp-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
  background:
    radial-gradient(circle at 8% 4%, rgba(119, 200, 232, 0.32) 0, rgba(119, 200, 232, 0.0) 28%),
    radial-gradient(circle at 92% 8%, rgba(200, 199, 244, 0.36) 0, rgba(200, 199, 244, 0.0) 30%),
    radial-gradient(circle at 50% 96%, rgba(143, 216, 198, 0.22) 0, rgba(143, 216, 198, 0.0) 38%),
    linear-gradient(180deg, #f8fbff 0%, #f5f7fb 48%, #f6f8fc 100%) !important;
}
#sp-app-frame {
  position: relative;
  border-radius: 36px;
  padding: 1px;
  margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(91, 141, 239, 0.26), rgba(255,255,255,0.65), rgba(119, 200, 232, 0.22));
  box-shadow: var(--sp-shadow-float);
  overflow: hidden;
}
#sp-app-frame::before {
  content: "";
  position: absolute;
  inset: 1px;
  border-radius: 35px;
  background: rgba(255, 255, 255, 0.48);
  backdrop-filter: blur(20px);
  pointer-events: none;
}
#sp-app-inner { position: relative; border-radius: 35px; padding: 26px 28px 24px; overflow: hidden; }
#sp-topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 24px; }
.sp-brand { display: flex; align-items: center; gap: 14px; }
.sp-logo {
  width: 48px; height: 48px; border-radius: 16px;
  background: linear-gradient(135deg, #ffffff 0%, #eaf3ff 35%, #dff7ff 100%);
  border: 1px solid rgba(255,255,255,0.86);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.95), 0 14px 28px rgba(91,141,239,0.18);
  display: grid; place-items: center; color: var(--sp-primary); font-weight: 900; letter-spacing: -0.08em; font-size: 21px;
}
.sp-brand h1 { margin: 0; font-size: 25px; letter-spacing: -0.04em; color: #142033; line-height: 1.05; }
.sp-brand p { margin: 6px 0 0; color: var(--sp-muted); font-size: 13.5px; }
.sp-status-strip { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 9px; }
.sp-pill {
  display: inline-flex; align-items: center; gap: 7px; padding: 9px 12px; border-radius: 999px;
  color: #334155; background: rgba(255,255,255,0.70); border: 1px solid rgba(148,163,184,0.22);
  box-shadow: 0 8px 20px rgba(31,41,55,0.045); font-size: 12.5px; font-weight: 650; white-space: nowrap;
}
.sp-pill-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--sp-success); box-shadow: 0 0 0 4px rgba(72,185,159,0.13); }
#sp-hero-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr); gap: 20px; align-items: stretch; }
.sp-hero-card, .sp-showcase-card {
  position: relative; border-radius: 30px;
  background: linear-gradient(145deg, rgba(255,255,255,0.80), rgba(255,255,255,0.54));
  border: 1px solid rgba(255,255,255,0.74); box-shadow: var(--sp-shadow-card); overflow: hidden;
}
.sp-hero-card { padding: 34px 36px; }
.sp-hero-card::after {
  content: ""; position: absolute; width: 360px; height: 360px; right: -110px; top: -140px; border-radius: 999px;
  background: radial-gradient(circle, rgba(91,141,239,0.18), rgba(119,200,232,0.04) 58%, transparent 70%);
}
.sp-kicker {
  display: inline-flex; align-items: center; padding: 7px 12px; border-radius: 999px;
  background: rgba(91, 141, 239, 0.10); color: #3d69c6; font-weight: 760; font-size: 12px;
  letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 16px;
}
.sp-hero-card h2 { margin: 0; max-width: 740px; font-size: 42px; line-height: 1.08; letter-spacing: -0.06em; color: #101928; }
.sp-hero-card h2 span { background: linear-gradient(110deg, #3d69c6, #2aa9c8 55%, #54b99f); -webkit-background-clip: text; color: transparent; }
.sp-hero-card p { max-width: 850px; margin: 16px 0 0; font-size: 15.5px; line-height: 1.85; color: #5f6f87; }
.sp-hero-actions { display: flex; flex-wrap: wrap; gap: 11px; margin-top: 22px; }
.sp-soft-tag { padding: 9px 12px; border-radius: 14px; background: rgba(255,255,255,0.72); border: 1px solid rgba(148,163,184,0.20); color: #41516a; font-size: 13px; box-shadow: 0 8px 20px rgba(31,41,55,0.04); }
.sp-showcase-card { padding: 24px; display: flex; flex-direction: column; justify-content: space-between; min-height: 280px; }
.sp-score-orbit {
  height: 150px; border-radius: 28px;
  background: radial-gradient(circle at 32% 30%, rgba(91,141,239,0.26) 0 12%, transparent 13%), radial-gradient(circle at 70% 62%, rgba(119,200,232,0.26) 0 16%, transparent 17%), linear-gradient(145deg, rgba(246,250,255,0.90), rgba(255,255,255,0.62));
  border: 1px solid rgba(255,255,255,0.78); box-shadow: inset 0 1px 0 rgba(255,255,255,0.92), 0 18px 34px rgba(91,141,239,0.12); position: relative; overflow: hidden;
}
.sp-score-orbit::after { content: "Top-K"; position: absolute; right: 20px; bottom: 18px; color: #3d69c6; font-weight: 850; font-size: 28px; letter-spacing: -0.04em; }
.sp-showcase-card h3 { margin: 20px 0 6px; font-size: 18px; letter-spacing: -0.03em; color: #172033; }
.sp-showcase-card p { margin: 0; color: var(--sp-muted); line-height: 1.7; font-size: 13.5px; }
.sp-metric-row { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin: 18px 0 22px; }
.sp-metric {
  border-radius: 22px; padding: 17px 18px; background: rgba(255,255,255,0.68); border: 1px solid rgba(255,255,255,0.76);
  box-shadow: var(--sp-shadow-card); position: relative; overflow: hidden;
}
.sp-metric::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: linear-gradient(90deg, rgba(91,141,239,0.68), rgba(119,200,232,0.26)); }
.sp-metric .label { color: var(--sp-muted); font-size: 12px; margin-bottom: 9px; font-weight: 650; }
.sp-metric .value { color: #172033; font-weight: 820; font-size: 17px; letter-spacing: -0.02em; }
.sp-card, .sp-card-tight {
  border-radius: var(--sp-radius-lg) !important; background: var(--sp-panel) !important; border: 1px solid rgba(255,255,255,0.78) !important;
  box-shadow: var(--sp-shadow-card) !important; backdrop-filter: blur(18px) saturate(140%); position: relative; overflow: hidden;
}
.sp-card { padding: 20px !important; }
.sp-card-tight { padding: 14px !important; }
.sp-card::before, .sp-card-tight::before { content: ""; position: absolute; inset: 0; pointer-events: none; border-radius: inherit; background: linear-gradient(135deg, rgba(255,255,255,0.55), transparent 36%, rgba(119,200,232,0.07)); }
.sp-section-title { position: relative; display: flex; align-items: center; gap: 10px; margin: 0 0 7px 0; color: #172033; font-weight: 830; font-size: 16px; letter-spacing: -0.02em; }
.sp-section-title::before { content: ""; width: 9px; height: 9px; border-radius: 999px; background: linear-gradient(135deg, var(--sp-primary), var(--sp-primary-2)); box-shadow: 0 0 0 5px rgba(91,141,239,0.11); flex: none; }
.sp-subtitle { position: relative; margin: 0 0 16px 19px; color: var(--sp-muted); font-size: 13px; line-height: 1.65; }
.sp-mini-guide { display: grid; gap: 10px; margin-bottom: 15px; }
.sp-guide-step { display: grid; grid-template-columns: 34px 1fr; gap: 10px; align-items: center; padding: 11px; border-radius: 16px; background: rgba(255,255,255,0.58); border: 1px solid rgba(148,163,184,0.17); }
.sp-guide-num { width: 34px; height: 34px; border-radius: 12px; display: grid; place-items: center; background: linear-gradient(135deg, #eaf3ff, #e9fbff); color: #3d69c6; font-weight: 850; box-shadow: inset 0 1px 0 rgba(255,255,255,0.9); }
.sp-guide-step b { display: block; font-size: 13.5px; color: #26364f; }
.sp-guide-step span { display: block; margin-top: 3px; color: var(--sp-muted); font-size: 12px; }
.tabs, .tabitem, .tab-nav, .tabitem > div { border-radius: 22px !important; }
.tab-nav button { border-radius: 999px !important; padding: 10px 17px !important; font-weight: 720 !important; color: #5c6b82 !important; }
.tab-nav button.selected { background: rgba(255,255,255,0.80) !important; color: #2f5fb8 !important; box-shadow: 0 10px 26px rgba(91,141,239,0.14) !important; }
button, .gradio-button { border-radius: 15px !important; font-weight: 760 !important; letter-spacing: -0.01em !important; min-height: 42px !important; transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important; }
button:hover, .gradio-button:hover { transform: translateY(-1px); filter: brightness(1.01); }
button.primary, .gradio-button.primary { background: linear-gradient(135deg, #5b8def 0%, #65bfe0 100%) !important; border: 0 !important; color: #ffffff !important; box-shadow: 0 14px 28px rgba(91, 141, 239, 0.25) !important; }
.sp-blue button, .sp-green button { border: 0 !important; color: white !important; box-shadow: 0 14px 30px rgba(91, 141, 239, 0.22) !important; }
.sp-blue button { background: linear-gradient(135deg, #5b8def, #77c8e8) !important; }
.sp-green button { background: linear-gradient(135deg, #48b99f, #77c8e8) !important; }
.sp-danger button { background: rgba(255,255,255,0.74) !important; color: #c45858 !important; border: 1px solid rgba(224,122,122,0.26) !important; box-shadow: 0 10px 22px rgba(224,122,122,0.08) !important; }
textarea, input, select, .wrap, .dataframe, .table-wrap { border-radius: 15px !important; }
label, .label-wrap span { color: #42526a !important; font-weight: 650 !important; }
input, textarea { background: rgba(255,255,255,0.78) !important; border-color: rgba(148,163,184,0.24) !important; }
.block, .form, .form > div { border-radius: var(--sp-radius-md) !important; }
.image-container, .gallery, .gallery > div, .file-preview, .download-button { border-radius: 18px !important; }
.dataframe table { font-size: 13px !important; }
.dataframe th { background: rgba(239,246,255,0.78) !important; color: #334155 !important; font-weight: 750 !important; }
.dataframe td { color: #475569 !important; }
.accordion { border-radius: 16px !important; border-color: rgba(148,163,184,0.18) !important; background: rgba(255,255,255,0.44) !important; }
#canvas-shell iframe { border-radius: 24px !important; box-shadow: 0 22px 56px rgba(31, 41, 55, 0.12) !important; }
.sp-file-row .file-preview, .sp-file-row .download-button { border-radius: 18px !important; }

/* Comfort display refresh: bigger, airier, easier to present */
.gradio-container {
  max-width: 1680px !important;
  padding: 30px 34px 52px !important;
}
#sp-app-inner {
  padding: 34px 38px 34px !important;
}
#sp-topbar {
  margin-bottom: 30px !important;
}
.sp-logo {
  width: 58px !important;
  height: 58px !important;
  border-radius: 20px !important;
  font-size: 25px !important;
}
.sp-brand h1 {
  font-size: 32px !important;
  letter-spacing: -0.03em !important;
}
.sp-brand p {
  font-size: 15px !important;
}
#sp-hero-grid {
  grid-template-columns: minmax(0, 1.55fr) minmax(360px, 0.45fr) !important;
  gap: 26px !important;
}
.sp-hero-card {
  padding: 44px 48px !important;
}
.sp-hero-card h2 {
  font-size: 50px !important;
  line-height: 1.06 !important;
  letter-spacing: -0.045em !important;
}
.sp-hero-card p {
  font-size: 17px !important;
  line-height: 1.9 !important;
}
.sp-showcase-card {
  min-height: 330px !important;
}
.sp-metric-row {
  gap: 18px !important;
  margin: 24px 0 28px !important;
}
.sp-metric {
  padding: 22px 24px !important;
}
.sp-metric .label {
  font-size: 13px !important;
}
.sp-metric .value {
  font-size: 20px !important;
}
.sp-card, .sp-card-tight {
  padding: 28px !important;
  border-radius: 26px !important;
}
.sp-card-tight {
  padding: 24px !important;
}
.sp-section-title {
  font-size: 20px !important;
  margin-bottom: 10px !important;
}
.sp-subtitle {
  font-size: 14.5px !important;
  line-height: 1.75 !important;
  margin-bottom: 20px !important;
}
.sp-guide-step {
  padding: 15px !important;
  grid-template-columns: 42px 1fr !important;
}
.sp-guide-num {
  width: 42px !important;
  height: 42px !important;
}
.sp-guide-step b {
  font-size: 15px !important;
}
.sp-guide-step span {
  font-size: 13px !important;
}
.tab-nav button {
  min-height: 48px !important;
  padding: 12px 22px !important;
  font-size: 15px !important;
}
button, .gradio-button {
  min-height: 52px !important;
  font-size: 15px !important;
}
label, .label-wrap span {
  font-size: 14px !important;
}
input, textarea {
  min-height: 46px !important;
  font-size: 14.5px !important;
}
.sp-preview-card {
  min-height: 390px !important;
}
.sp-preview-card .image-container,
.sp-preview-card [data-testid="image"],
.sp-preview-card img {
  min-height: 280px !important;
}
#canvas-shell iframe {
  border-radius: 28px !important;
  min-height: 620px;
}
.dataframe table {
  font-size: 14px !important;
}

@media (max-width: 1120px) {
  #sp-hero-grid { grid-template-columns: 1fr; }
  .sp-metric-row { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
  #sp-topbar { align-items: flex-start; flex-direction: column; }
  .sp-status-strip { justify-content: flex-start; }
}
@media (max-width: 680px) {
  .gradio-container { padding: 14px !important; }
  #sp-app-inner { padding: 18px; }
  .sp-hero-card { padding: 24px; }
  .sp-hero-card h2 { font-size: 30px; }
  .sp-metric-row { grid-template-columns: 1fr; }
  .sp-card, .sp-card-tight { padding: 20px !important; }
}

/* Workspace style inspired by LibCom: restrained, tool-first, thin borders. */
:root {
  --sp-bg: #f7f8fb;
  --sp-panel: #ffffff;
  --sp-ink: #1f2937;
  --sp-muted: #8a94a6;
  --sp-primary: #2563eb;
  --sp-primary-2: #2563eb;
  --sp-border: #e6e9ef;
  --sp-shadow-card: 0 1px 2px rgba(15, 23, 42, 0.04);
  --sp-shadow-float: 0 8px 24px rgba(15, 23, 42, 0.06);
  --sp-radius-sm: 6px;
  --sp-radius-md: 8px;
  --sp-radius-lg: 10px;
}
html, body { background: var(--sp-bg) !important; }
.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
  background: #f7f8fb !important;
}
.gradio-container main,
.gradio-container .contain,
.gradio-container .tabs,
.gradio-container .tabitem {
  max-width: none !important;
}
.gradio-container main.contain {
  width: calc(100vw - 56px) !important;
  max-width: 1920px !important;
  margin: 0 auto !important;
}
#sp-app-frame {
  max-width: 1680px;
  margin: 0 auto;
  padding: 0 18px 22px 92px;
  border: 0;
  border-radius: 0;
  background: transparent !important;
  box-shadow: none !important;
}
#sp-app-frame::before { display: none !important; }
#sp-app-inner { padding: 0 !important; }
#sp-topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  min-height: 78px;
  margin: 0 -18px 16px -92px !important;
  padding: 16px 24px 16px 92px;
  background: rgba(247,248,251,0.96);
  border-bottom: 1px solid var(--sp-border);
  backdrop-filter: blur(12px);
}
.sp-brand {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 30;
  width: 72px;
  padding: 26px 10px;
  display: flex !important;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  background: #ffffff;
  border-right: 1px solid var(--sp-border);
  box-shadow: 1px 0 0 rgba(15,23,42,0.02);
}
.sp-logo {
  width: 38px !important;
  height: 38px !important;
  border-radius: 8px !important;
  display: grid;
  place-items: center;
  background: #111827 !important;
  color: #ffffff !important;
  font-weight: 800;
  box-shadow: none !important;
}
.sp-brand h1 {
  writing-mode: vertical-rl;
  text-orientation: mixed;
  letter-spacing: 0.12em !important;
  color: #111827 !important;
  font-weight: 760;
  font-size: 14px !important;
  line-height: 1.35;
  margin: 0 !important;
}
.sp-brand p { display: none !important; }
.sp-pill {
  border-radius: 999px;
  background: #ffffff !important;
  border: 1px solid var(--sp-border) !important;
  box-shadow: none !important;
  color: #5f6b7a !important;
}
.sp-pill-dot { background: #22c55e !important; box-shadow: none !important; }
#sp-hero-grid { display: none !important; }
.sp-metric-row {
  margin: 0 0 14px 92px !important;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 10px !important;
}
.sp-metric {
  border-radius: 8px !important;
  padding: 13px 14px !important;
  background: #ffffff !important;
  border: 1px solid var(--sp-border) !important;
  box-shadow: var(--sp-shadow-card) !important;
}
.sp-metric::before { display: none !important; }
.sp-metric .label { color: #8a94a6 !important; font-size: 12px !important; font-weight: 600 !important; }
.sp-metric .value { color: #1f2937 !important; font-size: 15px !important; font-weight: 760 !important; }
.sp-card, .sp-card-tight {
  border-radius: 8px !important;
  background: #ffffff !important;
  border: 1px solid var(--sp-border) !important;
  box-shadow: var(--sp-shadow-card) !important;
  backdrop-filter: none !important;
  overflow: hidden;
}
.sp-card::before, .sp-card-tight::before { display: none !important; }
.sp-card { padding: 16px !important; }
.sp-card-tight { padding: 14px !important; }
.sp-section-title {
  color: #1f2937 !important;
  font-size: 17px !important;
  font-weight: 760 !important;
  margin: 0 0 6px 0 !important;
  letter-spacing: 0 !important;
}
.sp-section-title::before {
  width: 3px !important;
  height: 16px !important;
  border-radius: 3px !important;
  background: #2563eb !important;
  box-shadow: none !important;
}
.sp-subtitle {
  margin: 0 0 14px 13px !important;
  color: #8a94a6 !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
}
.sp-guide-step { border-radius: 8px !important; background: #f8fafc !important; border: 1px solid #edf0f5 !important; }
.sp-guide-num { border-radius: 6px !important; background: #eef4ff !important; color: #2563eb !important; box-shadow: none !important; }
.tabs, .tabitem, .tabitem > div { border-radius: 0 !important; }
.tab-nav {
  gap: 0 !important;
  border-bottom: 1px solid var(--sp-border) !important;
  background: #ffffff !important;
  padding: 0 10px !important;
}
.tab-nav button {
  border-radius: 0 !important;
  padding: 13px 16px !important;
  color: #7b8494 !important;
  background: transparent !important;
  box-shadow: none !important;
  border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
  color: #111827 !important;
  background: transparent !important;
  border-bottom-color: #2563eb !important;
  box-shadow: none !important;
}
button, .gradio-button {
  border-radius: 6px !important;
  min-height: 38px !important;
  font-weight: 650 !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
  transform: none !important;
}
button.primary, .gradio-button.primary, .sp-blue button, .sp-green button {
  background: #111827 !important;
  color: #ffffff !important;
  border: 1px solid #111827 !important;
  box-shadow: none !important;
}
.sp-danger button {
  background: #ffffff !important;
  color: #b42318 !important;
  border: 1px solid #f1b5ad !important;
  box-shadow: none !important;
}
textarea, input, select, .wrap, .dataframe, .table-wrap, .image-container, .gallery, .gallery > div, .file-preview, .download-button, .accordion {
  border-radius: 8px !important;
}
input, textarea { background: #ffffff !important; border-color: var(--sp-border) !important; }
.dataframe th { background: #f8fafc !important; color: #334155 !important; }
#canvas-shell iframe {
  border-radius: 8px !important;
  border: 1px solid var(--sp-border) !important;
  box-shadow: none !important;
}
@media (max-width: 900px) {
  #sp-app-frame { padding-left: 18px; }
  .sp-brand { display: none !important; }
  #sp-topbar { margin-left: -18px !important; padding-left: 24px; }
  .sp-metric-row { margin-left: 0 !important; }
}

#sp-topbar,
.sp-metric-row {
  display: none !important;
}
#sp-app-frame {
  max-width: 1920px !important;
  padding: 18px 28px 28px !important;
}
.sp-brand { display: none !important; }
.tab-nav {
  margin-top: 0 !important;
}
.sp-card { padding: 20px !important; }
.sp-card-tight { padding: 18px !important; }
.sp-section-title {
  font-size: 18px !important;
  margin-bottom: 12px !important;
}
.sp-subtitle {
  display: none !important;
  margin-bottom: 8px !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
button, .gradio-button {
  min-height: 44px !important;
  font-size: 14px !important;
}
.tab-nav button {
  min-height: 44px !important;
  padding: 10px 18px !important;
  font-size: 14px !important;
}
.sp-preview-card {
  min-height: 330px !important;
}
.sp-preview-card .image-container,
.sp-preview-card [data-testid="image"],
.sp-preview-card img {
  min-height: 235px !important;
}
#canvas-shell iframe {
  min-height: 520px !important;
  max-height: 680px !important;
}
#workspace-main-row {
  flex-wrap: nowrap !important;
  gap: 20px !important;
  align-items: flex-start !important;
  overflow-x: auto !important;
  padding-bottom: 8px !important;
}
#sp-left-panel {
  flex: 0 1 clamp(340px, 20vw, 420px) !important;
  min-width: 340px !important;
  max-width: 460px !important;
}
#sp-center-panel {
  flex: 1 1 auto !important;
  min-width: 420px !important;
}
#sp-right-panel {
  flex: 0 1 clamp(560px, 34vw, 680px) !important;
  min-width: 560px !important;
  max-width: 680px !important;
}
#sp-left-panel .sp-card {
  min-height: calc(100vh - 150px) !important;
  max-height: none !important;
  overflow-y: visible !important;
}
#sp-right-panel .sp-card {
  min-height: calc(100vh - 150px) !important;
  max-height: none !important;
  overflow-y: visible !important;
  overflow-x: visible !important;
}
#sp-right-panel .accordion,
#sp-right-panel .form,
#sp-right-panel .block,
#sp-right-panel .wrap {
  max-width: 100% !important;
  overflow: visible !important;
  box-sizing: border-box !important;
}
#sp-right-panel .form,
#sp-right-panel .block {
  min-height: auto !important;
  height: auto !important;
}
#sp-right-panel input {
  min-height: 46px !important;
  height: 46px !important;
}
#sp-right-panel label,
#sp-right-panel .wrap {
  min-height: 58px !important;
}
#sp-right-panel label,
#sp-right-panel .label-wrap,
#sp-right-panel .prose {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}
#sp-right-panel .accordion {
  border: 1px solid var(--sp-border) !important;
  box-shadow: none !important;
}
#sp-right-panel .accordion button {
  width: 100% !important;
  border: 0 !important;
  outline: none !important;
  box-shadow: none !important;
}
#sp-right-panel .accordion:focus-within,
#sp-right-panel .accordion button:focus {
  outline: none !important;
  box-shadow: none !important;
}
#run-analysis-text textarea {
  max-height: 430px !important;
  overflow-y: auto !important;
  resize: vertical !important;
  line-height: 1.65 !important;
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
libcom_cfg = scorer_cfg.get("libcom_opa_subprocess", {})
scorer = LibcomOPASubprocessScorer(
    python_path=libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe"),
    script_path=libcom_cfg.get("script_path", "scripts/libcom_opa_infer_once.py"),
    batch_script_path=libcom_cfg.get("batch_script_path", "scripts/libcom_opa_infer_batch.py"),
    device=libcom_cfg.get("device", "cuda:0"),
    model_type=libcom_cfg.get("model_type", "SimOPA"),
    temp_dir=libcom_cfg.get("temp_dir", "outputs/libcom_subprocess"),
    logger=logger,
)

multi_cfg = scorer_cfg.get("libcom_multimodel", {})
libcom_multimodel = LibcomMultiModelSubprocess(
    python_path=multi_cfg.get("python_path", libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe")),
    script_path=multi_cfg.get("script_path", "scripts/libcom_multi_model_infer.py"),
    device=multi_cfg.get("device", libcom_cfg.get("device", "cuda:0")),
    temp_dir=multi_cfg.get("temp_dir", "outputs/libcom_multimodel"),
    logger=logger,
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
    info = scorer.get_model_info()
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


def build_run_analysis_text(summary: Dict[str, Any], ranked: List[Dict[str, Any]], mask_info: Dict[str, Any], drag_mode: str, explanation_text: str = "") -> str:
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
    lines.append("复杂交互：用户可以在浏览器画布中直接拖动前景物体，并将拖拽位置记录为候选位置。")
    lines.append("参考模型评分：系统将拖拽候选合成为 composite image + composite mask，并调用 libcom OPAScoreModel 批量评分。")
    lines.append("多工具串联：前景 mask 处理 → 拖拽候选记录 → 图像合成 → OPA 评分 → Top-K 推荐 → 导出结果。")
    if explanation_text:
        lines.append("")
        lines.append(explanation_text)
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

    canvas_scale = min(980 / bg_w, 680 / bg_h, 1.0)
    canvas_w = int(bg_w * canvas_scale)
    canvas_h = int(bg_h * canvas_scale)
    display_fg_w = int(fg_w * canvas_scale)
    display_fg_h = int(fg_h * canvas_scale)

    init_x = max(0, (bg_w - fg_w) // 2)
    init_y = max(0, bg_h - fg_h - int(bg_h * 0.08))

    bg_url = pil_to_data_url(background)
    fg_url = pil_to_data_url(resized_fg)

    # 关键改动：
    # 不再直接把 <script> 塞进 gr.HTML。
    # 改为 iframe srcdoc，让脚本在 iframe 内稳定执行。
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
    border: 1px solid rgba(255,255,255,0.78);
    padding: 24px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(255,255,255,0.86), rgba(255,255,255,0.58));
    box-shadow: 0 18px 44px rgba(31,41,55,0.10);
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
    border: 1px solid rgba(148,163,184,0.30);
    border-radius: 20px;
    background: #fff;
    overflow: hidden;
    user-select: none;
    touch-action: none;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.90), 0 16px 34px rgba(31,41,55,0.08);
  }}
  #stage::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
    border-radius: 20px;
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
    filter: drop-shadow(0 16px 22px rgba(15,23,42,0.16));
  }}
  #box {{
    position: absolute;
    left: {int(init_x * canvas_scale)}px;
    top: {int(init_y * canvas_scale)}px;
    width: {display_fg_w}px;
    height: {display_fg_h}px;
    border: 2px solid rgba(91,141,239,0.92);
    border-radius: 14px;
    box-sizing: border-box;
    pointer-events: none;
    box-shadow: 0 0 0 5px rgba(91,141,239,0.14), 0 0 24px rgba(119,200,232,0.22);
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

    status.innerText =
      "当前位置：x=" + ox +
      ", y=" + oy +
      ", scale=" + currentScale.toFixed(3) +
      "；物体尺寸≈" + originalFgW + "×" + originalFgH;
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


def score_drag_candidates(
    bg_state,
    fg_state,
    mask_info_state,
    candidate_points,
    top_k,
    filter_out_of_bounds,
    enable_explanation,
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
    run_id = time.strftime("%Y%m%d_%H%M%S")

    logger.section("[SmartPlace-Drag] Start drag candidate scoring")
    logger.log(f"[Input] background_size={background.size}")
    logger.log(f"[Input] foreground_size={foreground.size}")
    logger.log(f"[Param] recorded_candidates={len(candidate_points)}")
    logger.log(f"[Param] top_k={top_k}")
    logger.log(f"[Param] filter_out_of_bounds={filter_out_of_bounds}")
    logger.log(f"[Param] enable_explanation={enable_explanation}")

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

    ranked_all = assign_relative_labels_in_place(results, top_k=top_k)
    ranked = ranked_all[:top_k]

    if cfg.get("output", {}).get("save_images", True):
        save_candidate_images(results, run_id=run_id)

    summary = summarize_run(results, top_k=top_k)

    explanation_gallery = []
    explanation_text = ""
    explanation_overlay_path = None
    explanation_report_path = None
    libcom_suite_text = "未运行 LibCom 增强模型。"
    libcom_suite_gallery = []

    if enable_explanation and ranked:
        top1 = ranked[0]
        logger.section("[SmartPlace-Drag] Start explanation for Top-1 candidate")
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
        explanation_gallery.append((explanation_overlay_path, f"候选 {top1['id']} 遮挡实验热力图"))

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
        explanation_gallery,
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
    bg_state = gr.State(value=None)
    fg_state = gr.State(value=None)
    mask_info_state = gr.State(value={})
    candidate_points_state = gr.State(value=[])
    drag_mode_state = gr.State(value="用户通过鼠标在画布中拖动前景物体，并记录多个候选位置。")

    gr.HTML(
        """
<div id="sp-app-frame">
  <div id="sp-app-inner">
    <div id="sp-topbar">
      <div class="sp-brand">
        <div class="sp-logo">SP</div>
        <div>
          <h1>SmartPlace Studio</h1>
          <p>智能物体放置 · 交互式评分 · 课堂答辩展示平台</p>
        </div>
      </div>
      <div class="sp-status-strip">
        <span class="sp-pill"><span class="sp-pill-dot"></span>Local Inference Ready</span>
        <span class="sp-pill">libcom OPA</span>
        <span class="sp-pill">Drag Canvas</span>
        <span class="sp-pill">Top-K Report</span>
      </div>
    </div>

    <div id="sp-hero-grid">
      <div class="sp-hero-card">
        <div class="sp-kicker">Science & Innovation Demo</div>
        <h2>把物体放到更自然的位置，<span>让模型给出可解释推荐。</span></h2>
        <p>
          上传背景图与前景物体，在画布中直接拖拽生成候选位置。系统会合成 composite image 与 composite mask，
          调用真实 libcom OPAScoreModel 完成批量评分，并生成 Top-K 推荐、候选评分表、案例记录与可复现实验文件。
        </p>
        <div class="sp-hero-actions">
          <span class="sp-soft-tag">清爽浅色系</span>
          <span class="sp-soft-tag">模型真实运行证据</span>
          <span class="sp-soft-tag">答辩演示友好</span>
          <span class="sp-soft-tag">报告自动导出</span>
        </div>
      </div>
      <div class="sp-showcase-card">
        <div class="sp-score-orbit"></div>
        <div>
          <h3>展示逻辑更像产品，而不是脚本界面</h3>
          <p>输入、交互、评分、解释、导出分区明确，现场演示时可以顺着流程自然讲解。</p>
        </div>
      </div>
    </div>
  </div>
</div>
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

    with gr.Tabs(selected="workspace") as main_tabs:
        with gr.Tab("01 · 创作工作台", id="workspace"):
            with gr.Row(equal_height=False, elem_id="workspace-main-row"):
                with gr.Column(scale=3, min_width=340, elem_id="sp-left-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>前景图片区域</div><div class='sp-subtitle'>预制前景图片 / 本地上传</div>")
                        foreground_input = gr.Image(label="前景物体 Foreground", type="numpy", height=240)
                        gr.Examples(
                            examples=[
                                os.path.join("assets", "foregrounds", "cup.png"),
                                os.path.join("assets", "foregrounds", "chair.png"),
                                os.path.join("assets", "foregrounds", "car.png"),
                            ],
                            inputs=foreground_input,
                            label="预制前景图片",
                        )
                        gr.HTML("<div class='sp-section-title'>背景图片区域</div><div class='sp-subtitle'>预制背景图片 / 本地上传</div>")
                        background_input = gr.Image(label="背景图 Background", type="numpy", height=240)
                        gr.Examples(
                            examples=[
                                os.path.join("assets", "backgrounds", "desk.png"),
                                os.path.join("assets", "backgrounds", "classroom.png"),
                                os.path.join("assets", "backgrounds", "street.png"),
                            ],
                            inputs=background_input,
                            label="预制背景图片",
                        )

                    case_name_input = gr.State("SmartPlace 工作台案例")
                    background_note_input = gr.State("")
                    foreground_note_input = gr.State("")
                    manual_label_input = gr.State("未填写")
                    manual_reason_input = gr.State("")

                with gr.Column(scale=6, min_width=420, elem_id="sp-center-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>交互画布</div><div class='sp-subtitle'>在画布中拖动物体，记录多个候选位置后再统一评分。</div>")
                        drag_canvas_html = gr.HTML(label="拖拽画布", elem_id="canvas-shell")
                        with gr.Row():
                            load_canvas_button = gr.Button("加载 / 刷新拖拽画布", variant="primary", elem_classes=["sp-blue"])
                            add_candidate_button = gr.Button("记录当前位置为候选", variant="primary", elem_classes=["sp-blue"])
                        with gr.Row():
                            clear_candidate_button = gr.Button("清空候选", variant="primary", elem_classes=["sp-blue"])
                            score_button = gr.Button("批量评分并生成结果", variant="primary", elem_classes=["sp-blue"])

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, elem_classes=["sp-card-tight", "sp-preview-card"]):
                            gr.HTML("<div class='sp-section-title'>Mask 预览</div><div class='sp-subtitle'>加载画布后自动显示前景分割结果。</div>")
                            mask_preview_output = gr.Image(label="Mask", type="pil", height=190)
                        with gr.Column(scale=1, elem_classes=["sp-card-tight", "sp-preview-card"]):
                            gr.HTML("<div class='sp-section-title'>处理后前景</div><div class='sp-subtitle'>去底后的前景物体会显示在这里。</div>")
                            processed_foreground_output = gr.Image(label="Foreground", type="pil", height=190)

                    with gr.Accordion("已记录候选", open=False):
                        candidate_points_table = gr.Dataframe(label="候选位置表", wrap=True)

                    with gr.Accordion("调试坐标，可用于证明拖拽交互确实写入候选", open=False):
                        with gr.Row():
                            drag_x_input = gr.Textbox(label="当前 x", value="0", elem_id="drag_x_input")
                            drag_y_input = gr.Textbox(label="当前 y", value="0", elem_id="drag_y_input")
                            drag_scale_input = gr.Textbox(label="当前 scale", value="0.25", elem_id="drag_scale_input")

                with gr.Column(scale=3, min_width=560, elem_id="sp-right-panel"):
                    with gr.Group(elem_classes=["sp-card"]):
                        gr.HTML("<div class='sp-section-title'>模型与参数</div><div class='sp-subtitle'>参数越少越适合演示，更多选项已收纳。</div>")
                        mask_mode_input = gr.State(cfg.get("mask_processor", {}).get("default_mode", "自动判断"))
                        white_bg_threshold_input = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=cfg.get("mask_processor", {}).get("white_bg_threshold", 38),
                            step=2,
                            label="去底阈值",
                            buttons=[],
                        )
                        scale_input = gr.Slider(minimum=0.05, maximum=0.8, value=0.25, step=0.05, label="前景缩放比例", buttons=[])
                        top_k_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-K 数量", buttons=[])
                        filter_out_of_bounds_input = gr.Checkbox(value=True, label="过滤明显越界候选", interactive=True)
                        enable_explanation_input = gr.Checkbox(value=False, label="生成模型解释图（较慢）", interactive=True)
                        enable_libcom_suite_input = gr.Checkbox(value=False, label="启用 LibCom 增强模型（较慢）", interactive=True)
                        libcom_suite_models_input = gr.CheckboxGroup(
                            choices=["fopa", "fos", "harmony", "pctnet", "lbm"],
                            value=["fos", "harmony", "pctnet"],
                            label="增强模型选择",
                        )
                        with gr.Accordion("LibCom 增强模型参数", open=True):
                            lbm_steps_input = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="LBM steps", buttons=[])
                            lbm_resolution_input = gr.Slider(minimum=512, maximum=1024, value=768, step=128, label="LBM resolution", buttons=[])
                        with gr.Accordion("高级解释图参数", open=True):
                            occlusion_patch_size_input = gr.Slider(minimum=48, maximum=160, value=96, step=16, label="遮挡块大小", buttons=[])
                            occlusion_stride_input = gr.Slider(minimum=32, maximum=128, value=96, step=16, label="遮挡滑动步长", buttons=[])

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
                    gr.HTML("<div class='sp-section-title'>自动分析说明</div><div class='sp-subtitle'>可直接作为汇报讲解素材。</div>")
                    run_analysis_text = gr.Textbox(label="分析说明", lines=14, max_lines=14, elem_id="run-analysis-text")

        with gr.Tab("03 · 解释与案例库", id="analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>模型解释图</div><div class='sp-subtitle'>开启解释后，对 Top-1 候选做遮挡实验，显示影响区域。</div>")
                    explanation_gallery = gr.Gallery(label="遮挡实验热力图", columns=1, height="auto")
                    gr.HTML("<div class='sp-section-title'>LibCom 增强模型</div><div class='sp-subtitle'>可选运行 FOPA、FOS、HarmonyScore、PCTNet、LBM，对 Top-1 候选做进一步评估与协调。</div>")
                    libcom_suite_gallery = gr.Gallery(label="FOPA 热力图 / 协调结果", columns=1, height="auto")
                    libcom_suite_text = gr.Textbox(label="多模型结果摘要", lines=12)
                with gr.Column(scale=1, elem_classes=["sp-card"]):
                    gr.HTML("<div class='sp-section-title'>测试案例汇总</div><div class='sp-subtitle'>所有评分案例会自动沉淀，方便报告展示。</div>")
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

    score_event = score_button.click(
        fn=lambda: gr.update(selected="results"),
        inputs=[],
        outputs=[main_tabs],
    )
    score_event.then(
        fn=score_drag_candidates,
        inputs=[
            bg_state,
            fg_state,
            mask_info_state,
            candidate_points_state,
            top_k_input,
            filter_out_of_bounds_input,
            enable_explanation_input,
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
        ],
        outputs=[
            candidate_gallery,
            topk_gallery,
            score_table,
            run_analysis_text,
            explanation_gallery,
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
