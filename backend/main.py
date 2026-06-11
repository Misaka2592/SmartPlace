from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services.runtime import PROJECT_ROOT
from backend.services.workbench_service import (
    compose_session,
    export_report_bundle,
    generate_heatmap,
    save_upload_image,
    score_session,
)


app = FastAPI(title="SmartPlace React Workbench API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(PROJECT_ROOT)), name="files")


class ComposeRequest(BaseModel):
    background_path: str
    foreground_path: str
    mask_mode: str = "自动判断"
    white_bg_threshold: int = 38
    scale: float = 0.25


class CandidatePoint(BaseModel):
    id: int
    x: int
    y: int
    scale: float


class ScoreCandidatesRequest(BaseModel):
    session_id: str
    candidate_points: List[CandidatePoint]
    top_k: int = 3
    filter_out_of_bounds: bool = True
    enable_explanation: bool = False
    enable_saliency: bool = False
    enable_feature_analysis: bool = False
    occlusion_patch_size: int = 96
    occlusion_stride: int = 96
    enable_libcom_suite: bool = False
    libcom_suite_models: List[str] = Field(default_factory=lambda: ["fopa", "fos", "harmony"])
    lbm_steps: int = 4
    lbm_resolution: int = 768
    case_name: str = "React Workbench Case"
    background_note: str = ""
    foreground_note: str = ""
    manual_label: str = ""
    manual_reason: str = ""
    drag_mode_state: str = "React drag canvas"
    score_backend: str = "handin_opa_subprocess"


class HeatmapRequest(BaseModel):
    run_id: str
    candidate_id: Optional[int] = None
    patch_size: int = 96
    stride: int = 96
    score_backend: Optional[str] = None


class ExportReportRequest(BaseModel):
    run_id: str


def _read_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return data


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/api/upload_foreground")
def upload_foreground(file: UploadFile = File(...)) -> Dict[str, Any]:
    return save_upload_image(_read_bytes(file), "foreground", file.filename or "foreground.png")


@app.post("/api/upload_background")
def upload_background(file: UploadFile = File(...)) -> Dict[str, Any]:
    return save_upload_image(_read_bytes(file), "background", file.filename or "background.png")


@app.post("/api/compose")
def compose(request: ComposeRequest) -> Dict[str, Any]:
    try:
        return compose_session(
            background_path=request.background_path,
            foreground_path=request.foreground_path,
            mask_mode=request.mask_mode,
            white_bg_threshold=request.white_bg_threshold,
            scale=request.scale,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/score_candidates")
def score_candidates(request: ScoreCandidatesRequest) -> Dict[str, Any]:
    try:
        return score_session(
            session_id=request.session_id,
            candidate_points=[item.model_dump() for item in request.candidate_points],
            top_k=request.top_k,
            filter_out_of_bounds=request.filter_out_of_bounds,
            enable_explanation=request.enable_explanation,
            enable_saliency=request.enable_saliency,
            enable_feature_analysis=request.enable_feature_analysis,
            occlusion_patch_size=request.occlusion_patch_size,
            occlusion_stride=request.occlusion_stride,
            enable_libcom_suite=request.enable_libcom_suite,
            libcom_suite_models=request.libcom_suite_models,
            lbm_steps=request.lbm_steps,
            lbm_resolution=request.lbm_resolution,
            case_name=request.case_name,
            background_note=request.background_note,
            foreground_note=request.foreground_note,
            manual_label=request.manual_label,
            manual_reason=request.manual_reason,
            drag_mode_state=request.drag_mode_state,
            score_backend=request.score_backend,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/generate_heatmap")
def api_generate_heatmap(request: HeatmapRequest) -> Dict[str, Any]:
    try:
        return generate_heatmap(
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            patch_size=request.patch_size,
            stride=request.stride,
            score_backend=request.score_backend,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export_report")
def api_export_report(request: ExportReportRequest) -> Dict[str, Any]:
    try:
        return export_report_bundle(run_id=request.run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
