from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from models.handin_opa_subprocess_scorer import HandinOPASubprocessScorer
from models.libcom_multimodel_subprocess import LibcomMultiModelSubprocess
from models.libcom_opa_subprocess_scorer import LibcomOPASubprocessScorer
from models.smartplace_opa_calibrated_scorer import SmartPlaceOPACalibratedScorer
from utils.handin_u2net_subprocess import HandinU2NetSubprocessMatting
from utils.logger import InferenceLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
STORAGE_ROOT = BACKEND_ROOT / "storage"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
SESSION_ROOT = STORAGE_ROOT / "sessions"
RUN_ROOT = STORAGE_ROOT / "runs"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
COMPOSITE_DIR = OUTPUT_DIR / "composites"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"
EXPLAIN_DIR = OUTPUT_DIR / "explanations"
MASK_DIR = OUTPUT_DIR / "masks"
REPORT_RESULT_DIR = PROJECT_ROOT / "report" / "results"
CASE_DIR = PROJECT_ROOT / "report" / "cases"

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

for path in [
    STORAGE_ROOT,
    UPLOAD_ROOT / "foregrounds",
    UPLOAD_ROOT / "backgrounds",
    SESSION_ROOT,
    RUN_ROOT,
    COMPOSITE_DIR,
    TABLE_DIR,
    LOG_DIR,
    EXPLAIN_DIR,
    MASK_DIR,
    REPORT_RESULT_DIR,
    CASE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
logger = InferenceLogger(
    log_dir=str(LOG_DIR),
    enable_file_log=cfg.get("output", {}).get("save_log", True),
)
scorer_cfg = cfg.get("scorer", {})
active_backend = scorer_cfg.get("active_backend", "handin_opa_subprocess")
calibration_cfg = scorer_cfg.get("smartplace_opa_calibrated", {})
libcom_cfg = scorer_cfg.get("libcom_opa_subprocess", {})
runtime_scorer_cache: Dict[str, SmartPlaceOPACalibratedScorer] = {}


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _build_calibrated_scorer(backend_key: str) -> SmartPlaceOPACalibratedScorer:
    if backend_key == "handin_opa_subprocess":
        handin_cfg = scorer_cfg.get("handin_opa_subprocess", {})
        base_scorer = HandinOPASubprocessScorer(
            python_path=_resolve_path(handin_cfg.get("python_path", "../handin/.venv/Scripts/python.exe")),
            script_path=_resolve_path(handin_cfg.get("script_path", "scripts/handin_opa_infer_once.py")),
            batch_script_path=_resolve_path(handin_cfg.get("batch_script_path", "scripts/handin_opa_infer_batch.py")),
            handin_root=_resolve_path(handin_cfg.get("handin_root", "../handin")),
            weight_path=_resolve_path(handin_cfg.get("weight_path", "../handin/experiments/ablation_study/resnet18_w05_20260609_161229/checkpoints/resnet18_w05_best-acc-0.718_epoch15_f1-0.614.pth")),
            device=handin_cfg.get("device", "cpu"),
            model_name=handin_cfg.get("model_name", "resnet"),
            layers=handin_cfg.get("layers", 18),
            width_factor=handin_cfg.get("width_factor", 0.5),
            temp_dir=_resolve_path(handin_cfg.get("temp_dir", "outputs/handin_subprocess")),
            timeout_seconds=handin_cfg.get("timeout_seconds", 120),
            logger=logger,
        )
    else:
        base_scorer = LibcomOPASubprocessScorer(
            python_path=_resolve_path(libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe")),
            script_path=_resolve_path(libcom_cfg.get("script_path", "scripts/libcom_opa_infer_once.py")),
            batch_script_path=_resolve_path(libcom_cfg.get("batch_script_path", "scripts/libcom_opa_infer_batch.py")),
            device=libcom_cfg.get("device", "cuda:0"),
            model_type=libcom_cfg.get("model_type", "SimOPA"),
            temp_dir=_resolve_path(libcom_cfg.get("temp_dir", "outputs/libcom_subprocess")),
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


multi_cfg = scorer_cfg.get("libcom_multimodel", {})
libcom_multimodel = LibcomMultiModelSubprocess(
    python_path=_resolve_path(multi_cfg.get("python_path", libcom_cfg.get("python_path", ".venv_libcom/Scripts/python.exe"))),
    script_path=_resolve_path(multi_cfg.get("script_path", "scripts/libcom_multi_model_infer.py")),
    device=multi_cfg.get("device", libcom_cfg.get("device", "cuda:0")),
    temp_dir=_resolve_path(multi_cfg.get("temp_dir", "outputs/libcom_multimodel")),
    logger=logger,
)

u2net_cfg = cfg.get("u2net", {})
u2net_runner = HandinU2NetSubprocessMatting(
    python_path=_resolve_path(u2net_cfg.get("python_path", "../handin/.venv/Scripts/python.exe")),
    script_path=_resolve_path(u2net_cfg.get("script_path", "scripts/handin_u2net_infer_once.py")),
    handin_root=_resolve_path(u2net_cfg.get("handin_root", "../handin")),
    model_type=u2net_cfg.get("model_type", "u2netp"),
    weight_path=_resolve_path(u2net_cfg.get("weight_path", "../handin/u2netp.pth")),
    device=u2net_cfg.get("device", "cpu"),
    threshold=u2net_cfg.get("threshold", 0.5),
    temp_dir=_resolve_path(u2net_cfg.get("temp_dir", "outputs/handin_u2net")),
    timeout_seconds=u2net_cfg.get("timeout_seconds", 120),
)
