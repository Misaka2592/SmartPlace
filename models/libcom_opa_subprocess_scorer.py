import json
import os
import subprocess
import time
from typing import Dict, List

from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class LibcomOPASubprocessScorer(BaseScorer):
    """
    主 Web 环境使用的 libcom OPA 子进程评分器。
    当前进程不 import libcom，避免 Gradio 与 libcom 依赖冲突。
    """

    def __init__(
        self,
        python_path: str = ".venv_libcom/Scripts/python.exe",
        script_path: str = "scripts/libcom_opa_infer_once.py",
        batch_script_path: str = "scripts/libcom_opa_infer_batch.py",
        device: str = "cuda:0",
        model_type: str = "SimOPA",
        temp_dir: str = "outputs/libcom_subprocess",
        timeout_seconds: int = 120,
        logger: InferenceLogger = None,
    ):
        self.python_path = python_path
        self.script_path = script_path
        self.batch_script_path = batch_script_path
        self.device = device
        self.model_type = model_type
        self.temp_dir = temp_dir
        self.timeout_seconds = int(timeout_seconds)
        self.logger = logger or InferenceLogger()
        self.model_name = "libcom.OPAScoreModel.SimOPA.subprocess"
        self.is_loaded = True
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger.section("[LibcomOPASubprocessScorer] Init")
        self.logger.log(f"[Model] name={self.model_name}")
        self.logger.log(f"[Model] python_path={self.python_path}")
        self.logger.log(f"[Model] script_path={self.script_path}")
        self.logger.log(f"[Model] batch_script_path={self.batch_script_path}")
        self.logger.log(f"[Model] device={self.device}")
        self.logger.log(f"[Model] model_type={self.model_type}")
        self.logger.log(f"[Model] timeout_seconds={self.timeout_seconds}")

    def _extract_json_from_stdout(self, stdout: str) -> Dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError("Cannot parse JSON from libcom subprocess stdout.\n" + stdout)

    def _save_image_and_mask(self, image: Image.Image, info: Dict, candidate_id: str):
        composite_path = os.path.join(self.temp_dir, f"candidate_{candidate_id}_composite.png")
        mask_path = os.path.join(self.temp_dir, f"candidate_{candidate_id}_mask.png")
        image.convert("RGB").save(composite_path)

        composite_mask = info.get("composite_mask")
        if composite_mask is None:
            raise ValueError(f"candidate_id={candidate_id} 缺少 composite_mask，无法调用 libcom OPA。")
        if isinstance(composite_mask, Image.Image):
            composite_mask.convert("L").save(mask_path)
        else:
            Image.fromarray(composite_mask).convert("L").save(mask_path)
        return composite_path, mask_path

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        candidate_id = candidate_info.get("candidate_id", "unknown")
        composite_path, mask_path = self._save_image_and_mask(image, candidate_info, str(candidate_id))
        cmd = [
            self.python_path,
            self.script_path,
            "--composite", composite_path,
            "--mask", mask_path,
            "--device", self.device,
            "--model_type", self.model_type,
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.log(f"[LibcomOPA-Subprocess] timeout after {self.timeout_seconds}s")
            raise RuntimeError(f"LibCom OPA single inference timed out after {self.timeout_seconds}s") from exc
        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[LibcomOPA-Subprocess] failed")
            self.logger.log(proc.stderr)
            raise RuntimeError(proc.stderr)
        result = self._extract_json_from_stdout(proc.stdout)
        score = float(result["score"])
        self.logger.log(
            f"[LibcomOPA-Subprocess] candidate_id={candidate_id}, score={score:.6f}, subprocess_time={elapsed:.6f}s"
        )
        return max(0.0, min(1.0, score))

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        self.logger.section("[LibcomOPASubprocessScorer] Batch inference")
        composite_paths = []
        mask_paths = []
        for image, info in zip(images, candidate_infos):
            candidate_id = info.get("candidate_id", "unknown")
            comp_path, mask_path = self._save_image_and_mask(image, info, str(candidate_id))
            composite_paths.append(comp_path)
            mask_paths.append(mask_path)

        cmd = [
            self.python_path,
            self.batch_script_path,
            "--device", self.device,
            "--model_type", self.model_type,
            "--composites", *composite_paths,
            "--masks", *mask_paths,
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.log(f"[LibcomOPA-Subprocess-Batch] timeout after {self.timeout_seconds}s")
            raise RuntimeError(f"LibCom OPA batch inference timed out after {self.timeout_seconds}s") from exc
        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[LibcomOPA-Subprocess-Batch] failed")
            self.logger.log("[stderr]")
            self.logger.log(proc.stderr)
            self.logger.log("[stdout]")
            self.logger.log(proc.stdout)
            raise RuntimeError(proc.stderr)

        output = self._extract_json_from_stdout(proc.stdout)
        scores = [float(item["score"]) for item in output["results"]]
        self.logger.log(
            f"[LibcomOPA-Subprocess-Batch] batch_size={len(scores)}, "
            f"scores={[round(s, 6) for s in scores]}, "
            f"subprocess_time={elapsed:.6f}s, "
            f"libcom_batch_time={float(output.get('batch_time', 0.0)):.6f}s"
        )
        return scores

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "source": "BCMI/libcom OPAScoreModel via subprocess",
            "device": self.device,
            "model_type": self.model_type,
            "python_path": self.python_path,
            "script_path": self.script_path,
            "batch_script_path": self.batch_script_path,
            "timeout_seconds": self.timeout_seconds,
            "is_loaded": self.is_loaded,
        }
