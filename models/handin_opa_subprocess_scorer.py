import json
import os
import subprocess
import time
from typing import Dict, List

from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class HandinOPASubprocessScorer(BaseScorer):
    """
    Calls the handin OPA model in a separate Python environment.

    This keeps the SmartPlace app environment lightweight while reusing the
    trained OPA checkpoints and PyTorch inference code stored in ../handin.
    """

    def __init__(
        self,
        python_path: str = "../handin/.venv/Scripts/python.exe",
        script_path: str = "scripts/handin_opa_infer_once.py",
        batch_script_path: str = "scripts/handin_opa_infer_batch.py",
        handin_root: str = "../handin",
        weight_path: str = "../handin/experiments/ablation_study/resnet18_w05_20260609_161229/checkpoints/resnet18_w05_best-acc-0.718_epoch15_f1-0.614.pth",
        model_name: str = "resnet",
        layers: int = 18,
        width_factor: float = 0.5,
        device: str = "cpu",
        temp_dir: str = "outputs/handin_subprocess",
        timeout_seconds: int = 120,
        logger: InferenceLogger = None,
    ):
        self.python_path = os.path.normpath(python_path)
        self.script_path = os.path.normpath(script_path)
        self.batch_script_path = os.path.normpath(batch_script_path)
        self.handin_root = os.path.normpath(handin_root)
        self.weight_path = os.path.normpath(weight_path)
        self.model_name = model_name
        self.layers = int(layers)
        self.width_factor = float(width_factor)
        self.device = device
        self.temp_dir = temp_dir
        self.timeout_seconds = int(timeout_seconds)
        self.logger = logger or InferenceLogger()
        self.runtime_name = f"handin.OPA.{self.model_name}.subprocess"
        self.is_loaded = True
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger.section("[HandinOPASubprocessScorer] Init")
        self.logger.log(f"[Model] name={self.runtime_name}")
        self.logger.log(f"[Model] python_path={self.python_path}")
        self.logger.log(f"[Model] script_path={self.script_path}")
        self.logger.log(f"[Model] batch_script_path={self.batch_script_path}")
        self.logger.log(f"[Model] handin_root={self.handin_root}")
        self.logger.log(f"[Model] weight_path={self.weight_path}")
        self.logger.log(f"[Model] model_name={self.model_name}")
        self.logger.log(f"[Model] layers={self.layers}")
        self.logger.log(f"[Model] width_factor={self.width_factor}")
        self.logger.log(f"[Model] device={self.device}")
        self.logger.log(f"[Model] timeout_seconds={self.timeout_seconds}")

    def _extract_json_from_stdout(self, stdout: str) -> Dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError("Cannot parse JSON from handin subprocess stdout.\n" + stdout)

    def _save_image_and_mask(self, image: Image.Image, info: Dict, candidate_id: str):
        composite_path = os.path.join(self.temp_dir, f"candidate_{candidate_id}_composite.png")
        mask_path = os.path.join(self.temp_dir, f"candidate_{candidate_id}_mask.png")
        image.convert("RGB").save(composite_path)

        composite_mask = info.get("composite_mask")
        if composite_mask is None:
            raise ValueError(f"candidate_id={candidate_id} missing composite_mask.")
        if isinstance(composite_mask, Image.Image):
            composite_mask.convert("L").save(mask_path)
        else:
            Image.fromarray(composite_mask).convert("L").save(mask_path)
        return composite_path, mask_path

    def _build_base_cmd(self, script_path: str) -> List[str]:
        return [
            self.python_path,
            script_path,
            "--handin_root", self.handin_root,
            "--weight", self.weight_path,
            "--model_name", self.model_name,
            "--layers", str(self.layers),
            "--width_factor", str(self.width_factor),
            "--device", self.device,
        ]

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        candidate_id = candidate_info.get("candidate_id", "unknown")
        composite_path, mask_path = self._save_image_and_mask(image, candidate_info, str(candidate_id))
        cmd = self._build_base_cmd(self.script_path) + [
            "--composite", composite_path,
            "--mask", mask_path,
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
            self.logger.log(f"[HandinOPA-Subprocess] timeout after {self.timeout_seconds}s")
            raise RuntimeError(f"handin OPA single inference timed out after {self.timeout_seconds}s") from exc
        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[HandinOPA-Subprocess] failed")
            self.logger.log(proc.stderr)
            raise RuntimeError(proc.stderr)
        result = self._extract_json_from_stdout(proc.stdout)
        score = float(result["score"])
        self.logger.log(
            f"[HandinOPA-Subprocess] candidate_id={candidate_id}, score={score:.6f}, subprocess_time={elapsed:.6f}s"
        )
        return max(0.0, min(1.0, score))

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        self.logger.section("[HandinOPASubprocessScorer] Batch inference")
        composite_paths = []
        mask_paths = []
        for image, info in zip(images, candidate_infos):
            candidate_id = info.get("candidate_id", "unknown")
            comp_path, mask_path = self._save_image_and_mask(image, info, str(candidate_id))
            composite_paths.append(comp_path)
            mask_paths.append(mask_path)

        cmd = self._build_base_cmd(self.batch_script_path) + [
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
            self.logger.log(f"[HandinOPA-Subprocess-Batch] timeout after {self.timeout_seconds}s")
            raise RuntimeError(f"handin OPA batch inference timed out after {self.timeout_seconds}s") from exc
        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[HandinOPA-Subprocess-Batch] failed")
            self.logger.log("[stderr]")
            self.logger.log(proc.stderr)
            self.logger.log("[stdout]")
            self.logger.log(proc.stdout)
            raise RuntimeError(proc.stderr)

        output = self._extract_json_from_stdout(proc.stdout)
        scores = [float(item["score"]) for item in output["results"]]
        self.logger.log(
            f"[HandinOPA-Subprocess-Batch] batch_size={len(scores)}, "
            f"scores={[round(s, 6) for s in scores]}, "
            f"subprocess_time={elapsed:.6f}s, "
            f"handin_batch_time={float(output.get('batch_time', 0.0)):.6f}s"
        )
        return scores

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.runtime_name,
            "source": "handin OPA checkpoint via subprocess",
            "device": self.device,
            "python_path": self.python_path,
            "script_path": self.script_path,
            "batch_script_path": self.batch_script_path,
            "handin_root": self.handin_root,
            "weight_path": self.weight_path,
            "layers": self.layers,
            "width_factor": self.width_factor,
            "timeout_seconds": self.timeout_seconds,
            "is_loaded": self.is_loaded,
        }
