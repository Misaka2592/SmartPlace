import json
import os
import subprocess
import time
from typing import Dict, Tuple

from PIL import Image

from utils.logger import InferenceLogger


class HandinU2NetSubprocessMatting:
    def __init__(
        self,
        python_path: str = "../handin/.venv/Scripts/python.exe",
        script_path: str = "../scripts/handin_u2net_infer_once.py",
        handin_root: str = "../handin",
        model_type: str = "u2netp",
        weight_path: str = "../handin/u2netp.pth",
        device: str = "cpu",
        threshold: float = 0.5,
        temp_dir: str = "../outputs/handin_u2net",
        timeout_seconds: int = 120,
        logger: InferenceLogger = None,
    ):
        self.python_path = os.path.normpath(python_path)
        self.script_path = script_path
        self.handin_root = os.path.normpath(handin_root)
        self.model_type = model_type
        self.weight_path = os.path.normpath(weight_path)
        self.device = device
        self.threshold = float(threshold)
        self.temp_dir = os.path.normpath(temp_dir)
        self.timeout_seconds = int(timeout_seconds)
        self.logger = logger or InferenceLogger()
        os.makedirs(self.temp_dir, exist_ok=True)

    def _extract_json_from_stdout(self, stdout: str) -> Dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError("Cannot parse JSON from handin U2Net subprocess stdout.\n" + stdout)

    def process(self, image: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(self.temp_dir, f"{timestamp}_input.png")
        rgba_path = os.path.join(self.temp_dir, f"{timestamp}_foreground_rgba.png")
        mask_path = os.path.join(self.temp_dir, f"{timestamp}_mask.png")
        image.convert("RGBA").save(input_path)

        self.logger.log(f"[DEBUG LOG BEGINS]")
        self.logger.log(f"[HandinU2Net-Subprocess] python_path={self.python_path}")
        self.logger.log(f"[HandinU2Net-Subprocess] script_path={self.script_path}")
        self.logger.log(f"[HandinU2Net-Subprocess] handin_root={self.handin_root}")
        self.logger.log(f"[HandinU2Net-Subprocess] input_path={input_path}")
        self.logger.log(f"[DEBUG LOG ENDS]")

        cmd = [
            self.python_path,
            self.script_path,
            "--input", input_path,
            "--output_rgba", rgba_path,
            "--output_mask", mask_path,
            "--handin_root", self.handin_root,
            "--model_type", self.model_type,
            "--weight", self.weight_path,
            "--device", self.device,
            "--threshold", str(self.threshold),
        ]

        start = time.time()
        self.logger.log(f"[DEBUG] START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
        self.logger.log(f"[DEBUG] CMD: {' '.join(cmd)}")

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
            self.logger.log(f"[HandinU2Net-Subprocess] timeout after {self.timeout_seconds}s")
            raise RuntimeError(f"handin U2Net inference timed out after {self.timeout_seconds}s") from exc

        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[HandinU2Net-Subprocess] failed")
            self.logger.log(proc.stderr)
            raise RuntimeError(proc.stderr)

        result = self._extract_json_from_stdout(proc.stdout)
        self.logger.log(
            f"[HandinU2Net-Subprocess] model={self.model_type}, "
            f"foreground_pixel_ratio={float(result.get('foreground_pixel_ratio', 0.0)):.6f}, "
            f"subprocess_time={elapsed:.6f}s"
        )

        foreground_rgba = Image.open(rgba_path).convert("RGBA")
        mask_preview = Image.open(mask_path).convert("L").convert("RGB")
        info = {
            "mode_used": "U2Net 自动抠图",
            "requested_mode": "U2Net 自动抠图",
            "has_alpha": True,
            "foreground_pixel_ratio": float(result.get("foreground_pixel_ratio", 0.0)),
            "input_size": tuple(result.get("input_size", foreground_rgba.size)),
            "output_size": foreground_rgba.size,
            "model_type": self.model_type,
            "weight_path": self.weight_path,
            "device": result.get("device", self.device),
            "mask_path": mask_path,
            "processed_foreground_path": rgba_path,
        }
        return foreground_rgba, mask_preview, info
