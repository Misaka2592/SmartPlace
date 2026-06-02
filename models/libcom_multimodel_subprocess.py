import json
import os
import subprocess
import time
from typing import Dict, List, Tuple

from PIL import Image

from utils.logger import InferenceLogger


class LibcomMultiModelSubprocess:
    """
    Runs optional LibCom models in an isolated subprocess.

    This wrapper keeps the main Gradio app stable while exposing FOPA, FOS,
    HarmonyScore, PCTNet, and LBM as advanced SmartPlace evidence modules.
    """

    def __init__(
        self,
        python_path: str = ".venv_libcom/Scripts/python.exe",
        script_path: str = "scripts/libcom_multi_model_infer.py",
        device: str = "cuda:0",
        temp_dir: str = "outputs/libcom_multimodel",
        logger: InferenceLogger = None,
    ):
        self.python_path = python_path
        self.script_path = script_path
        self.device = device
        self.temp_dir = temp_dir
        self.logger = logger or InferenceLogger()
        os.makedirs(self.temp_dir, exist_ok=True)

    def _extract_json_from_stdout(self, stdout: str) -> Dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError("Cannot parse JSON from LibCom multi-model stdout.\n" + stdout)

    def _save_inputs(
        self,
        background: Image.Image,
        foreground: Image.Image,
        composite: Image.Image,
        composite_mask: Image.Image,
        bbox: List[int],
        run_id: str,
    ) -> Dict[str, str]:
        run_dir = os.path.join(self.temp_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        background_path = os.path.join(run_dir, "background.png")
        foreground_path = os.path.join(run_dir, "foreground.png")
        foreground_mask_path = os.path.join(run_dir, "foreground_mask.png")
        composite_path = os.path.join(run_dir, "top1_composite.png")
        composite_mask_path = os.path.join(run_dir, "top1_composite_mask.png")

        background.convert("RGB").save(background_path)

        foreground_rgba = foreground.convert("RGBA")
        foreground_rgba.convert("RGB").save(foreground_path)
        foreground_rgba.getchannel("A").convert("L").save(foreground_mask_path)

        composite.convert("RGB").save(composite_path)
        composite_mask.convert("L").save(composite_mask_path)

        return {
            "run_dir": run_dir,
            "background": background_path,
            "foreground": foreground_path,
            "foreground_mask": foreground_mask_path,
            "composite": composite_path,
            "composite_mask": composite_mask_path,
            "bbox": json.dumps([int(v) for v in bbox]),
        }

    def run(
        self,
        background: Image.Image,
        foreground: Image.Image,
        composite: Image.Image,
        composite_mask: Image.Image,
        candidate_info: Dict,
        models: List[str],
        lbm_steps: int = 4,
        lbm_resolution: int = 768,
        run_id: str = None,
    ) -> Dict:
        if not models:
            return {"ok": False, "results": [], "message": "No optional LibCom models selected."}

        run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        x1 = int(candidate_info["x"])
        y1 = int(candidate_info["y"])
        x2 = x1 + int(candidate_info["fg_width"])
        y2 = y1 + int(candidate_info["fg_height"])

        paths = self._save_inputs(
            background=background,
            foreground=foreground,
            composite=composite,
            composite_mask=candidate_info["composite_mask"],
            bbox=[x1, y1, x2, y2],
            run_id=run_id,
        )

        cmd = [
            self.python_path,
            self.script_path,
            "--background", paths["background"],
            "--foreground", paths["foreground"],
            "--foreground_mask", paths["foreground_mask"],
            "--composite", paths["composite"],
            "--composite_mask", paths["composite_mask"],
            "--bbox", paths["bbox"],
            "--device", self.device,
            "--output_dir", paths["run_dir"],
            "--lbm_steps", str(int(lbm_steps)),
            "--lbm_resolution", str(int(lbm_resolution)),
            "--models",
            *models,
        ]

        self.logger.section("[LibcomMultiModelSubprocess] Run optional models")
        self.logger.log(f"[Models] {models}")
        self.logger.log(f"[OutputDir] {paths['run_dir']}")

        start = time.time()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            self.logger.log("[LibcomMultiModelSubprocess] failed")
            self.logger.log("[stderr]")
            self.logger.log(proc.stderr)
            self.logger.log("[stdout]")
            self.logger.log(proc.stdout)
            raise RuntimeError(proc.stderr)

        output = self._extract_json_from_stdout(proc.stdout)
        output["elapsed_sec"] = elapsed
        output["run_dir"] = paths["run_dir"]
        self.logger.log(f"[LibcomMultiModelSubprocess] elapsed={elapsed:.3f}s")
        return output

    def build_ui_payload(self, output: Dict) -> Tuple[str, List[Tuple[str, str]]]:
        lines = []
        gallery = []
        if not output or not output.get("results"):
            return "未运行 LibCom 增强模型。", gallery

        lines.append("【LibCom 多模型增强结果】")
        lines.append(f"输出目录：{output.get('run_dir', output.get('output_dir', ''))}")
        lines.append(f"总耗时：{float(output.get('total_time_sec', output.get('elapsed_sec', 0.0))):.3f}s")
        lines.append("")

        for item in output["results"]:
            name = item.get("model", "unknown")
            model_type = item.get("model_type")
            title = f"{name}" + (f" / {model_type}" if model_type else "")
            if item.get("ok"):
                lines.append(f"- {title}: 成功")
                if "score" in item:
                    lines.append(f"  分数：{float(item['score']):.4f}")
                if item.get("preview_path"):
                    gallery.append((item["preview_path"], f"{title} 位置热力图"))
                if item.get("output_path"):
                    gallery.append((item["output_path"], f"{title} 协调结果"))
                if item.get("bboxes"):
                    lines.append(f"  FOPA候选框数量：{len(item['bboxes'])}")
            else:
                lines.append(f"- {title}: 失败")
                lines.append(f"  原因：{item.get('error', '')}")

        return "\n".join(lines), gallery

