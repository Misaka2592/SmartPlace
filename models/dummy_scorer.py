import os
import time
from typing import Dict, List

import numpy as np
from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class DummyScorer(BaseScorer):
    """
    第 2 版占位评分器。

    它仍然是规则评分器，但接口和日志形式尽量模拟真实深度学习模型。

    作用：
    1. 跑通完整应用流程；
    2. 提供“模型加载、输入张量、输出分数、推理时间”的演示证据；
    3. 为后续替换 OPA/FOPA 做接口准备。
    """

    def __init__(
        self,
        weight_path: str = "weights/dummy_scorer_rule_based.pth",
        device: str = "cpu",
        input_size: int = 224,
        logger: InferenceLogger = None,
    ):
        self.weight_path = weight_path
        self.device = device
        self.input_size = int(input_size)
        self.logger = logger or InferenceLogger()

        self.model_name = "DummyScorerV2"
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """
        模拟真实模型加载。
        现在不需要真实权重文件存在。
        后续 OPA/FOPA 会在这里真实加载权重。
        """
        self.logger.section("[DummyScorerV2] Load model")

        self.logger.log(f"[Model] name={self.model_name}")
        self.logger.log(f"[Model] weight_path={self.weight_path}")
        self.logger.log(f"[Model] device={self.device}")
        self.logger.log(f"[Model] input_size={self.input_size}")

        if os.path.exists(self.weight_path):
            self.logger.log("[Model] weight_status=found")
        else:
            self.logger.log("[Model] weight_status=not_found, using rule-based dummy scorer")

        self.is_loaded = True
        self.logger.log("[Model] load_status=success")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        模拟图像预处理：
        PIL RGB image -> resized numpy tensor -> CHW -> batch tensor

        返回 shape:
            (1, 3, input_size, input_size)
        """
        image = image.convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        arr = np.array(image).astype(np.float32) / 255.0

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))

        # add batch dim
        tensor = np.expand_dims(arr, axis=0)

        return tensor

    def _rule_forward(self, candidate_info: Dict) -> float:
        """
        规则打分，模拟 raw output。
        """
        bg_w = candidate_info.get("bg_width", 1)
        bg_h = candidate_info.get("bg_height", 1)
        fg_w = candidate_info.get("fg_width", 1)
        fg_h = candidate_info.get("fg_height", 1)
        x = candidate_info.get("x", 0)
        y = candidate_info.get("y", 0)

        out_of_bounds = candidate_info.get("out_of_bounds", False)

        if out_of_bounds:
            return 0.15

        cx = x + fg_w / 2
        cy = y + fg_h / 2

        center_x = bg_w / 2
        horizontal_distance = abs(cx - center_x) / max(1, bg_w / 2)
        horizontal_score = 1.0 - horizontal_distance

        vertical_ratio = cy / max(1, bg_h)

        if vertical_ratio < 0.25:
            vertical_score = 0.25
        elif vertical_ratio < 0.45:
            vertical_score = 0.55
        elif vertical_ratio < 0.85:
            vertical_score = 0.90
        else:
            vertical_score = 0.65

        area_ratio = (fg_w * fg_h) / max(1, bg_w * bg_h)

        if area_ratio < 0.02:
            scale_score = 0.45
        elif area_ratio > 0.45:
            scale_score = 0.35
        else:
            scale_score = 0.85

        raw_output = (
            0.35 * horizontal_score
            + 0.40 * vertical_score
            + 0.25 * scale_score
        )

        raw_output = max(0.0, min(1.0, raw_output))

        return float(raw_output)

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        """
        单张候选图评分。
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        start_time = time.time()

        input_tensor = self._preprocess(image)
        raw_output = self._rule_forward(candidate_info)

        # 当前 raw_output 已经是 0~1 概率。
        # 后续真实模型如果输出 logits，可以在这里 sigmoid/softmax。
        score = float(raw_output)

        elapsed = time.time() - start_time

        candidate_id = candidate_info.get("candidate_id", "unknown")

        self.logger.log(
            f"[Inference] candidate_id={candidate_id}, "
            f"input_tensor_shape={input_tensor.shape}, "
            f"raw_output={raw_output:.6f}, "
            f"score={score:.6f}, "
            f"inference_time={elapsed:.6f}s"
        )

        return score

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        """
        批量评分接口。

        当前仍然逐张处理，但日志形式按照 batch 组织。
        后续真实模型可以在这里改为 batch tensor 推理。
        """
        self.logger.section("[DummyScorerV2] Batch inference")

        batch_start = time.time()

        scores = []

        for image, info in zip(images, candidate_infos):
            score = self.score(image, info)
            scores.append(score)

        batch_elapsed = time.time() - batch_start

        self.logger.log(
            f"[BatchInference] batch_size={len(images)}, "
            f"scores={[round(s, 6) for s in scores]}, "
            f"batch_time={batch_elapsed:.6f}s"
        )

        return scores

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "weight_path": self.weight_path,
            "device": self.device,
            "input_size": self.input_size,
            "is_loaded": self.is_loaded,
        }