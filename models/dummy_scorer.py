import os
import time
from typing import Dict, List

import numpy as np
from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class DummyScorer(BaseScorer):
    """
    第 4 版占位评分器。

    它仍然不是最终真实模型，但相比 v0.3：
    1. 保留位置规则评分；
    2. 新增简单图像内容评分；
    3. 支持遮挡实验产生非平坦热力图；
    4. 接口仍然与真实模型保持一致。
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

        self.model_name = "DummyScorerV4"
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        self.logger.section("[DummyScorerV4] Load model")

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
        image = image.convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = np.expand_dims(arr, axis=0)

        return tensor

    def _position_forward(self, candidate_info: Dict) -> float:
        """
        位置规则评分。
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

        score = (
            0.35 * horizontal_score
            + 0.40 * vertical_score
            + 0.25 * scale_score
        )

        return float(max(0.0, min(1.0, score)))

    def _image_context_forward(self, image: Image.Image, candidate_info: Dict) -> float:
        """
        简单图像内容评分。

        目的不是替代真实模型，而是让遮挡实验能够体现：
        - 前景附近区域被遮挡后，score 会发生变化；
        - 热力图能集中在前景和接触区域附近。
        """
        image = image.convert("RGB")
        arr = np.array(image).astype(np.float32) / 255.0

        h, w = arr.shape[:2]

        x = int(candidate_info.get("x", 0))
        y = int(candidate_info.get("y", 0))
        fg_w = int(candidate_info.get("fg_width", 1))
        fg_h = int(candidate_info.get("fg_height", 1))

        # 限制 bbox 范围
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fg_w)
        y2 = min(h, y + fg_h)

        if x2 <= x1 or y2 <= y1:
            return 0.2

        object_region = arr[y1:y2, x1:x2]

        # 接触区域：前景底部附近的一条区域
        contact_y1 = max(0, y2 - max(4, fg_h // 5))
        contact_y2 = min(h, y2 + max(4, fg_h // 10))
        contact_x1 = max(0, x1)
        contact_x2 = min(w, x2)

        contact_region = arr[contact_y1:contact_y2, contact_x1:contact_x2]

        if object_region.size == 0 or contact_region.size == 0:
            return 0.2

        # 颜色变化越极端，说明可能被遮挡或不稳定
        object_std = float(object_region.std())
        contact_std = float(contact_region.std())

        # 亮度不能太暗也不能太亮
        contact_brightness = float(contact_region.mean())
        brightness_score = 1.0 - abs(contact_brightness - 0.5) * 1.2
        brightness_score = max(0.0, min(1.0, brightness_score))

        # 适度纹理较好，完全纯色或过乱都扣分
        texture_score = 1.0 - abs(contact_std - 0.18) * 2.0
        texture_score = max(0.0, min(1.0, texture_score))

        object_score = 1.0 - abs(object_std - 0.22) * 1.5
        object_score = max(0.0, min(1.0, object_score))

        score = (
            0.45 * brightness_score
            + 0.35 * texture_score
            + 0.20 * object_score
        )

        return float(max(0.0, min(1.0, score)))

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        """
        单张候选图评分。
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        start_time = time.time()

        input_tensor = self._preprocess(image)

        position_score = self._position_forward(candidate_info)
        image_score = self._image_context_forward(image, candidate_info)

        # 位置仍然占主要部分，图像内容用于让解释图有效。
        raw_output = 0.75 * position_score + 0.25 * image_score
        score = float(max(0.0, min(1.0, raw_output)))

        elapsed = time.time() - start_time

        candidate_id = candidate_info.get("candidate_id", "unknown")

        self.logger.log(
            f"[Inference] candidate_id={candidate_id}, "
            f"input_tensor_shape={input_tensor.shape}, "
            f"position_score={position_score:.6f}, "
            f"image_score={image_score:.6f}, "
            f"raw_output={raw_output:.6f}, "
            f"score={score:.6f}, "
            f"inference_time={elapsed:.6f}s"
        )

        return score

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        self.logger.section("[DummyScorerV4] Batch inference")

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