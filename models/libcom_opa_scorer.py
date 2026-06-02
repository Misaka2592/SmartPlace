import os
import time
import tempfile
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class LibcomOPAScorer(BaseScorer):
    """
    使用 libcom OPAScoreModel 的真实参考模型评分器。

    输入:
        composite PIL Image
        candidate_info 中需要包含 composite_mask 或前景 bbox 信息

    输出:
        opa_score in [0, 1]
    """

    def __init__(
        self,
        device: str = "cpu",
        model_type: str = "SimOPA",
        logger: InferenceLogger = None,
    ):
        self.device = device
        self.model_type = model_type
        self.logger = logger or InferenceLogger()
        self.model_name = f"libcom.OPAScoreModel.{model_type}"
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        self.logger.section("[LibcomOPAScorer] Load model")
        self.logger.log(f"[Model] name={self.model_name}")
        self.logger.log(f"[Model] device={self.device}")
        self.logger.log(f"[Model] model_type={self.model_type}")

        try:
            from libcom import OPAScoreModel
            self.net = OPAScoreModel(device=self.device, model_type=self.model_type)
            self.is_loaded = True
            self.logger.log("[Model] load_status=success")
        except Exception as e:
            self.is_loaded = False
            self.logger.log(f"[Model] load_status=failed, error={repr(e)}")
            raise

    def _build_composite_mask_from_candidate(self, image: Image.Image, candidate_info: Dict) -> np.ndarray:
        """
        根据 candidate_info 里的前景 bbox 生成 composite mask。

        注意:
        这是兜底方案。
        更理想的方式是 composer.py 在合成时直接返回真实 mask。
        """
        w, h = image.size

        mask = np.zeros((h, w), dtype=np.uint8)

        x = int(candidate_info.get("x", 0))
        y = int(candidate_info.get("y", 0))
        fg_w = int(candidate_info.get("fg_width", 1))
        fg_h = int(candidate_info.get("fg_height", 1))

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fg_w)
        y2 = min(h, y + fg_h)

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

        return mask

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        if not self.is_loaded:
            raise RuntimeError("LibcomOPAScorer model is not loaded.")

        start = time.time()

        image = image.convert("RGB")
        comp_np = np.array(image)

        # PIL / RGB -> OpenCV / BGR, libcom 示例使用 cv2 图像也可以
        comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)

        if "composite_mask" in candidate_info and candidate_info["composite_mask"] is not None:
            comp_mask = candidate_info["composite_mask"]
            if isinstance(comp_mask, Image.Image):
                comp_mask = np.array(comp_mask.convert("L"))
            else:
                comp_mask = np.array(comp_mask).astype(np.uint8)
        else:
            comp_mask = self._build_composite_mask_from_candidate(image, candidate_info)

        raw_score = self.net(comp_bgr, comp_mask)
        score = float(raw_score)

        elapsed = time.time() - start
        candidate_id = candidate_info.get("candidate_id", "unknown")

        self.logger.log(
            f"[LibcomOPA] candidate_id={candidate_id}, "
            f"input_image_shape={comp_bgr.shape}, "
            f"input_mask_shape={comp_mask.shape}, "
            f"opa_score={score:.6f}, "
            f"inference_time={elapsed:.6f}s"
        )

        return max(0.0, min(1.0, score))

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        self.logger.section("[LibcomOPAScorer] Batch inference")

        start = time.time()
        scores = []

        for image, info in zip(images, candidate_infos):
            scores.append(self.score(image, info))

        elapsed = time.time() - start

        self.logger.log(
            f"[LibcomOPA-Batch] batch_size={len(scores)}, "
            f"scores={[round(s, 6) for s in scores]}, "
            f"batch_time={elapsed:.6f}s"
        )

        return scores

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": self.model_type,
            "is_loaded": self.is_loaded,
            "source": "BCMI/libcom OPAScoreModel",
        }