from typing import Dict, List

import numpy as np
from PIL import Image

from models.base_scorer import BaseScorer
from utils.logger import InferenceLogger


class SmartPlaceOPACalibratedScorer(BaseScorer):
    """
    SmartPlace application-level model modification.

    The scorer keeps the real LibCom OPAScoreModel as the backbone, then
    calibrates its output with placement features that are important for this
    application: boundary validity, object scale, vertical support, and local
    contact-region consistency.
    """

    def __init__(
        self,
        base_scorer: BaseScorer,
        opa_weight: float = 0.72,
        geometry_weight: float = 0.14,
        contact_weight: float = 0.08,
        support_weight: float = 0.06,
        out_of_bounds_cap: float = 0.20,
        logger: InferenceLogger = None,
    ):
        self.base_scorer = base_scorer
        self.opa_weight = float(opa_weight)
        self.geometry_weight = float(geometry_weight)
        self.contact_weight = float(contact_weight)
        self.support_weight = float(support_weight)
        self.out_of_bounds_cap = float(out_of_bounds_cap)
        self.logger = logger or InferenceLogger()
        self.model_name = "SmartPlaceOPACalibratedScorer"
        self.is_loaded = True

        total = self.opa_weight + self.geometry_weight + self.contact_weight + self.support_weight
        if total <= 0:
            raise ValueError("Calibration weights must sum to a positive value.")
        self._weight_sum = total

        self.logger.section("[SmartPlaceOPACalibratedScorer] Init")
        self.logger.log(f"[Model] name={self.model_name}")
        self.logger.log(f"[Model] base_model={self.base_scorer.get_model_info().get('model_name')}")
        self.logger.log(
            "[Model] weights="
            f"opa:{self.opa_weight},geometry:{self.geometry_weight},"
            f"contact:{self.contact_weight},support:{self.support_weight}"
        )
        self.logger.log(f"[Model] out_of_bounds_cap={self.out_of_bounds_cap}")

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _geometry_score(self, info: Dict) -> float:
        if info.get("out_of_bounds", False):
            return 0.0

        bg_w = max(1.0, float(info.get("bg_width", 1)))
        bg_h = max(1.0, float(info.get("bg_height", 1)))
        fg_w = max(1.0, float(info.get("fg_width", 1)))
        fg_h = max(1.0, float(info.get("fg_height", 1)))
        x = float(info.get("x", 0))
        y = float(info.get("y", 0))

        area_ratio = (fg_w * fg_h) / max(1.0, bg_w * bg_h)
        if area_ratio < 0.02:
            area_score = area_ratio / 0.02
        elif area_ratio <= 0.35:
            area_score = 1.0
        else:
            area_score = 1.0 - min(1.0, (area_ratio - 0.35) / 0.30)

        cx = (x + fg_w / 2.0) / bg_w
        cy = (y + fg_h / 2.0) / bg_h
        margin_score = min(cx, 1.0 - cx, cy, 1.0 - cy) / 0.12
        margin_score = self._clip01(margin_score)

        return self._clip01(0.70 * area_score + 0.30 * margin_score)

    def _support_score(self, info: Dict) -> float:
        if info.get("out_of_bounds", False):
            return 0.0

        bg_h = max(1.0, float(info.get("bg_height", 1)))
        fg_h = max(1.0, float(info.get("fg_height", 1)))
        y = float(info.get("y", 0))
        bottom_ratio = (y + fg_h) / bg_h

        if bottom_ratio < 0.28:
            return 0.20
        if bottom_ratio < 0.55:
            return 0.70
        if bottom_ratio <= 0.92:
            return 1.00
        return 0.72

    def _contact_score(self, image: Image.Image, info: Dict) -> float:
        if info.get("out_of_bounds", False):
            return 0.0

        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        x = int(round(float(info.get("x", 0))))
        y = int(round(float(info.get("y", 0))))
        fg_w = int(round(float(info.get("fg_width", 1))))
        fg_h = int(round(float(info.get("fg_height", 1))))

        x1 = max(0, x)
        x2 = min(w, x + fg_w)
        y1 = max(0, y)
        y2 = min(h, y + fg_h)
        if x2 <= x1 or y2 <= y1:
            return 0.2

        band = max(4, fg_h // 8)
        object_y1 = max(y1, y2 - band)
        object_region = arr[object_y1:y2, x1:x2]

        contact_y1 = min(h, y2)
        contact_y2 = min(h, y2 + band)
        contact_region = arr[contact_y1:contact_y2, x1:x2]

        if object_region.size == 0 or contact_region.size == 0:
            return 0.55

        obj_mean = float(object_region.mean())
        contact_mean = float(contact_region.mean())
        obj_std = float(object_region.std())
        contact_std = float(contact_region.std())

        brightness_score = 1.0 - min(1.0, abs(obj_mean - contact_mean) / 0.45)
        texture_score = 1.0 - min(1.0, abs(obj_std - contact_std) / 0.35)
        return self._clip01(0.55 * brightness_score + 0.45 * texture_score)

    def _calibrate(self, raw_opa_score: float, image: Image.Image, info: Dict) -> float:
        raw_opa_score = self._clip01(raw_opa_score)
        geometry = self._geometry_score(info)
        contact = self._contact_score(image, info)
        support = self._support_score(info)

        calibrated = (
            self.opa_weight * raw_opa_score
            + self.geometry_weight * geometry
            + self.contact_weight * contact
            + self.support_weight * support
        ) / self._weight_sum

        if info.get("out_of_bounds", False):
            calibrated = min(calibrated, self.out_of_bounds_cap)

        calibrated = self._clip01(calibrated)
        info["raw_opa_score"] = raw_opa_score
        info["smartplace_calibrated_score"] = calibrated
        info["calibration_features"] = {
            "geometry_score": geometry,
            "contact_score": contact,
            "support_score": support,
            "out_of_bounds_cap": self.out_of_bounds_cap,
        }
        return calibrated

    def score(self, image: Image.Image, candidate_info: Dict) -> float:
        raw_score = self.base_scorer.score(image, candidate_info)
        calibrated = self._calibrate(raw_score, image, candidate_info)
        self.logger.log(
            f"[SmartPlaceCalibration] candidate_id={candidate_info.get('candidate_id', 'unknown')}, "
            f"raw_opa={raw_score:.6f}, calibrated={calibrated:.6f}, "
            f"features={candidate_info.get('calibration_features')}"
        )
        return calibrated

    def explain_score(self, image: Image.Image, candidate_info: Dict) -> float:
        """
        Fast score used by occlusion explanations.

        The final candidate score is still produced by the real OPA backbone.
        During occlusion, however, repeatedly launching OPA for every patch is
        too slow for interactive use. We therefore reuse the already recorded
        raw OPA score and explain the SmartPlace calibration layer itself.
        """
        raw_score = float(candidate_info.get("raw_opa_score", candidate_info.get("smartplace_calibrated_score", 0.5)))
        return self._calibrate(raw_score, image, candidate_info)

    def batch_score(self, images: List[Image.Image], candidate_infos: List[Dict]) -> List[float]:
        self.logger.section("[SmartPlaceOPACalibratedScorer] Batch calibrated inference")
        raw_scores = self.base_scorer.batch_score(images, candidate_infos)
        calibrated_scores = []
        for image, info, raw_score in zip(images, candidate_infos, raw_scores):
            calibrated = self._calibrate(raw_score, image, info)
            calibrated_scores.append(calibrated)
            self.logger.log(
                f"[SmartPlaceCalibration] candidate_id={info.get('candidate_id', 'unknown')}, "
                f"raw_opa={raw_score:.6f}, calibrated={calibrated:.6f}, "
                f"features={info.get('calibration_features')}"
            )
        self.logger.log(
            f"[SmartPlaceCalibration-Batch] scores={[round(s, 6) for s in calibrated_scores]}"
        )
        return calibrated_scores

    def get_model_info(self) -> Dict:
        base_info = self.base_scorer.get_model_info()
        return {
            "model_name": self.model_name,
            "source": "LibCom OPAScoreModel + SmartPlace placement calibration",
            "base_model": base_info,
            "device": base_info.get("device"),
            "model_type": base_info.get("model_type"),
            "is_loaded": self.is_loaded and bool(base_info.get("is_loaded", True)),
            "calibration": {
                "opa_weight": self.opa_weight,
                "geometry_weight": self.geometry_weight,
                "contact_weight": self.contact_weight,
                "support_weight": self.support_weight,
                "out_of_bounds_cap": self.out_of_bounds_cap,
            },
        }
