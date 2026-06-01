from abc import ABC, abstractmethod
from typing import Dict, List


class BaseScorer(ABC):
    """
    评分模型统一接口。

    后续所有评分器都应该支持：
    1. score 单张图评分；
    2. batch_score 多候选批量评分；
    3. get_model_info 返回模型信息。
    """

    @abstractmethod
    def score(self, image, candidate_info: Dict) -> float:
        pass

    def batch_score(self, images: List, candidate_infos: List[Dict]) -> List[float]:
        """
        默认逐张评分。
        后续真实模型可以改成真正的 batch tensor 推理。
        """
        scores = []

        for image, info in zip(images, candidate_infos):
            score = self.score(image, info)
            scores.append(score)

        return scores

    @abstractmethod
    def get_model_info(self) -> Dict:
        pass