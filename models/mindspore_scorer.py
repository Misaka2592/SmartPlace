from typing import Dict

from PIL import Image


class MindSporeScorer:
    """
    MindSpore 辅助评分器接口预留。

    注意：
    这个类需要在安装了 MindSpore 的独立环境中使用。
    主 Web 应用目前不直接依赖它，避免破坏已有 Gradio 环境。
    """

    def __init__(
        self,
        ckpt_path: str = "outputs/mindspore/mindspore_aux_scorer.ckpt",
        device_target: str = "CPU",
        image_size: int = 96,
    ):
        try:
            import numpy as np
            import mindspore as ms
            from mindspore import Tensor, ops, context, load_checkpoint, load_param_into_net
            from scripts.run_mindspore_demo import SmallCNN, LABEL_NAMES, load_image_as_array
        except ImportError as e:
            raise ImportError(
                "MindSporeScorer 需要在已安装 MindSpore 的环境中运行。"
                "请使用单独的 .venv_ms 环境。"
            ) from e

        self.np = np
        self.ms = ms
        self.Tensor = Tensor
        self.ops = ops
        self.LABEL_NAMES = LABEL_NAMES
        self.load_image_as_array = load_image_as_array

        context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)

        self.ckpt_path = ckpt_path
        self.device_target = device_target
        self.image_size = image_size

        self.model = SmallCNN(num_classes=3)
        params = load_checkpoint(ckpt_path)
        load_param_into_net(self.model, params)
        self.model.set_train(False)

    def score(self, image: Image.Image, candidate_info: Dict = None) -> float:
        """
        返回“推荐”概率，作为辅助评分。
        """
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            image.convert("RGB").save(temp_path)
            arr = self.load_image_as_array(temp_path, self.image_size)
            tensor = self.Tensor(self.np.expand_dims(arr, axis=0), self.ms.float32)

            logits = self.model(tensor)
            probs = self.ops.Softmax(axis=1)(logits).asnumpy()[0]

            recommend_prob = float(probs[2])
            return recommend_prob

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)