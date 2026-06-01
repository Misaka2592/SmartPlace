from PIL import Image
import numpy as np


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    保证图像为 RGB 格式。
    """
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def ensure_rgba(image: Image.Image) -> Image.Image:
    """
    保证图像为 RGBA 格式。
    如果前景图没有 alpha 通道，则自动生成不透明 alpha 通道。
    """
    if image.mode == "RGBA":
        return image
    return image.convert("RGBA")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8))