import os
import time
from typing import Optional


class InferenceLogger:
    """
    推理日志工具。

    作用：
    1. 在终端打印模型推理证据；
    2. 同时保存到 outputs/logs；
    3. 方便课堂展示“模型确实被调用”。
    """

    def __init__(self, log_dir: str = "outputs/logs", enable_file_log: bool = True):
        self.log_dir = log_dir
        self.enable_file_log = enable_file_log

        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"inference_{timestamp}.log")

    def log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"

        print(line)

        if self.enable_file_log:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def section(self, title: str):
        self.log("=" * 80)
        self.log(title)
        self.log("=" * 80)

    def get_log_path(self) -> Optional[str]:
        if self.enable_file_log:
            return self.log_path
        return None