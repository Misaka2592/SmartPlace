import zipfile
import os

from typing import List
from datetime import datetime

import utils.config as utcfg

from utils.logger import InferenceLogger

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

cfg = utcfg.load_config()
output_cfg = cfg.get("output", {})
LOG_DIR = output_cfg.get("log_dir", None)
ZIP_DIR = output_cfg.get("zip_dir", None)
ZIP_PATH = os.path.join(ZIP_DIR, f"导出报告-{timestamp}.zip")
ENABLE_FILE_LOG = output_cfg.get("enable_log", True)
ENABLE_FILE_ZIP = output_cfg.get("enable_zip", True)

def zip_folder(output_path : str = ZIP_PATH, files : List[str] = []) -> str:
    if not ENABLE_FILE_ZIP:
        raise RuntimeError("文件压缩功能未启用，请在配置文件中设置 enable_zip: true")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = InferenceLogger(log_dir=LOG_DIR, enable_file_log=ENABLE_FILE_LOG)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"文件不存在: {file}")

            arcname = os.path.basename(file)
            zipf.write(file, arcname=arcname)
            logger.log(f"[ZIPPER] 压缩文件: {file} -> {arcname}")

    return output_path
