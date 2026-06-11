import yaml
import os
from typing import Dict, Any

CONFIG_PATH = "configs/default.yaml"

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)