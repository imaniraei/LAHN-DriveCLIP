from pathlib import Path
import yaml

def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML configuration must be a mapping.")
    return cfg
