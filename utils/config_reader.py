import yaml
from pathlib import Path

def load_config(path="config/config.yaml") -> dict:
    """加载YAML配置（增加路径校验）"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 关键参数校验
    required_keys = ['input_root', 'target_year', 'resolution']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"缺失必要配置项: {key}")

    return config