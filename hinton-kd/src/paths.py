from pathlib import Path

# 项目根目录(src 的上一级),所有路径都从这里推导,避免相对路径依赖运行目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与输出目录
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def checkpoint_path(name: str) -> Path:
    """返回 checkpoint 的绝对路径。"""
    return CHECKPOINT_DIR / name
