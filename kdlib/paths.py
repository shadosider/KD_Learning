from pathlib import Path

# 项目根目录(src 的上一级),所有路径都从这里推导,避免相对路径依赖运行目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与输出目录
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def checkpoint_path(name: str) -> Path:
    """返回 checkpoint 的绝对路径。"""
    return CHECKPOINT_DIR / name


def deep_suffix(deep: bool = False) -> str:
    """深度模式的文件名后缀,用于区分浅/深 teacher 的产物。"""
    return "_deep" if deep else ""


def teacher_checkpoint(deep: bool = False) -> str:
    """teacher 权重文件名(深模式带 _deep 后缀)。"""
    return f"teacher{deep_suffix(deep)}.pt"


def kd_checkpoint(deep: bool = False) -> str:
    """蒸馏学生权重文件名(深模式带 _deep 后缀)。"""
    return f"student_kd{deep_suffix(deep)}.pt"
