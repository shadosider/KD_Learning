"""结果缓存:参数未变时跳过重复的昂贵计算。

每个结果 JSON 都附带一个 ``meta`` 字段,记录生成该结果时的关键参数。
再次运行前读取已有文件并比对 meta:一致则复用,不一致或缺失则重跑。
"""
import json
from pathlib import Path


def write_result(path: Path, meta: dict, **payload) -> None:
    """把 meta 与数据一起写成 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, **payload}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def read_if_fresh(path: Path, meta: dict) -> dict | None:
    """文件存在且其 meta 与当前一致时,返回数据部分(不含 meta);否则返回 None。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("meta") != meta:
        return None
    return {k: v for k, v in data.items() if k != "meta"}
