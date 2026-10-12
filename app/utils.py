"""通用小工具：时间戳、ID 生成、模型输出里的 JSON 抽取。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（秒级），全库统一用它写时间字段。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "conv") -> str:
    """生成带前缀的短 ID，例如 `conv-3f2a9c1d4b5e`。"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def parse_json_block(text: str | None) -> dict | None:
    """从模型回复里抠出 JSON 对象。

    容忍三种常见情况：```json 围栏、前后有解释性文字、直接就是 JSON。
    解析不出来返回 None，由调用方决定重试还是降级，绝不抛异常。
    """
    if not text:
        return None

    candidates: list[str] = _JSON_FENCE_RE.findall(text)
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                data = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data
    return None
