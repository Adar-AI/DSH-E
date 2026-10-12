"""通用小工具：时间戳与 ID 生成。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（秒级），全库统一用它写时间字段。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "conv") -> str:
    """生成带前缀的短 ID，例如 `conv-3f2a9c1d4b5e`。"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
