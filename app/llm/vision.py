"""图片理解：把客户上传的报错截图 / 实物照片转成结构化故障描述。

流程：读图 → base64 内联（不外传图床）→ 多模态模型 → JSON → 紧凑中文描述。
失败一律返回 `ok=False` 且带可读原因，由对话流程决定「请客户补充文字描述」，
不让接口 500。
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.llm.prompts import VISION_SYSTEM, build_vision_prompt
from app.llm.providers import OpenAICompatProvider, ProviderError, build_vision_provider
from app.utils import parse_json_block

logger = logging.getLogger(__name__)

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

SUPPORTED_IMAGE_SUFFIXES = set(MIME_BY_SUFFIX)
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_DETAIL_LABELS = {
    "kind": "图片类型",
    "product_model": "产品型号",
    "error_codes": "错误码或提示",
    "screen_location": "报错位置",
    "visible_damage": "可见异常",
    "other_text": "其他文字",
    "summary": "概述",
}

# 从「ERROR E05」这类描述里抠出真正的错误码（与工单抽取用的是同一套规则）
_CODE_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,3}\d{1,3})(?![A-Za-z0-9])")


def normalize_error_codes(values: object) -> list[str]:
    """规范化图片返回的错误码。

    模型常把整句提示当成错误码返回（如 `"ERROR E05"`、`"报错码：E05"`），
    直接拿去检索会把 `error` 这类噪声词混进查询，因此这里只保留真正的编码部分；
    抠不出编码时保留原文，避免丢信息。
    """
    if values is None:
        return []
    if isinstance(values, str):
        items: list[str] = [values]
    elif isinstance(values, (list, tuple, set)):
        items = [str(item) for item in values]
    else:
        items = [str(values)]

    codes: dict[str, None] = {}
    for item in items:
        text = str(item).strip().upper()
        if not text:
            continue
        matches = _CODE_IN_TEXT_RE.findall(text)
        if matches:
            for match in matches:
                codes.setdefault(match, None)
        else:
            codes.setdefault(text, None)
    return list(codes)


@dataclass(frozen=True)
class VisionResult:
    """图片理解结果。`summary` 是给提示词用的紧凑中文描述。"""

    ok: bool
    summary: str = ""
    raw: str = ""
    error: str = ""
    model: str = ""
    provider: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "detail": self.detail,
        }


def detect_image_mime(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in MIME_BY_SUFFIX:
        raise ValueError(
            f"不支持的图片类型 {suffix or '(无扩展名)'}，支持 "
            + "、".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        )
    return MIME_BY_SUFFIX[suffix]


def encode_image_data_url(path: Path | str) -> str:
    """把图片编码成 data URL（内联给模型，不依赖任何图床）。"""
    path = Path(path)
    mime = detect_image_mime(path)
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_vision_messages(context: str, data_url: str) -> list[dict]:
    """组装多模态请求（OpenAI 兼容的 content 数组格式）。"""
    return [
        {"role": "system", "content": VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_vision_prompt(context),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def describe_detail(detail: dict[str, Any]) -> str:
    """把结构化字段拼成紧凑中文描述（给回答提示词用）。"""
    lines: list[str] = []
    for key, label in _DETAIL_LABELS.items():
        value = detail.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = "、".join(str(item) for item in value if item)
        if not value:
            continue
        lines.append(f"{label}：{value}")

    confidence = detail.get("confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"识别置信度：{round(float(confidence), 2)}")
    return "\n".join(lines)


def analyze_image(
    image_path: Path | str,
    *,
    provider: OpenAICompatProvider | None = None,
    settings: Settings | None = None,
    context: str = "",
) -> VisionResult:
    """分析一张图片。任何失败都返回 ok=False，不抛异常。"""
    settings = settings or get_settings()
    path = Path(image_path)

    if not path.is_file():
        return VisionResult(ok=False, error=f"图片不存在：{path}")

    try:
        size = path.stat().st_size
    except OSError as exc:
        return VisionResult(ok=False, error=f"读取图片失败：{exc}")

    if size > MAX_IMAGE_BYTES:
        return VisionResult(
            ok=False,
            error=f"图片过大（{size / 1024 / 1024:.1f}MB），请压缩到 8MB 以内",
        )

    try:
        data_url = encode_image_data_url(path)
    except ValueError as exc:
        return VisionResult(ok=False, error=str(exc))

    provider = provider or build_vision_provider(settings)
    available, reason = provider.available()
    if not available:
        return VisionResult(
            ok=False,
            error=f"图片理解通道不可用：{reason}",
            provider=provider.name,
            model=provider.model,
        )

    messages = build_vision_messages(context, data_url)
    try:
        raw = provider.chat(messages)
    except ProviderError as exc:
        logger.warning("图片理解失败：%s", exc)
        return VisionResult(
            ok=False, error=str(exc), provider=provider.name, model=provider.model
        )

    detail = parse_json_block(raw) or {}
    if detail.get("error_codes"):
        detail["error_codes"] = normalize_error_codes(detail["error_codes"])
    summary = describe_detail(detail) if detail else raw.strip()[:500]

    if not summary:
        return VisionResult(
            ok=False,
            error="模型没有返回可用的图片描述",
            raw=raw,
            provider=provider.name,
            model=provider.model,
        )

    return VisionResult(
        ok=True,
        summary=summary,
        raw=raw,
        provider=provider.name,
        model=provider.model,
        detail=detail,
    )
