"""图片理解：把客户上传的报错截图 / 实物照片转成结构化故障描述。

流程：读图 → base64 内联（不外传图床）→ 多模态模型 → JSON → 紧凑中文描述。
失败一律返回 `ok=False` 且带可读原因，由对话流程决定「请客户补充文字描述」，
不让接口 500。
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.config import Settings, get_settings, is_local_url
from app.llm.prompts import VISION_SYSTEM, build_vision_prompt
from app.llm.providers import (
    OpenAICompatProvider,
    ProviderError,
    build_vision_providers,
)
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
    local: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    image_info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "local": self.local,
            "detail": self.detail,
            "image_info": self.image_info,
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
    """把图片原样编码成 data URL（不缩放；缩放走 prepare_image）。"""
    path = Path(path)
    mime = detect_image_mime(path)
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def prepare_image(
    path: Path | str, *, max_side: int = 1280
) -> tuple[str, dict[str, Any]]:
    """读取图片并按需等比缩放，返回 (data URL, 元信息)。

    为什么要缩放：本地小视觉模型对超过 ~1280px 的图，推理耗时随像素数迅速上涨，
    而识别报错码这类任务在 1280px 已经足够。透明图仍用 PNG，其余转 JPEG 以减小体积。
    **不需要缩放时原样发送原始字节**（不重编码，省时且不损画质）；
    缩放失败时同样退回原图，绝不因为"优化"而让识别不可用。
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    mime = detect_image_mime(path)
    info: dict[str, Any] = {
        "original_bytes": len(raw_bytes),
        "scaled": False,
        "max_side": max_side,
    }

    original_url = (
        f"data:{mime};base64,{base64.b64encode(raw_bytes).decode('ascii')}"
    )

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            info["original_size"] = [width, height]
            info["sent_size"] = [width, height]

            if not max_side or max_side <= 0 or max(width, height) <= max_side:
                info["sent_bytes"] = len(raw_bytes)
                return original_url, info

            ratio = max_side / float(max(width, height))
            target = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resized = image.resize(target, Image.LANCZOS)
            info["scaled"] = True
            info["sent_size"] = list(resized.size)

            buffer = io.BytesIO()
            if resized.mode in ("RGBA", "LA", "P"):
                resized.convert("RGBA").save(buffer, format="PNG", optimize=True)
                mime = "image/png"
            else:
                resized.convert("RGB").save(
                    buffer, format="JPEG", quality=88, optimize=True
                )
                mime = "image/jpeg"
            payload = buffer.getvalue()
            info["sent_bytes"] = len(payload)
            return (
                f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}",
                info,
            )
    except Exception as exc:  # noqa: BLE001 - 缩放失败退回原图
        logger.warning("图片缩放失败，改用原图：%s", exc)

    info["sent_bytes"] = len(raw_bytes)
    return original_url, info


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
    providers: Sequence[OpenAICompatProvider] | None = None,
    settings: Settings | None = None,
    context: str = "",
) -> VisionResult:
    """分析一张图片。任何失败都返回 ok=False，不抛异常。

    通道顺序：`providers`（或单个 `provider`）→ 配置里的视觉链（本地 Ollama 优先，
    可按 `VISION_AUTO_CLOUD_FALLBACK` 自动回退云端）。逐个尝试，第一个成功即返回。
    """
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
        data_url, image_info = prepare_image(path, max_side=settings.vision_max_side)
    except ValueError as exc:
        return VisionResult(ok=False, error=str(exc))
    except OSError as exc:
        return VisionResult(ok=False, error=f"读取图片失败：{exc}")

    if providers is not None:
        candidates = list(providers)
    elif provider is not None:
        candidates = [provider]
    else:
        candidates = build_vision_providers(settings)

    messages = build_vision_messages(context, data_url)
    errors: list[str] = []

    for candidate in candidates:
        available, reason = candidate.available()
        if not available:
            errors.append(f"{candidate.name} 不可用：{reason}")
            continue

        try:
            raw = candidate.chat(messages)
        except ProviderError as exc:
            logger.warning("图片理解失败（%s）：%s", candidate.name, exc)
            errors.append(f"{candidate.name}：{exc}")
            continue

        detail = parse_json_block(raw) or {}
        if detail.get("error_codes"):
            detail["error_codes"] = normalize_error_codes(detail["error_codes"])
        summary = describe_detail(detail) if detail else raw.strip()[:500]

        if not summary:
            errors.append(f"{candidate.name}：模型没有返回可用的图片描述")
            continue

        local = is_local_url(candidate.base_url)
        logger.info(
            "图片识别成功：通道=%s 模型=%s 本地=%s 缩放=%s",
            candidate.name,
            candidate.model,
            local,
            image_info.get("scaled"),
        )
        return VisionResult(
            ok=True,
            summary=summary,
            raw=raw,
            provider=candidate.name,
            model=candidate.model,
            local=local,
            detail=detail,
            image_info=image_info,
        )

    return VisionResult(
        ok=False,
        error="；".join(errors) or "没有可用的图片理解通道",
        image_info=image_info,
    )
