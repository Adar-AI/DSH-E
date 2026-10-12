"""会话结束时的工单字段抽取：大模型结构化 + Pydantic 校验 + 规则兜底。

三级可靠性设计（对应需求第 8 节「非法输出不落库或标记待人工补录」）：

1. 让模型按固定 JSON 结构输出，`parse_json_block` 容忍围栏与前后废话；
2. Pydantic 校验并规范化（列表转字符串、紧急程度纠偏），失败则**重试一次**并强调"只输出 JSON"；
3. 两次都失败或模型不可用时，**回退到规则版草稿**（`tickets/draft.py`），
   保证任何情况下工单都有内容，并把失败原因记录下来供人工核对。

最终字段 = 模型结果 ∪ 规则草稿（**模型值优先**，模型留空的字段用规则结果补齐）。
这样既有模型的语义归纳（诉求、现象、建议），又不会漏掉规则稳定抓到的错误码与型号；
紧急程度按"取更高一级"合并，报错码取并集。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.config import Settings, get_settings
from app.llm.prompts import build_ticket_messages
from app.llm.providers import ChatChain, ProviderError
from app.tickets.draft import completeness, merge_draft
from app.utils import parse_json_block

logger = logging.getLogger(__name__)

URGENCY_VALUES = ("普通", "较急", "紧急")
ROLE_LABELS = {"user": "客户", "assistant": "客服", "system": "系统"}


class TicketFields(BaseModel):
    """工单结构化字段（模型输出必须能落进这个模型）。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    customer_request: str = ""
    product_model: str = ""
    fault_phenomenon: str = ""
    error_info: str = ""
    tried_actions: str = ""
    urgency: str = "普通"
    suggestion: str = ""

    @field_validator(
        "customer_request",
        "product_model",
        "fault_phenomenon",
        "error_info",
        "tried_actions",
        "suggestion",
        mode="after",
    )
    @classmethod
    def _to_single_line(cls, value: str) -> str:
        """字段值压成单行：工单台是按行展示的，换行会撑坏排版。"""
        if not isinstance(value, str):
            return value
        collapsed = " ".join(part.strip() for part in value.splitlines() if part.strip())
        return collapsed.strip()

    @field_validator("error_info", "tried_actions", "fault_phenomenon", mode="before")
    @classmethod
    def _join_sequences(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return "、".join(str(item).strip() for item in value if str(item).strip())
        return value

    @field_validator("urgency", mode="before")
    @classmethod
    def _normalize_urgency(cls, value: Any) -> str:
        text = str(value or "").strip()
        if text in URGENCY_VALUES:
            return text
        for level in ("紧急", "较急"):
            if level in text:
                return level
        if "急" in text:
            return "较急"
        return "普通"

    def as_dict(self) -> dict[str, str]:
        return {key: value for key, value in self.model_dump().items()}


@dataclass
class TicketExtraction:
    """一次工单抽取的结果与来源说明。"""

    fields: dict[str, Any] = field(default_factory=dict)
    source: str = "draft"  # llm | draft
    attempts: int = 0
    error: str = ""
    raw: str = ""
    filled: tuple[int, int] = (0, 0)

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "source": self.source,
            "attempts": self.attempts,
            "error": self.error,
            "filled": list(self.filled),
        }


def build_transcript(
    messages: Sequence[dict], *, limit: int = 20, max_chars: int = 4000
) -> str:
    """把消息列表拼成抽取用的对话记录（含图片识别结论）。"""
    lines: list[str] = []
    for item in list(messages)[-limit:]:
        role = ROLE_LABELS.get(str(item.get("role")), str(item.get("role")))
        text = str(item.get("text") or "").strip()

        vision = item.get("vision") or {}
        detail = vision.get("detail") if isinstance(vision.get("detail"), dict) else vision
        bits: list[str] = []
        if isinstance(detail, dict):
            codes = detail.get("error_codes")
            if codes:
                if isinstance(codes, str):
                    codes = [codes]
                bits.append("错误码 " + "、".join(str(code) for code in codes))
            if detail.get("product_model"):
                bits.append("型号 " + str(detail["product_model"]))
            if detail.get("summary"):
                bits.append(str(detail["summary"]))
        note = f"（客户上传了图片，识别结果：{'；'.join(bits)}）" if bits else (
            "（客户上传了图片）" if item.get("image_path") else ""
        )

        if not text and not note:
            continue
        lines.append(f"{role}：{text}{note}")

    transcript = "\n".join(lines)
    return transcript if len(transcript) <= max_chars else transcript[-max_chars:]


def extract_ticket(
    messages: Sequence[dict],
    *,
    draft: dict[str, Any] | None = None,
    chat_chain: ChatChain | None = None,
    settings: Settings | None = None,
    max_attempts: int = 2,
) -> TicketExtraction:
    """抽取工单字段。任何失败都返回可落库的结果，绝不抛异常。"""
    settings = settings or get_settings()
    base = merge_draft({}, draft or {})
    transcript = build_transcript(messages)

    if not transcript.strip():
        return TicketExtraction(
            fields=base,
            source="draft",
            filled=completeness(base),
            error="没有可抽取的对话内容",
        )
    if chat_chain is None:
        return TicketExtraction(fields=base, source="draft", filled=completeness(base))

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            outcome = chat_chain.chat(
                build_ticket_messages(transcript, base, strict=attempt > 1),
                max_tokens=600,
                temperature=0.0,
            )
        except ProviderError as exc:
            last_error = f"模型调用失败：{exc}"
            break
        except Exception as exc:  # noqa: BLE001 - 抽取失败必须能兜底
            last_error = f"模型调用异常：{exc}"
            break

        raw = getattr(outcome, "text", None) or str(outcome)
        data = parse_json_block(raw)
        if data is None:
            last_error = "模型没有返回可解析的 JSON"
            logger.warning("工单抽取第 %d 次未返回 JSON", attempt)
            continue

        try:
            fields = TicketFields.model_validate(data)
        except ValidationError as exc:
            last_error = f"字段校验失败：{exc.error_count()} 处"
            logger.warning("工单抽取第 %d 次校验失败：%s", attempt, exc)
            continue

        # 模型结果作为主体，规则草稿用于补齐模型留空的字段
        merged = merge_draft(
            {key: value for key, value in fields.as_dict().items() if value}, base
        )
        return TicketExtraction(
            fields=merged,
            source="llm",
            attempts=attempt,
            raw=raw,
            filled=completeness(merged),
        )

    logger.info("工单抽取回退到规则草稿：%s", last_error)
    return TicketExtraction(
        fields=base,
        source="draft",
        attempts=max_attempts,
        error=last_error,
        filled=completeness(base),
    )


__all__ = [
    "TicketExtraction",
    "TicketFields",
    "build_transcript",
    "extract_ticket",
    "URGENCY_VALUES",
]
