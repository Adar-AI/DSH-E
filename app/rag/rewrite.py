"""查询改写：把用户问题变成 2~3 个检索式，并支持可选的 LLM 实体对齐。

规则版（默认，0ms，`LLM_ALIGN_ENABLED=false`）：

1. **对齐后的查询**：把上下文实体（型号）补进问题，解决"缺主语"；
2. **原问题**：保留原始措辞，避免改写引入偏差；
3. **错误码变体**：问题里出现错误码时补一条"错误码 + 处理"，便于命中报错码对照表。

LLM 版（`LLM_ALIGN_ENABLED=true`）：**一次调用**同时产出意图、实体与查询变体
（不为"LLM 改写查询"再多花一次调用）；任何失败都退回规则版，绝不影响主链路。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.llm.prompts import build_align_messages
from app.llm.providers import ChatChain, ProviderError
from app.rag.confidence import EntityHint
from app.utils import parse_json_block

logger = logging.getLogger(__name__)

INTENTS = ("售后咨询", "故障报修", "投诉", "闲聊", "无关")


@dataclass
class AlignmentResult:
    """LLM 对齐结果。"""

    intent: str = "售后咨询"
    entities: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    confidence: str = ""
    reason: str = ""
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "queries": self.queries,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def build_query_variants(
    question: str,
    *,
    entities: Sequence[str] = (),
    error_codes: Sequence[str] = (),
    max_variants: int = 3,
) -> list[str]:
    """规则版查询变体（去重、保序、限长）。"""
    question = (question or "").strip()
    variants: list[str] = []

    def add(value: str) -> None:
        text = " ".join(str(value or "").split())
        if text and text not in variants:
            variants.append(text[:400])

    entity_text = " ".join(str(item).strip() for item in entities if str(item or "").strip())
    if entity_text and entity_text.upper() not in question.upper():
        add(f"{entity_text} {question}")
    add(question)

    codes = [str(code).strip() for code in error_codes if str(code or "").strip()]
    if codes:
        add(f"{' '.join(codes)} 报错 处理 方法")

    return variants[: max(1, int(max_variants))]


def align_with_llm(
    question: str,
    *,
    context_entities: Sequence[EntityHint] = (),
    chat_chain: ChatChain | None = None,
    settings: Settings | None = None,
) -> AlignmentResult | None:
    """可选的 LLM 实体对齐；失败一律返回 None（调用方退回规则版）。"""
    settings = settings or get_settings()
    if chat_chain is None:
        return None

    entity_names = [item.name for item in context_entities]
    try:
        outcome = chat_chain.chat(
            build_align_messages(question, entity_names), max_tokens=300, temperature=0.0
        )
    except ProviderError as exc:
        logger.warning("LLM 实体对齐失败：%s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - 对齐失败不能影响主链路
        logger.warning("LLM 实体对齐异常：%s", exc)
        return None

    raw = getattr(outcome, "text", None) or str(outcome)
    data = parse_json_block(raw)
    if not data:
        logger.info("LLM 实体对齐没有返回可解析 JSON，退回规则版")
        return None

    def as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.replace("、", ",").split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item or "").strip()]
        return []

    intent = str(data.get("intent") or "").strip()
    confidence = str(data.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""

    entities = as_list(data.get("entities"))
    queries = as_list(data.get("queries"))[:3]

    # 什么都没给出（既没实体也没查询式也没置信度）→ 当作没对齐成功，让规则结果生效
    if not entities and not queries and not confidence:
        logger.info("LLM 实体对齐没有给出可用信息，退回规则版")
        return None

    return AlignmentResult(
        intent=intent if intent in INTENTS else "售后咨询",
        entities=entities,
        queries=queries,
        confidence=confidence,
        reason=str(data.get("reason") or "").strip(),
        raw=raw,
    )


def merge_alignment(
    *,
    rule_queries: Sequence[str],
    rule_entities: Sequence[str],
    alignment: AlignmentResult | None,
    max_variants: int = 3,
) -> tuple[list[str], list[str], str]:
    """把 LLM 结果与规则结果合并：查询变体并集（规则优先保底），实体并集。

    返回 (查询变体, 实体, 置信度提示)。
    """
    queries = list(rule_queries)
    entities = list(rule_entities)
    confidence = ""

    if alignment is not None:
        for query in alignment.queries:
            if query and query not in queries:
                queries.append(query)
        for entity in alignment.entities:
            if entity and entity not in entities:
                entities.append(entity)
        confidence = alignment.confidence

    return queries[: max(1, int(max_variants))], entities, confidence


__all__ = [
    "AlignmentResult",
    "INTENTS",
    "align_with_llm",
    "build_query_variants",
    "merge_alignment",
]
