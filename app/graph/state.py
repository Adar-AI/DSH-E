"""LangGraph 对话状态定义。

约定：
* `prepare` 节点每轮都会把「本轮字段」重置一遍（图片描述、检索结果、回答、引用……），
  只有 `ticket_draft` 与会话无关的字段跨轮保留——否则上一轮的图片描述会串到这一轮；
* 历史对话不放在状态里做累积，而是每轮从 messages 表读取（数据库才是唯一事实来源）。
"""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # ---- 输入 ----
    conversation_id: str
    text: str
    image_path: str | None

    # ---- 多轮上下文 ----
    history: list[dict[str, str]]

    # ---- 图片理解 ----
    vision_summary: str
    vision_ok: bool
    vision_error: str
    vision_detail: dict[str, Any]
    vision_provider: str
    vision_local: bool

    # ---- 意图确认与改写（第一段判定）----
    query: str
    queries: list[str]
    entities: list[str]
    model_entities: list[str]
    pre_level: str
    pre_reason: str
    clarify_candidates: list[str]
    clarify_question: str
    reject_message: str

    # ---- 召回与融合（第二段判定的输入）----
    internal_channels: dict[str, list[dict[str, Any]]]
    internal_best: float
    candidates: list[dict[str, Any]]
    kept: list[dict[str, Any]]
    selection: dict[str, Any]
    rerank_top1: float
    rerank_mode: str
    rerank_query: str
    recall_sources: dict[str, int]
    web_used: bool
    web_detail: dict[str, Any]

    # ---- 置信度（两段判定的结论）----
    confidence: str
    confidence_reason: str

    # ---- 回答 ----
    answer: str
    provider: str
    answer_error: str
    citations: list[str]
    web_citations: list[str]
    fallback_used: bool

    # ---- 工单草稿 ----
    ticket_draft: dict[str, Any]

    # ---- 异常 ----
    error: str


__all__ = ["GraphState"]
