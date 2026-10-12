"""对话图的节点实现。

节点都是「输入状态 → 输出局部状态」的纯函数式写法，依赖通过闭包注入（`GraphDeps`），
因此可以脱离 LangGraph 直接单测：给定状态与假依赖，断言输出即可。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.deps import GraphDeps
from app.graph.state import GraphState
from app.llm.prompts import (
    FALLBACK_TEMPLATE,
    build_answer_messages,
    build_fallback_messages,
)
from app.llm.providers import ProviderError
from app.llm.vision import analyze_image
from app.rag.retriever import hybrid_retrieve
from app.tickets.draft import extract_from_text, merge_draft

logger = logging.getLogger(__name__)

VISION_FAILED_HINT = "（图片没能识别成功：{error}。我先按你的文字描述处理，也可以补一句文字说明。）"
ANSWER_UNAVAILABLE = (
    "抱歉，回答生成通道暂时不可用（{error}）。检索到的资料片段已列在右侧，建议转人工客服跟进。"
)

NodeFn = Callable[[GraphState], dict[str, Any]]


def build_nodes(deps: GraphDeps) -> dict[str, NodeFn]:
    """按依赖组装全部节点。"""

    # ------------------------------------------------------------------ 入口

    def prepare(state: GraphState) -> dict[str, Any]:
        conversation_id = str(state.get("conversation_id") or "").strip()
        text = str(state.get("text") or "").strip()
        image_path = state.get("image_path") or None

        if not conversation_id:
            return {"error": "缺少会话 ID", "text": text, "image_path": image_path}
        if not text and not image_path:
            return {
                "conversation_id": conversation_id,
                "error": "请先输入问题，或上传一张故障截图 / 实物照片",
            }

        history = deps.store.history_before_last_user(
            conversation_id, turns=deps.settings.history_turns
        )
        return {
            "conversation_id": conversation_id,
            "text": text,
            "image_path": image_path,
            "history": history,
            # 本轮字段必须重置，避免上一轮内容串味
            "vision_summary": "",
            "vision_ok": False,
            "vision_error": "",
            "vision_detail": {},
            "query": "",
            "retrieved": [],
            "hit": False,
            "best_score": 0.0,
            "retrieval_mode": "",
            "retrieval_reason": "",
            "answer": "",
            "provider": "",
            "answer_error": "",
            "citations": [],
            "fallback_used": False,
            "error": "",
        }

    def reject(state: GraphState) -> dict[str, Any]:
        message = str(state.get("error") or "无法处理这次请求")
        deps.emitter({"type": "error", "message": message})
        return {"answer": message, "citations": [], "provider": "", "fallback_used": False}

    # ------------------------------------------------------------ 图片理解

    def vision_understand(state: GraphState) -> dict[str, Any]:
        deps.emitter({"type": "stage", "stage": "vision", "message": "正在识别图片…"})
        result = analyze_image(
            state.get("image_path") or "",
            providers=deps.vision_candidates,
            settings=deps.settings,
            context=str(state.get("text") or ""),
        )
        deps.emitter(
            {
                "type": "vision",
                "ok": result.ok,
                "summary": result.summary,
                "error": result.error,
                "detail": result.detail,
                "provider": result.provider,
                "model": result.model,
                "local": result.local,
                "image_info": result.image_info,
            }
        )
        if not result.ok:
            return {
                "vision_ok": False,
                "vision_error": result.error,
                "vision_summary": "",
                "vision_detail": {},
                "vision_provider": result.provider,
                "vision_local": result.local,
            }
        return {
            "vision_ok": True,
            "vision_error": "",
            "vision_summary": result.summary,
            "vision_detail": result.detail,
            "vision_provider": result.provider,
            "vision_local": result.local,
        }

    def rewrite_query(state: GraphState) -> dict[str, Any]:
        """把文字与图片描述合并成检索查询（确定性改写，不额外调用模型）。"""
        text = str(state.get("text") or "").strip()
        summary = str(state.get("vision_summary") or "").strip()

        parts: list[str] = [text] if text else []
        if summary:
            keys = {"错误码或提示", "产品型号", "概述", "可见异常", "其他文字", "图片类型"}
            extra = [
                line.strip()
                for line in summary.splitlines()
                if line.split("：")[0].strip() in keys
            ]
            parts.extend(extra or [summary])
        return {"query": "\n".join(part for part in parts if part).strip()}

    # ---------------------------------------------------------------- 检索

    def retrieve(state: GraphState) -> dict[str, Any]:
        deps.emitter({"type": "stage", "stage": "retrieval", "message": "正在检索内部资料…"})
        outcome = hybrid_retrieve(
            deps.conn,
            str(state.get("query") or ""),
            embedder=deps.embedder,
            settings=deps.settings,
        )
        chunks = [chunk.as_dict() for chunk in outcome.chunks]
        deps.emitter(
            {
                "type": "retrieval",
                "hit": outcome.hit,
                "best_score": outcome.best_score,
                "mode": outcome.mode,
                "reason": outcome.reason,
                "keyword_candidates": outcome.keyword_candidates,
                "vector_candidates": outcome.vector_candidates,
                "chunks": chunks,
            }
        )
        return {
            "retrieved": chunks,
            "hit": outcome.hit,
            "best_score": outcome.best_score,
            "retrieval_mode": outcome.mode,
            "retrieval_reason": outcome.reason,
        }

    # ---------------------------------------------------------------- 回答

    def _stream_answer(messages: list[dict]) -> dict[str, Any]:
        """流式生成并推送 token 事件；流式起不来就整体失败，由调用方兜底。"""
        try:
            provider_name, iterator = deps.chat_chain.stream(
                messages, max_tokens=deps.settings.chat_max_tokens
            )
        except ProviderError as exc:
            return {"answer": "", "provider": "", "answer_error": str(exc)}

        pieces: list[str] = []
        try:
            for piece in iterator:
                pieces.append(piece)
                deps.emitter({"type": "token", "text": piece})
        except ProviderError as exc:
            if not pieces:
                return {"answer": "", "provider": provider_name, "answer_error": str(exc)}
            logger.warning("流式回答中断，返回已生成的部分：%s", exc)
            return {
                "answer": "".join(pieces).strip(),
                "provider": provider_name,
                "answer_error": str(exc),
            }

        return {"answer": "".join(pieces).strip(), "provider": provider_name, "answer_error": ""}

    def _vision_prefix(state: GraphState) -> str:
        """图片识别失败时的提示语：必须在回答正文之前推送，页面顺序才正确。"""
        error = str(state.get("vision_error") or "").strip()
        if not error:
            return ""
        hint = VISION_FAILED_HINT.format(error=error) + "\n\n"
        deps.emitter({"type": "token", "text": hint})
        return hint

    def generate_answer(state: GraphState) -> dict[str, Any]:
        prefix = _vision_prefix(state)
        chunks = state.get("retrieved") or []
        messages = build_answer_messages(
            str(state.get("text") or ""),
            chunks,
            vision_summary=str(state.get("vision_summary") or ""),
            history=state.get("history") or [],
        )
        result = _stream_answer(messages)
        answer = str(result.get("answer") or "")
        if not answer:
            answer = ANSWER_UNAVAILABLE.format(
                error=result.get("answer_error") or "没有返回内容"
            )
            deps.emitter({"type": "token", "text": answer})

        citations = [str(chunk.get("citation") or "") for chunk in chunks if chunk.get("citation")]
        return {
            "answer": prefix + answer,
            "provider": str(result.get("provider") or ""),
            "answer_error": str(result.get("answer_error") or ""),
            "citations": citations,
            "fallback_used": False,
        }

    def fallback_answer(state: GraphState) -> dict[str, Any]:
        """未命中知识库：走兜底话术。

        默认**不调用模型**，直接用模板逐字输出：兜底路径本身就是「不许编」的场景，
        交给模型润色反而有改写成模糊说法、甚至编造无关建议的风险；同时这条路
        在模型全挂时也必须可用，确定性输出天然满足。需要润色时把
        `FALLBACK_USE_MODEL` 打开即可。
        """
        prefix = _vision_prefix(state)
        question = str(state.get("text") or "").strip() or "（客户只上传了图片）"
        vision_summary = str(state.get("vision_summary") or "").strip()
        extra = f"【图片分析结果】{vision_summary}" if vision_summary else ""
        template = FALLBACK_TEMPLATE.format(question=question, extra=extra)

        provider = ""
        answer_error = ""
        if deps.settings.fallback_use_model:
            result = _stream_answer(
                build_fallback_messages(
                    question,
                    vision_summary=vision_summary,
                    history=state.get("history") or [],
                )
            )
            answer = str(result.get("answer") or "")
            provider = str(result.get("provider") or "")
            answer_error = str(result.get("answer_error") or "")
            if not answer:
                answer = template
                deps.emitter({"type": "token", "text": template})
        else:
            answer = template
            deps.emitter({"type": "token", "text": template})

        return {
            "answer": prefix + answer,
            "provider": provider,
            "answer_error": answer_error,
            "citations": [],
            "fallback_used": True,
        }

    # ------------------------------------------------------------ 工单草稿

    def update_draft(state: GraphState) -> dict[str, Any]:
        conversation_id = str(state.get("conversation_id") or "")
        delta = extract_from_text(
            str(state.get("text") or ""),
            vision_detail=state.get("vision_detail") or {},
            answer=str(state.get("answer") or ""),
            citations=state.get("citations") or [],
        )
        existing = deps.store.get_draft(conversation_id) if conversation_id else {}
        merged = merge_draft(existing, delta)
        if conversation_id:
            deps.store.set_draft(conversation_id, merged)
        deps.emitter({"type": "draft", "draft": merged})
        return {"ticket_draft": merged}

    return {
        "prepare": prepare,
        "reject": reject,
        "vision_understand": vision_understand,
        "rewrite_query": rewrite_query,
        "hybrid_retrieve": retrieve,
        "generate_answer": generate_answer,
        "fallback_answer": fallback_answer,
        "update_draft": update_draft,
    }


# ---------------------------------------------------------------------- 路由

def route_after_prepare(state: GraphState) -> str:
    if state.get("error"):
        return "reject"
    return "vision_understand" if state.get("image_path") else "rewrite_query"


def route_after_retrieve(state: GraphState) -> str:
    return "generate_answer" if state.get("hit") else "fallback_answer"


__all__ = ["build_nodes", "route_after_prepare", "route_after_retrieve"]
