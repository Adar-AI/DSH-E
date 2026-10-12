"""对话图的节点实现（v2.1 六节点）。

节点与需求规格第 13.1 节的对应关系：

| 需求节点 | 实现 |
|---|---|
| ① 意图确认与改写 | `intent_align` |
| ② 并发召回（内部 + MCP 联网） | `recall_internal` + `fuse_rerank` 内的联网判定 |
| ③ RRF 融合 | `fuse_rerank` |
| ④ 重排 | `fuse_rerank` |
| ⑤ 动态 TopK | `fuse_rerank`（`select_supporting`） |
| ⑥ LLM 生成 | `generate_answer` / `fallback_answer` / `clarify` / `reject_low` |

为什么 ②③④⑤ 合并在一个节点里：联网的触发条件依赖**内部候选的重排结果**
（"内部 top1 < 阈值才联网"），拆成两个节点就必须把候选在状态里来回搬运；
而图状态要能被 checkpointer 序列化，所以候选只能以 dict 形式传递。
合并后逻辑更清晰，也少一次状态往返。

所有节点都是「输入状态 → 输出局部状态」的纯函数式写法，依赖通过闭包注入（`GraphDeps`），
可以脱离 LangGraph 直接单测。
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
from app.rag.confidence import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    build_clarify_message,
    build_reject_message,
    collect_entity_pool,
    extract_entities,
    judge_after_recall,
    judge_before_recall,
)
from app.rag.fusion import (
    Candidate,
    SOURCE_INTERNAL,
    SOURCE_WEB,
    apply_rerank_scores,
    rrf_fuse,
    select_supporting,
    summarize_sources,
)
from app.rag.retriever import hybrid_retrieve
from app.rag.rewrite import align_with_llm, build_query_variants, merge_alignment
from app.tickets.draft import extract_from_text, merge_draft

logger = logging.getLogger(__name__)

VISION_FAILED_HINT = "（图片没能识别成功：{error}。我先按你的文字描述处理，也可以补一句文字说明。）"
ANSWER_UNAVAILABLE = (
    "抱歉，回答生成通道暂时不可用（{error}）。检索到的资料片段已列在右侧，建议转人工客服跟进。"
)

NodeFn = Callable[[GraphState], dict[str, Any]]


def label_contexts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给支撑片段打引用标号：内部 `[1]`、联网 `[网1]`（两者分开编号）。"""
    labelled: list[dict[str, Any]] = []
    internal_index = 0
    web_index = 0
    for item in candidates:
        payload = dict(item)
        if payload.get("source") == SOURCE_WEB:
            web_index += 1
            payload["label"] = f"[网{web_index}]"
        else:
            internal_index += 1
            payload["label"] = f"[{internal_index}]"
        labelled.append(payload)
    return labelled


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
            "vision_provider": "",
            "vision_local": False,
            "queries": [],
            "entities": [],
            "internal_channels": {},
            "candidates": [],
            "kept": [],
            "selection": {},
            "rerank_top1": 0.0,
            "recall_sources": {},
            "web_used": False,
            "web_detail": {},
            "confidence": "",
            "confidence_reason": "",
            "clarify_question": "",
            "reject_message": "",
            "answer": "",
            "provider": "",
            "answer_error": "",
            "citations": [],
            "web_citations": [],
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

    # ------------------------------------------------- ① 意图确认与查询改写

    def intent_align(state: GraphState) -> dict[str, Any]:
        settings = deps.settings
        question = str(state.get("text") or "").strip()
        conversation_id = str(state.get("conversation_id") or "")

        history_texts = [str(item.get("content") or "") for item in (state.get("history") or [])]
        draft = deps.store.get_draft(conversation_id) if conversation_id else {}
        vision_detail = state.get("vision_detail") or {}
        pool = collect_entity_pool(
            history_texts=history_texts,
            draft=draft,
            vision_detail=vision_detail,
        )

        # 图片里识别出的型号/错误码等同于"问题里已经说了"，直接当作已知实体
        vision_entities = [
            str(vision_detail.get("product_model") or ""),
            *(str(code) for code in (vision_detail.get("error_codes") or [])),
            *(name for name, _ in extract_entities(str(state.get("vision_summary") or ""))),
        ]
        vision_names = [name for name in dict.fromkeys(vision_entities) if name]

        judgement = judge_before_recall(
            question,
            context_entities=pool,
            extra_inline=vision_names,
            settings=settings,
        )
        confidence_hint = ""

        # 边界情形（缺主语但可能有救）才调 LLM，默认关闭
        if judgement.needs_llm_align and settings.llm_align_enabled:
            alignment = align_with_llm(
                question, context_entities=pool, chat_chain=deps.chat_chain, settings=settings
            )
            if alignment is not None:
                confidence_hint = alignment.confidence
                if alignment.confidence == LEVEL_HIGH and alignment.entities:
                    judgement.level = LEVEL_HIGH
                    judgement.reason += f"；LLM 判定可确定实体：{'、'.join(alignment.entities)}"
                    judgement.context_entities = alignment.entities
                elif alignment.confidence == LEVEL_LOW:
                    judgement.level = LEVEL_LOW
                    judgement.reason += "；LLM 判定无法确定实体"
                    judgement.reject_message = judgement.reject_message or build_reject_message(question)

        inline = [name for name, _ in extract_entities(question)]
        error_codes = [name for name, kind in extract_entities(question) if kind == "error_code"]
        for name, kind in extract_entities(str(state.get("vision_summary") or "")):
            if kind == "error_code" and name not in error_codes:
                error_codes.append(name)
        error_codes.extend(
            str(code) for code in (vision_detail.get("error_codes") or []) if str(code) not in error_codes
        )

        # 型号候选：上下文（图片/草稿/历史）+ 问题自带的 + 图片识别出的。
        # 问题自带的型号也要算进去，否则"内部没收录、问题里带型号"时联网会被自己跳过。
        model_entities = list(
            dict.fromkeys(
                [
                    *(item.name for item in pool if item.kind == "product_model"),
                    *(name for name, kind in extract_entities(question) if kind == "product_model"),
                    *(name for name in vision_names if name not in error_codes),
                ]
            )
        )
        model_entities = [name for name in model_entities if name not in error_codes]
        # 检索式里带上所有已知实体（问题里的 + 图片里的 + 上下文对齐的），
        # 否则"图片显示 E05、文字只说打印不出来"这类问题检索不到正确片段
        rewrite_entities = [
            name
            for name in dict.fromkeys([*inline, *vision_names, *judgement.context_entities, *model_entities])
            if name and name not in error_codes
        ]
        rule_queries = build_query_variants(
            judgement.rewritten_query or question,
            entities=rewrite_entities,
            error_codes=error_codes,
            max_variants=3,
        )

        queries = rule_queries
        if judgement.needs_llm_align and settings.llm_align_enabled:
            alignment = align_with_llm(
                question, context_entities=pool, chat_chain=deps.chat_chain, settings=settings
            )
            queries, llm_entities, hint = merge_alignment(
                rule_queries=rule_queries,
                rule_entities=rewrite_entities,
                alignment=alignment,
                max_variants=3,
            )
            confidence_hint = confidence_hint or hint
            model_entities = list(dict.fromkeys([*model_entities, *llm_entities]))

        deps.emitter(
            {
                "type": "intent",
                "level": judgement.level,
                "reason": judgement.reason,
                "entities": [*inline, *vision_names, *judgement.context_entities],
                "queries": queries,
                "needs_llm_align": judgement.needs_llm_align,
                "confidence_hint": confidence_hint,
            }
        )

        return {
            "queries": queries,
            "query": queries[0] if queries else question,
            "entities": [*inline, *vision_names, *judgement.context_entities],
            "model_entities": model_entities,
            "pre_level": judgement.level,
            "pre_reason": judgement.reason,
            "confidence": judgement.level,
            "confidence_reason": judgement.reason,
            "clarify_question": judgement.clarify_question,
            "reject_message": judgement.reject_message,
            "clarify_candidates": judgement.context_entities,
        }

    # ------------------------------------------------------- ② 内部多路召回

    def recall_internal(state: GraphState) -> dict[str, Any]:
        settings = deps.settings
        queries = [query for query in (state.get("queries") or []) if query] or [
            str(state.get("text") or "")
        ]
        channels: dict[str, list[dict[str, Any]]] = {}
        best = 0.0
        total = 0

        deps.emitter({"type": "stage", "stage": "recall", "message": "正在检索内部资料…"})
        for index, query in enumerate(queries, start=1):
            outcome = hybrid_retrieve(
                deps.conn,
                query,
                embedder=deps.embedder,
                settings=settings,
                top_k=settings.rerank_candidates,
                candidates=settings.rerank_candidates,
                min_score=0.0,
                apply_support_filter=False,
            )
            name = f"internal:q{index}"
            channels[name] = [
                Candidate.from_retrieved(chunk.as_dict(), channel=name).as_dict()
                for chunk in outcome.chunks
            ]
            total += len(channels[name])
            best = max(best, outcome.best_score)

        return {
            "internal_channels": channels,
            "internal_best": best,
            "recall_sources": {SOURCE_INTERNAL: total},
        }

    # ------------------------------------------- ③④⑤ 融合 + 重排 + 动态 TopK

    def fuse_rerank(state: GraphState) -> dict[str, Any]:
        settings = deps.settings
        question = str(state.get("text") or "").strip()
        # 重排用的查询 = 原始问题 + 已知实体。
        # 为什么必须带上实体：客户可能只说"打印不出来了"，型号/错误码是图片或上下文给的；
        # 拿原始问题去精排，会把最相关的片段也打低分（实测过这个坑）。
        entity_hint = [str(item) for item in (state.get("entities") or []) if item]
        rerank_query = " ".join([question, *entity_hint]).strip() or question

        channels = {
            name: [Candidate.from_dict(item) for item in items]
            for name, items in (state.get("internal_channels") or {}).items()
        }

        fused = rrf_fuse(channels)
        reranker = deps.reranker
        rerank_mode = "rerank" if reranker is not None else "none"

        if fused and reranker is not None:
            try:
                scores = reranker.score(rerank_query, [item.text for item in fused])
                fused = apply_rerank_scores(fused, scores)
            except Exception as exc:  # noqa: BLE001 - 重排失败退化为融合分
                logger.warning("重排失败，改用融合分：%s", exc)
                rerank_mode = "fallback"

        internal_top1 = fused[0].score if fused else 0.0

        # 联网判定：内部证据不足才联网（内部优先）
        web_detail: dict[str, Any] = {}
        web_used = False
        model_entities = [str(item) for item in (state.get("model_entities") or [])]
        coverage = 1.0
        if deps.web_search is not None:
            from app.rag.web_search import build_query, question_coverage, should_search

            coverage = question_coverage(
                question,
                [item.text for item in fused],
                entities=[*model_entities, *(state.get("entities") or [])],
            )
            if should_search(internal_top1, coverage=coverage, settings=settings):
                deps.emitter({"type": "stage", "stage": "web", "message": "内部资料不足，正在联网检索…"})
                outcome = deps.web_search(
                    build_query(question, model_entities),
                    entities=model_entities,
                    settings=settings,
                )
                web_detail = outcome.as_dict()
                candidates = [result.as_candidate() for result in outcome.results]
                if candidates:
                    web_used = True
                    web_channel = rrf_fuse({"web": candidates})
                    if reranker is not None:
                        try:
                            web_scores = reranker.score(
                                rerank_query, [item.text for item in web_channel]
                            )
                            web_channel = apply_rerank_scores(web_channel, web_scores)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("联网候选重排失败，保留融合分：%s", exc)
                    merged = {item.key: item for item in fused}
                    for item in web_channel:
                        merged.setdefault(item.key, item)
                    fused = sorted(merged.values(), key=lambda entry: (-entry.score, entry.key))

        selection = select_supporting(
            fused,
            min_score=settings.rerank_min_score,
            relative_floor=settings.rerank_relative_floor,
            cliff_ratio=settings.rerank_cliff_ratio,
            top_k_max=settings.rerank_top_k_max,
        )

        sources = summarize_sources(fused)
        deps.emitter(
            {
                "type": "rerank",
                "mode": rerank_mode,
                "candidates": len(fused),
                "top1": round(selection.top1, 4),
                "kept": len(selection.kept),
                "cutoff": selection.cutoff,
                "reason": selection.reason,
                "sources": sources,
                "coverage": round(coverage, 4),
                "web": web_detail,
            }
        )

        return {
            "candidates": [item.as_dict() for item in fused],
            "kept": [item.as_dict() for item in selection.kept],
            "selection": selection.as_dict(),
            "rerank_top1": selection.top1,
            "rerank_mode": rerank_mode,
            "rerank_query": rerank_query,
            "recall_sources": sources,
            "web_used": web_used,
            "web_detail": web_detail,
        }

    # ------------------------------------------------------- ⑤′ 置信终判

    def post_judge(state: GraphState) -> dict[str, Any]:
        settings = deps.settings
        question = str(state.get("text") or "").strip()
        kept = state.get("kept") or []
        top1 = float(state.get("rerank_top1") or 0.0)

        pre = judge_before_recall(
            question, context_entities=(), settings=settings
        )
        pre.context_entities = list(state.get("clarify_candidates") or [])

        judgement = judge_after_recall(
            question,
            top1=top1,
            kept=len(kept),
            pre=pre,
            extra_options=list(state.get("model_entities") or []),
            settings=settings,
        )
        deps.emitter(
            {
                "type": "confidence",
                "level": judgement.level,
                "top1": round(judgement.top1, 4),
                "kept": judgement.kept,
                "reason": judgement.reason,
            }
        )
        return {
            "confidence": judgement.level,
            "confidence_reason": judgement.reason,
            "clarify_question": judgement.clarify_question or state.get("clarify_question") or "",
            "reject_message": judgement.reject_message or state.get("reject_message") or "",
        }

    # ------------------------------------------------------------ 澄清 / 拒答

    def clarify(state: GraphState) -> dict[str, Any]:
        """中置信：只反问，不给答案（话术是确定性模板）。"""
        question = str(state.get("text") or "").strip()
        candidates = list(state.get("clarify_candidates") or [])
        options = list(state.get("model_entities") or [])
        message = str(state.get("clarify_question") or "").strip() or build_clarify_message(
            question, candidates, extra_options=options
        )
        deps.emitter(
            {
                "type": "clarify",
                "question": message,
                "candidates": candidates,
                "reason": state.get("confidence_reason") or state.get("pre_reason") or "",
            }
        )
        deps.emitter({"type": "token", "text": message})
        return {"answer": message, "citations": [], "web_citations": [], "fallback_used": False}

    def reject_low(state: GraphState) -> dict[str, Any]:
        """低置信：拒答（说明缺什么、怎么补），与"资料未收录"的兜底话术区分开。"""
        question = str(state.get("text") or "").strip()
        message = str(state.get("reject_message") or "").strip() or build_reject_message(question)
        deps.emitter(
            {
                "type": "reject",
                "message": message,
                "reason": state.get("confidence_reason") or state.get("pre_reason") or "",
            }
        )
        deps.emitter({"type": "token", "text": message})
        return {"answer": message, "citations": [], "web_citations": [], "fallback_used": False}

    # ---------------------------------------------------------------- 回答

    def _stream_answer(messages: list[dict]) -> dict[str, Any]:
        """流式生成并推送 token 事件。"""
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
        error = str(state.get("vision_error") or "").strip()
        if not error:
            return ""
        hint = VISION_FAILED_HINT.format(error=error) + "\n\n"
        deps.emitter({"type": "token", "text": hint})
        return hint

    def generate_answer(state: GraphState) -> dict[str, Any]:
        prefix = _vision_prefix(state)
        contexts = label_contexts(list(state.get("kept") or []))
        messages = build_answer_messages(
            str(state.get("text") or ""),
            contexts,
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

        if any(item.get("source") == SOURCE_WEB for item in contexts):
            note = "\n\n（以上部分依据来自互联网公开资料，仅供参考；如需准确结论请以本公司资料或人工客服为准。）"
            deps.emitter({"type": "token", "text": note})
            answer += note

        # 内部引用进 messages/工单；联网来源只进 trace（用户确认：联网不入工单）
        citations = [
            str(item.get("citation") or "")
            for item in contexts
            if item.get("source") != SOURCE_WEB and item.get("citation")
        ]
        web_citations = [
            f"[网{i}] {item.get('citation')}（{item.get('url')}）"
            for i, item in enumerate(
                [entry for entry in contexts if entry.get("source") == SOURCE_WEB], start=1
            )
        ]

        return {
            "answer": prefix + answer,
            "provider": str(result.get("provider") or ""),
            "answer_error": str(result.get("answer_error") or ""),
            "citations": citations,
            "web_citations": web_citations,
            "fallback_used": False,
        }

    def fallback_answer(state: GraphState) -> dict[str, Any]:
        """未命中知识库：走兜底话术（默认不调用模型，逐字可控）。"""
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
            "web_citations": [],
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
        "intent_align": intent_align,
        "recall_internal": recall_internal,
        "fuse_rerank": fuse_rerank,
        "post_judge": post_judge,
        "clarify": clarify,
        "reject_low": reject_low,
        "generate_answer": generate_answer,
        "fallback_answer": fallback_answer,
        "update_draft": update_draft,
    }


# ---------------------------------------------------------------------- 路由

def route_after_prepare(state: GraphState) -> str:
    if state.get("error"):
        return "reject"
    return "vision_understand" if state.get("image_path") else "intent_align"


def route_after_intent(state: GraphState) -> str:
    """第一段判定：问得不清楚就别检索了。"""
    level = str(state.get("pre_level") or LEVEL_HIGH)
    if level == LEVEL_LOW:
        return "reject_low"
    if level == LEVEL_MEDIUM:
        return "clarify"
    return "recall_internal"


def route_after_judge(state: GraphState) -> str:
    """第二段判定：资料够不够答。"""
    level = str(state.get("confidence") or LEVEL_HIGH)
    if level == LEVEL_LOW:
        return "fallback_answer"
    if level == LEVEL_MEDIUM:
        return "clarify"
    return "generate_answer"


__all__ = [
    "build_nodes",
    "label_contexts",
    "route_after_intent",
    "route_after_judge",
    "route_after_prepare",
]
