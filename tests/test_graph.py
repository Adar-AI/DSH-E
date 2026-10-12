"""对话图测试（v2.1 六节点）：三档置信分支、融合重排、联网兜底、多轮记忆。

这些测试直接跑编译后的图（不经过 HTTP），依赖全部注入假实现，因此完全离线、毫秒级。
"""

from __future__ import annotations

import pytest

from app.graph.agent import build_graph
from app.graph.nodes import (
    label_contexts,
    route_after_intent,
    route_after_judge,
    route_after_prepare,
)
from app.llm.prompts import FALLBACK_TEMPLATE
from app.rag.confidence import LEVEL_HIGH, LEVEL_LOW, LEVEL_MEDIUM
from app.rag.fusion import SOURCE_INTERNAL, SOURCE_WEB
from app.rag.web_search import WebSearchOutcome, WebResult
from app.store import ConversationStore
from fakes import (
    GOOD_VISION_JSON,
    FakeChatChain,
    HashEmbedder,
    StubVisionProvider,
    load_corpus,
    write_png,
)

E05_TEXT = (
    "屏幕出现 E05 表示硒鼓未被识别，请重新安装硒鼓并确认芯片触点清洁；"
    "若仍报错说明硒鼓芯片损坏，需要更换新的硒鼓。"
)
WARRANTY_TEXT = "本机型整机保修期为两年，自购买之日起计算，人为损坏不在保修范围内。"
UNRELATED_TEXT = "无线网络连接失败时，请检查路由器是否开启了 2.4G 频段。"


@pytest.fixture()
def corpus(conn):
    return load_corpus(
        conn,
        [
            ("打印机产品手册.md", E05_TEXT, "报错处理", 12),
            ("售后服务规则.md", WARRANTY_TEXT, "保修政策", None),
            ("网络设置指南.md", UNRELATED_TEXT, "网络", 3),
        ],
        HashEmbedder(dim=64),
    )


def _conversation(conn, *, history: list[str] = ()) -> str:
    store = ConversationStore(conn)
    conv_id = store.create(title="测试会话")["id"]
    for text in history:
        store.add_message(conv_id, "user", text)
    return conv_id


def _ask(graph, conversation_id: str, text: str, **extra):
    return graph.invoke({"conversation_id": conversation_id, "text": text, **extra})


# ------------------------------------------------------------------ 路由函数

def test_route_after_prepare():
    assert route_after_prepare({"error": "缺输入"}) == "reject"
    assert route_after_prepare({"image_path": "/tmp/a.png"}) == "vision_understand"
    assert route_after_prepare({"text": "报错"}) == "intent_align"


@pytest.mark.parametrize(
    ("level", "expected"),
    [(LEVEL_LOW, "reject_low"), (LEVEL_MEDIUM, "clarify"), (LEVEL_HIGH, "recall_internal")],
)
def test_route_after_intent(level, expected):
    assert route_after_intent({"pre_level": level}) == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [(LEVEL_LOW, "fallback_answer"), (LEVEL_MEDIUM, "clarify"), (LEVEL_HIGH, "generate_answer")],
)
def test_route_after_judge(level, expected):
    assert route_after_judge({"confidence": level}) == expected


def test_label_contexts_numbers_internal_and_web_separately():
    labelled = label_contexts(
        [
            {"source": SOURCE_INTERNAL, "citation": "手册.md"},
            {"source": SOURCE_WEB, "citation": "某网页"},
            {"source": SOURCE_INTERNAL, "citation": "规则.md"},
            {"source": SOURCE_WEB, "citation": "另一网页"},
        ]
    )

    assert [item["label"] for item in labelled] == ["[1]", "[网1]", "[2]", "[网2]"]


# ------------------------------------------------------------------ 高置信

def test_high_confidence_path_answers_with_citations(conn, corpus, make_deps, events):
    conv_id = _conversation(conn)
    deps = make_deps(embedder=HashEmbedder(dim=64))
    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert state["pre_level"] == LEVEL_HIGH
    assert state["confidence"] == LEVEL_HIGH
    assert state["answer"]
    assert state["citations"] == ["打印机产品手册.md · 第 12 页 · 报错处理"]
    assert state["web_used"] is False
    assert state["fallback_used"] is False
    assert state["rerank_top1"] > 0

    events_by_type = {event["type"] for event in events}
    assert {"intent", "rerank", "confidence", "token", "draft"} <= events_by_type

    rerank_event = next(event for event in events if event["type"] == "rerank")
    assert rerank_event["mode"] == "rerank"
    assert rerank_event["kept"] == 1, "无关候选应被动态 TopK 丢掉"
    assert rerank_event["cutoff"] in {"floor", "relative", "cliff", "cap"}


def test_rerank_failure_degrades_to_fusion_scores(conn, corpus, make_deps):
    """重排器报错时退化为融合分，流程不中断。"""
    from fakes import StubReranker

    conv_id = _conversation(conn)
    deps = make_deps(
        embedder=HashEmbedder(dim=64), reranker=StubReranker(error="模型加载失败")
    )

    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert state["rerank_mode"] == "fallback"
    assert state["answer"]


def test_dynamic_topk_drops_irrelevant_candidates(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert len(state["candidates"]) >= 2, "应当召回多条候选"
    assert len(state["kept"]) < len(state["candidates"]), "无关候选不应进入支撑集"
    assert state["selection"]["cutoff"] in {"floor", "relative", "cliff", "cap"}


def test_answer_prompt_contains_numbered_contexts(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="请先检查硒鼓芯片触点 [1]。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    prompt = chain.calls[0][-1]["content"]
    assert "[1] 来源：打印机产品手册.md · 第 12 页 · 报错处理" in prompt
    assert "只能依据" in chain.calls[0][0]["content"]


# ------------------------------------------------------- 中置信：多实体反问

def test_multiple_entities_triggers_clarify_without_answer(conn, corpus, make_deps, events):
    """验收场景 2：上下文里两个型号都出现过 → 反问，且不生成答案。"""
    conv_id = _conversation(
        conn, history=["我的打印机是 L3250，报错 E05", "另外一台 M1136 也老是卡纸"]
    )
    store = ConversationStore(conn)
    # 与真实接口一致：当前问题先入库，图里取"最后一条用户消息之前"的历史
    store.add_message(conv_id, "user", "保修期是多少？")
    chain = FakeChatChain(text="这段回答不应该出现")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "保修期是多少？")

    assert state["pre_level"] == LEVEL_MEDIUM
    assert state["answer"]
    assert "L3250" in state["answer"] and "M1136" in state["answer"]
    assert "这段回答不应该出现" not in state["answer"]
    assert state["citations"] == []
    assert chain.calls == [], "反问阶段不应调用模型生成答案"

    clarify_event = next(event for event in events if event["type"] == "clarify")
    assert set(clarify_event["candidates"]) == {"L3250", "M1136"}


# ------------------------------------------------------- 低置信：缺主语拒答

def test_missing_entity_triggers_reject_without_answer(conn, corpus, make_deps, events):
    """验收场景 3：全新会话、问题缺主语 → 拒答并要求补充型号。"""
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="这段回答不应该出现")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "保修期是多少？")

    assert state["pre_level"] == LEVEL_LOW
    assert "缺少必要信息" in state["answer"]
    assert "产品型号" in state["answer"]
    assert "未收录" not in state["answer"], "拒答话术要与兜底话术区分开"
    assert chain.calls == []
    assert any(event["type"] == "reject" for event in events)


# ------------------------------------------------------- 低置信：检索无证据

def test_no_evidence_falls_back(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "预订国际航班机票价格")

    assert state["pre_level"] == LEVEL_HIGH, "这个问题不依赖型号，第一段放行"
    assert state["confidence"] == LEVEL_LOW
    assert state["fallback_used"] is True
    assert "内部资料中未收录" in state["answer"]
    assert state["citations"] == []


def test_fallback_still_works_when_all_models_down(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(chat_chain=FakeChatChain(error="所有通道都不可用"))

    state = _ask(build_graph(deps), conv_id, "预订国际航班机票价格")

    assert state["fallback_used"] is True
    assert "转人工" in state["answer"]


def test_hit_but_model_down_returns_readable_message(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(
        chat_chain=FakeChatChain(error="Ollama 未启动"), embedder=HashEmbedder(dim=64)
    )

    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert state["confidence"] == LEVEL_HIGH
    assert "回答生成通道暂时不可用" in state["answer"]
    assert state["citations"], "即使模型挂了，也应把检索到的依据返回给页面"


def test_stream_error_keeps_partial_answer(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="请检查触点", stream_error="流被中断")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert state["answer"]
    assert chain.text.startswith(state["answer"])
    assert "流被中断" in state["answer_error"]


# ------------------------------------------------------------------ 输入校验

def test_empty_input_is_rejected(conn, make_deps):
    conv_id = _conversation(conn)
    state = _ask(build_graph(make_deps()), conv_id, "   ")

    assert "请先输入问题" in state["answer"]
    assert state["error"]


def test_missing_conversation_id_is_rejected(conn, make_deps):
    state = build_graph(make_deps()).invoke({"conversation_id": "", "text": "报错"})
    assert state["error"] == "缺少会话 ID"


# ------------------------------------------------------------------ 图片链路

def test_vision_path_merges_image_into_query(conn, corpus, make_deps, events, work_dir):
    conv_id = _conversation(conn)
    image = write_png(work_dir, "e05.png")
    text = "打印不出来了，屏幕上有个提示"

    vision = StubVisionProvider()
    deps = make_deps(embedder=HashEmbedder(dim=64), vision_provider=vision)
    state = _ask(build_graph(deps), conv_id, text, image_path=str(image))

    assert state["vision_ok"] is True
    assert "E05" in " ".join(state["queries"])
    assert state["confidence"] == LEVEL_HIGH
    assert state["citations"][0].startswith("打印机产品手册.md")

    vision_event = next(event for event in events if event["type"] == "vision")
    assert vision_event["ok"] is True
    assert len(vision.calls) == 1


def test_vision_failure_degrades_to_text_only(conn, corpus, make_deps, work_dir):
    conv_id = _conversation(conn)
    image = write_png(work_dir, "broken.png")
    deps = make_deps(
        embedder=HashEmbedder(dim=64),
        vision_provider=StubVisionProvider(error="网关 429 限流"),
    )

    state = _ask(
        build_graph(deps), conv_id, "机型L3250屏幕出现E05", image_path=str(image)
    )

    assert state["vision_ok"] is False
    assert "图片没能识别成功" in state["answer"]
    assert state["confidence"] == LEVEL_HIGH, "图片识别失败不应阻断文字链路"


def test_good_vision_json_constant_matches_provider():
    import json

    payload = json.loads(GOOD_VISION_JSON)
    assert payload["error_codes"] == ["E05"]
    assert payload["product_model"] == "L3250"


# ------------------------------------------------------------------ 多轮记忆

def test_multi_turn_keeps_draft_and_passes_history(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    chain = FakeChatChain(text="第一轮回答：请检查触点。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))
    graph = build_graph(deps)

    first_text = "机型L3250屏幕出现E05，硒鼓未识别"
    store.add_message(conv_id, "user", first_text)
    first = _ask(graph, conv_id, first_text)
    assert first["confidence"] == LEVEL_HIGH
    store.add_message(conv_id, "assistant", first["answer"])
    assert store.get_draft(conv_id)["product_model"] == "L3250"

    chain.text = "第二轮回答：保修两年，人为损坏不在保修范围内。"
    second_text = "保修期是两年吗？人为损坏算不算"
    store.add_message(conv_id, "user", second_text)
    second = _ask(graph, conv_id, second_text)

    assert second["confidence"] == LEVEL_HIGH, "第二轮应当命中售后规则里的保修政策"
    prompt = "\n".join(str(item["content"]) for item in chain.calls[1])
    assert first_text in prompt, "第二轮提示词里应当带上第一轮的问答"
    assert store.get_draft(conv_id)["product_model"] == "L3250", "草稿应跨轮累积"


def test_missing_subject_aligns_to_history_entity(conn, corpus, make_deps):
    """验收场景 1：上下文已有 L3250 时问"保修期是多少？" → 对齐后正常作答。"""
    conv_id = _conversation(conn, history=["我的打印机是 L3250，屏幕显示 E05"])
    store = ConversationStore(conn)
    store.add_message(conv_id, "user", "保修期是多少？")
    chain = FakeChatChain(text="整机保修两年 [1]。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = _ask(build_graph(deps), conv_id, "保修期是多少？")

    assert state["pre_level"] == LEVEL_HIGH
    assert any("L3250" in query for query in state["queries"]), "改写后的查询应补上型号"
    assert state["confidence"] == LEVEL_HIGH
    assert state["answer"] == "整机保修两年 [1]。"
    assert state["citations"] == ["售后服务规则.md · 保修政策"]


def test_vision_context_does_not_leak_into_next_turn(conn, corpus, make_deps, work_dir):
    pytest.importorskip("langgraph.checkpoint.memory")
    from langgraph.checkpoint.memory import InMemorySaver

    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    image = write_png(work_dir, "e05.png")
    deps = make_deps(embedder=HashEmbedder(dim=64), vision_provider=StubVisionProvider())
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": conv_id}}

    store.add_message(conv_id, "user", "打印不出来", image_path=str(image))
    first = graph.invoke(
        {"conversation_id": conv_id, "text": "打印不出来", "image_path": str(image)},
        config=config,
    )
    assert first["vision_ok"] is True

    store.add_message(conv_id, "assistant", first["answer"])
    store.add_message(conv_id, "user", "保修期是多久")
    second = graph.invoke(
        {"conversation_id": conv_id, "text": "保修期是多久", "image_path": None},
        config=config,
    )

    assert second["vision_summary"] == "", "上一轮的图片描述不应残留"
    assert second["vision_ok"] is False


def test_state_persists_draft_with_checkpointer(conn, corpus, make_deps):
    pytest.importorskip("langgraph.checkpoint.memory")
    from langgraph.checkpoint.memory import InMemorySaver

    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    deps = make_deps(embedder=HashEmbedder(dim=64))
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": conv_id}}

    store.add_message(conv_id, "user", "型号 L3250 报错E05")
    graph.invoke({"conversation_id": conv_id, "text": "型号 L3250 报错E05"}, config=config)

    state = graph.invoke({"conversation_id": conv_id, "text": "谢谢"}, config=config)
    assert state["ticket_draft"]["product_model"] == "L3250"


# ------------------------------------------------------------------ 联网兜底

def web_page(title: str, snippet: str, url: str = "https://example.com/a") -> WebResult:
    return WebResult(title=title, url=url, hostname="example.com", snippet=snippet, rank=1)


def test_web_used_only_when_internal_evidence_is_weak(conn, corpus, make_deps, events):
    """内部不足 → 联网补充；联网来源单独标 [网N]，且不进工单引用。"""
    conv_id = _conversation(conn)
    calls: list[str] = []

    def fake_web(query, *, entities=(), settings=None):
        calls.append(query)
        return WebSearchOutcome(
            ok=True,
            query=query,
            tool="search_pro",
            results=(web_page("L3250 耗材回收政策", "L3250 硒鼓可交由回收点处理，请勿随意丢弃。"),),
            elapsed_ms=120,
        )

    deps = make_deps(embedder=HashEmbedder(dim=64), web_search=fake_web)
    state = _ask(build_graph(deps), conv_id, "L3250 的硒鼓能回收吗")

    assert calls, "内部证据不足时应当触发联网"
    assert state["web_used"] is True
    assert state["citations"] == [], "联网来源不进工单引用"
    assert state["web_citations"] and "example.com" in state["web_citations"][0]
    assert "[网1]" in state["answer"] or "互联网" in state["answer"]
    assert state["recall_sources"].get(SOURCE_WEB, 0) >= 1

    rerank_event = next(event for event in events if event["type"] == "rerank")
    assert rerank_event["web"]["ok"] is True


def test_web_triggered_by_low_coverage_even_when_rerank_is_high(conn, corpus, make_deps, events):
    """重排分被型号/主题词抬高、但问题主题资料里没有 → 仍要联网（覆盖率信号）。

    这条守卫来自真实实测：bge-reranker-large 对"型号匹配但主题没收录"的问题也给 0.74~0.84，
    只看重排分则联网永远不会触发。
    """
    from fakes import StubReranker

    conv_id = _conversation(conn)
    calls: list[str] = []

    def fake_web(query, *, entities=(), settings=None):
        calls.append(query)
        return WebSearchOutcome(
            ok=True,
            query=query,
            results=(web_page("L3250 以旧换新活动说明", "L3250 可参加以旧换新活动。"),),
        )

    deps = make_deps(
        embedder=HashEmbedder(dim=64),
        reranker=StubReranker(rules=[("硒鼓", 0.95)], default=0.02),
        web_search=fake_web,
    )
    state = _ask(build_graph(deps), conv_id, "L3250 硒鼓以旧换新活动")

    rerank = next(event for event in events if event["type"] == "rerank")
    assert rerank["top1"] >= 0.35, "重排分本身很高（型号/主题词匹配）"
    assert rerank["coverage"] < 0.34, "但问题主题在资料里没有覆盖"
    assert calls, "应当由覆盖率信号触发联网"
    assert state["web_used"] is True


def test_inline_model_entity_is_used_for_web_verification(conn, corpus, make_deps, events):
    """问题自带的型号必须进入联网校验实体，否则"内部没收录 + 问题带型号"时会被自己跳过。"""
    from fakes import StubReranker

    conv_id = _conversation(conn)
    seen: list[list[str]] = []

    def fake_web(query, *, entities=(), settings=None):
        seen.append(list(entities))
        return WebSearchOutcome(ok=True, query=query, results=(web_page("t", "s"),))

    deps = make_deps(
        embedder=HashEmbedder(dim=64),
        reranker=StubReranker(rules=[("硒鼓", 0.95)], default=0.02),
        web_search=fake_web,
    )
    _ask(build_graph(deps), conv_id, "L3250 硒鼓以旧换新活动")

    assert seen, "应当触发联网"
    assert "L3250" in seen[0], f"联网校验实体里缺少问题自带的型号：{seen[0]}"


def test_web_not_called_when_internal_is_confident(conn, corpus, make_deps):
    """验收场景 5：内部已高置信时不得调用 MCP。"""
    conv_id = _conversation(conn)
    calls: list[str] = []

    def fake_web(query, *, entities=(), settings=None):
        calls.append(query)
        return WebSearchOutcome(ok=True, query=query, results=(web_page("t", "s"),))

    deps = make_deps(embedder=HashEmbedder(dim=64), web_search=fake_web)
    state = _ask(build_graph(deps), conv_id, "机型L3250屏幕出现E05，硒鼓未识别")

    assert calls == [], "内部命中就不该联网"
    assert state["web_used"] is False
    assert state["confidence"] == LEVEL_HIGH


def test_web_failure_does_not_break_answer(conn, corpus, make_deps):
    conv_id = _conversation(conn)

    def failing_web(query, *, entities=(), settings=None):
        return WebSearchOutcome(ok=False, query=query, error="连接超时")

    deps = make_deps(embedder=HashEmbedder(dim=64), web_search=failing_web)
    state = _ask(build_graph(deps), conv_id, "L3250 的硒鼓能回收吗")

    assert state["web_used"] is False
    assert state["web_detail"]["error"] == "连接超时"
    assert state["answer"], "联网失败也必须给出回答"


def test_fallback_template_mentions_manual_handoff():
    assert "转人工" in FALLBACK_TEMPLATE
