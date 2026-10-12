"""对话图测试：命中/兜底分支、图片链路、多轮记忆、降级与事件推送。

这些测试直接跑编译后的图（不经过 HTTP），依赖全部注入假实现，因此完全离线。
"""

from __future__ import annotations

import pytest

from app.graph.agent import build_graph
from app.graph.nodes import route_after_prepare, route_after_retrieve
from app.llm.prompts import FALLBACK_TEMPLATE
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


def _conversation(conn) -> str:
    return ConversationStore(conn).create(title="测试会话")["id"]


# ------------------------------------------------------------------ 路由函数

def test_route_after_prepare():
    assert route_after_prepare({"error": "缺输入"}) == "reject"
    assert route_after_prepare({"image_path": "/tmp/a.png"}) == "vision_understand"
    assert route_after_prepare({"text": "报错"}) == "rewrite_query"


def test_route_after_retrieve():
    assert route_after_retrieve({"hit": True}) == "generate_answer"
    assert route_after_retrieve({"hit": False}) == "fallback_answer"


# ------------------------------------------------------------------ 命中分支

def test_hit_path_answers_with_citations_and_updates_draft(conn, corpus, make_deps, events):
    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    store.add_message(conv_id, "user", "我的打印机报错E05，硒鼓怎么更换")

    deps = make_deps(embedder=HashEmbedder(dim=64))
    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "我的打印机报错E05，硒鼓怎么更换"}
    )

    assert state["hit"] is True
    assert state["retrieval_mode"] == "hybrid"
    assert state["fallback_used"] is False
    assert state["answer"]
    assert state["citations"] == ["打印机产品手册.md · 第 12 页 · 报错处理"]

    draft = store.get_draft(conv_id)
    assert draft["error_info"] == "E05"
    assert draft["suggestion"]
    assert draft["citations"] == state["citations"]

    event_types = [event["type"] for event in events]
    assert "retrieval" in event_types
    assert "token" in event_types
    assert "draft" in event_types
    retrieval_event = next(event for event in events if event["type"] == "retrieval")
    assert retrieval_event["hit"] is True
    assert retrieval_event["chunks"][0]["citation"].startswith("打印机产品手册.md")
    assert retrieval_event["keyword_candidates"] >= 1
    assert retrieval_event["vector_candidates"] >= 1


def test_hit_path_prompt_contains_numbered_contexts(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="请先检查硒鼓芯片触点 [1]。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "报错E05怎么处理"}
    )

    prompt = chain.calls[0][-1]["content"]
    assert "[1] 来源：打印机产品手册.md · 第 12 页 · 报错处理" in prompt
    assert "只能依据" in chain.calls[0][0]["content"]


# ------------------------------------------------------------------ 兜底分支

def test_fallback_branch_for_unrelated_question(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="抱歉，内部资料未收录该问题。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "预订国际航班机票价格"}
    )

    assert state["hit"] is False
    assert state["fallback_used"] is True
    assert state["citations"] == []
    assert "内部资料中未收录" in state["answer"]
    assert chain.calls == [], "兜底话术默认不经过模型，逐字可控"


def test_fallback_can_use_model_when_enabled(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="抱歉，企业内部资料中未收录该问题。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))
    # 打开「模型润色兜底」开关
    deps.settings = deps.settings.model_copy(update={"fallback_use_model": True})

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "预订国际航班机票价格"}
    )

    assert state["fallback_used"] is True
    assert state["answer"] == chain.text
    assert len(chain.calls) == 1
    assert "原样保留" in chain.calls[0][-1]["content"]


def test_fallback_still_works_when_all_models_down(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(chat_chain=FakeChatChain(error="所有通道都不可用"))

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "预订国际航班机票价格"}
    )

    assert state["fallback_used"] is True
    assert "内部资料中未收录" in state["answer"]
    assert "转人工" in state["answer"]


def test_hit_but_model_down_returns_readable_message(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps(
        chat_chain=FakeChatChain(error="Ollama 未启动"), embedder=HashEmbedder(dim=64)
    )

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "报错E05怎么处理"}
    )

    assert state["hit"] is True
    assert "回答生成通道暂时不可用" in state["answer"]
    assert "Ollama 未启动" in state["answer"]
    assert state["citations"], "即使模型挂了，也应把检索到的依据返回给页面"


def test_stream_error_keeps_partial_answer(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    chain = FakeChatChain(text="请检查触点", stream_error="流被中断")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "报错E05怎么处理"}
    )

    assert state["answer"], "流中断时应保留已生成的部分内容"
    assert chain.text.startswith(state["answer"])
    assert state["answer"] != chain.text
    assert "流被中断" in state["answer_error"]


# ------------------------------------------------------------------ 输入校验

def test_empty_input_is_rejected(conn, make_deps):
    conv_id = _conversation(conn)
    deps = make_deps()

    state = build_graph(deps).invoke({"conversation_id": conv_id, "text": "   "})

    assert "请先输入问题" in state["answer"]
    assert state["error"]


def test_missing_conversation_id_is_rejected(conn, make_deps):
    deps = make_deps()

    state = build_graph(deps).invoke({"conversation_id": "", "text": "报错"})

    assert state["error"] == "缺少会话 ID"


# ------------------------------------------------------------------ 图片链路

def test_vision_path_merges_image_into_query(conn, corpus, make_deps, events, work_dir):
    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    image = write_png(work_dir, "e05.png")
    # 文字里没有报错码，只能靠图片识别补上
    text = "打印不出来了，屏幕上有个提示"
    store.add_message(conv_id, "user", text, image_path=str(image))

    vision = StubVisionProvider()
    deps = make_deps(embedder=HashEmbedder(dim=64), vision_provider=vision)

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": text, "image_path": str(image)}
    )

    assert state["vision_ok"] is True
    assert "E05" in state["query"]
    assert state["hit"] is True
    assert state["citations"][0].startswith("打印机产品手册.md")
    assert state["vision_detail"]["error_codes"] == ["E05"]

    vision_event = next(event for event in events if event["type"] == "vision")
    assert vision_event["ok"] is True
    assert "E05" in vision_event["summary"]
    assert len(vision.calls) == 1
    assert vision.calls[0][1]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_vision_failure_degrades_to_text_only(conn, corpus, make_deps, work_dir):
    conv_id = _conversation(conn)
    image = write_png(work_dir, "broken.png")
    deps = make_deps(
        embedder=HashEmbedder(dim=64),
        vision_provider=StubVisionProvider(error="网关 429 限流"),
    )

    state = build_graph(deps).invoke(
        {
            "conversation_id": conv_id,
            "text": "报错E05怎么处理",
            "image_path": str(image),
        }
    )

    assert state["vision_ok"] is False
    assert "网关 429 限流" in state["vision_error"]
    assert "图片没能识别成功" in state["answer"]
    assert state["hit"] is True, "图片识别失败不应阻断文字链路"


def test_vision_available_flag_false(conn, corpus, make_deps, work_dir):
    conv_id = _conversation(conn)
    image = write_png(work_dir, "no_key.png")
    deps = make_deps(
        embedder=HashEmbedder(dim=64),
        vision_provider=StubVisionProvider(available=False, reason="vision 未配置 API Key"),
    )

    state = build_graph(deps).invoke(
        {"conversation_id": conv_id, "text": "报错E05怎么处理", "image_path": str(image)}
    )

    assert state["vision_ok"] is False
    assert "API Key" in state["vision_error"]


# ------------------------------------------------------------------ 多轮记忆

def test_multi_turn_keeps_draft_and_passes_history(conn, corpus, make_deps):
    conv_id = _conversation(conn)
    store = ConversationStore(conn)
    chain = FakeChatChain(text="第一轮回答：请检查触点。")
    deps = make_deps(chat_chain=chain, embedder=HashEmbedder(dim=64))
    graph = build_graph(deps)

    first_text = "机型L3250屏幕出现E05，硒鼓未识别"
    store.add_message(conv_id, "user", first_text)
    first = graph.invoke({"conversation_id": conv_id, "text": first_text})

    assert first["hit"] is True, "第一轮应当命中产品手册里的报错处理"
    assert first["answer"] == "第一轮回答：请检查触点。"
    store.add_message(conv_id, "assistant", first["answer"])
    assert store.get_draft(conv_id)["product_model"] == "L3250"

    chain.text = "第二轮回答：保修两年，人为损坏不在保修范围内。"
    second_text = "保修期是两年吗？人为损坏算不算"
    store.add_message(conv_id, "user", second_text)
    second = graph.invoke({"conversation_id": conv_id, "text": second_text})

    assert second["hit"] is True, "第二轮应当命中售后规则里的保修政策"
    assert len(chain.calls) == 2, "命中分支才会调用模型"
    prompt = "\n".join(str(item["content"]) for item in chain.calls[1])
    assert first_text in prompt, "第二轮提示词里应当带上第一轮的问答"
    assert "第一轮回答" in prompt
    assert second["answer"] == "第二轮回答：保修两年，人为损坏不在保修范围内。"
    assert store.get_draft(conv_id)["product_model"] == "L3250", "草稿应跨轮累积"


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
    assert "E05" in first["query"]

    store.add_message(conv_id, "assistant", first["answer"])
    store.add_message(conv_id, "user", "保修期是多久")
    second = graph.invoke(
        {"conversation_id": conv_id, "text": "保修期是多久", "image_path": None},
        config=config,
    )

    assert second["vision_summary"] == "", "上一轮的图片描述不应残留"
    assert second["vision_ok"] is False
    assert "E05" not in second["query"], "上一轮的图片信息不应混进本轮检索"
    assert second["ticket_draft"]["product_model"] == "L3250"


def test_good_vision_json_constant_matches_provider():
    """防止 fakes 里的假响应与 vision 解析字段脱节。"""
    import json

    payload = json.loads(GOOD_VISION_JSON)
    assert payload["error_codes"] == ["E05"]
    assert payload["product_model"] == "L3250"


def test_fallback_template_mentions_manual_handoff():
    assert "转人工" in FALLBACK_TEMPLATE
