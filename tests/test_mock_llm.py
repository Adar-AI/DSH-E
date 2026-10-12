"""离线假模型测试：后端逻辑不依赖真实模型也能跑通的保证。"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.mock import (
    MOCK_BADGE,
    MockChatChain,
    MockVisionProvider,
    build_mock_answer,
    build_mock_reply,
    build_mock_ticket_json,
    is_mock_mode,
    parse_contexts,
)
from app.llm.prompts import build_answer_messages, build_ticket_messages

ANSWER_PROMPT_CONTEXTS = [
    {"citation": "常见故障排查表.md · 第 1 页 · 屏幕报错码对照", "text": "E05 表示硒鼓未识别，请重新安装硒鼓并清洁芯片触点。"},
    {"citation": "售后服务规则.md · 保修政策", "text": "整机保修两年，硒鼓保修三个月。"},
]


# ------------------------------------------------------------------ 提示词解析

def test_parse_contexts_from_real_answer_prompt():
    messages = build_answer_messages("报错E05怎么办", ANSWER_PROMPT_CONTEXTS)
    contexts = parse_contexts(messages[-1]["content"])

    assert len(contexts) == 2
    assert contexts[0][0].startswith("常见故障排查表.md")
    assert "E05" in contexts[0][1]


def test_parse_contexts_handles_missing_section():
    assert parse_contexts("没有资料片段的提示词") == []


# ------------------------------------------------------------------ 回答生成

def test_mock_answer_quotes_first_context_with_citation_marker():
    messages = build_answer_messages("报错E05怎么办", ANSWER_PROMPT_CONTEXTS)

    answer = build_mock_reply(messages)

    assert answer.startswith(MOCK_BADGE)
    assert "常见故障排查表.md" in answer
    assert "E05" in answer
    assert "[1]" in answer


def test_mock_answer_without_contexts_is_explicit():
    answer = build_mock_reply([{"role": "user", "content": "【客户问题】\n你好"}])

    assert MOCK_BADGE in answer
    assert "无法给出结论" in answer


# ------------------------------------------------------------------ 工单抽取

def test_mock_reply_returns_ticket_json_for_extraction_prompt():
    prompt = build_ticket_messages(
        "客户：打印机报错E05\n客服：请检查触点",
        {"customer_request": "报错E05", "product_model": "L3250", "error_info": "E05", "urgency": "较急"},
    )

    reply = build_mock_reply(prompt)
    data = json.loads(reply)

    assert data["customer_request"] == "报错E05"
    assert data["product_model"] == "L3250"
    assert data["error_info"] == "E05"
    assert data["urgency"] == "较急"
    assert "citations" not in data


def test_mock_ticket_json_defaults_missing_fields():
    payload = json.loads(build_mock_ticket_json("（无）"))
    assert set(payload) == {
        "customer_request",
        "product_model",
        "fault_phenomenon",
        "error_info",
        "tried_actions",
        "urgency",
        "suggestion",
    }
    assert payload["urgency"] == "普通"


# ------------------------------------------------------------------ 通道接口

def test_mock_chat_chain_matches_chain_interface():
    chain = MockChatChain()
    messages = build_answer_messages("报错E05", ANSWER_PROMPT_CONTEXTS)

    outcome = chain.chat(messages)
    assert outcome.provider == "mock"
    assert outcome.text.startswith(MOCK_BADGE)

    provider_name, iterator = chain.stream(messages)
    assert provider_name == "mock"
    assert "".join(iterator) == outcome.text

    described = chain.describe()
    assert described[0]["name"] == "mock"
    assert described[0]["available"] is True
    assert chain.providers[0].ping() == (True, "离线模式就绪")


def test_mock_vision_provider_returns_valid_json():
    provider = MockVisionProvider()

    assert provider.available()[0] is True
    payload = json.loads(provider.chat([{"role": "user", "content": "看图"}]))
    assert "summary" in payload
    assert payload["confidence"] == 0.0
    assert "未真正识别" in payload["summary"]


# ------------------------------------------------------------------ 开关判断

@pytest.mark.parametrize(
    ("value", "expected"),
    [("mock", True), ("MOCK", True), (" mock ", True), ("auto", False), ("", False)],
)
def test_is_mock_mode(value, expected):
    settings = Settings(_env_file=None, llm_mode=value)
    assert is_mock_mode(settings) is expected


def test_deps_switches_to_mock_in_mock_mode(api_env, monkeypatch):
    """端到端开关验证：LLM_MODE=mock 时接口层拿到的是假模型。"""
    from app import deps as deps_module

    deps_module.reset_caches()
    mock_settings = api_env.model_copy(update={"llm_mode": "mock"})

    chain = deps_module.get_chat_chain(mock_settings)
    vision = deps_module.get_vision_provider(mock_settings)

    assert isinstance(chain, MockChatChain)
    assert isinstance(vision, MockVisionProvider)
    deps_module.reset_caches()


def test_deps_uses_real_providers_in_auto_mode(api_env):
    from app import deps as deps_module
    from app.llm.providers import ChatChain

    deps_module.reset_caches()
    auto_settings = api_env.model_copy(update={"llm_mode": "auto"})

    chain = deps_module.get_chat_chain(auto_settings)

    assert isinstance(chain, ChatChain)
    assert [provider.name for provider in chain.providers][0] == "ollama"
    deps_module.reset_caches()
