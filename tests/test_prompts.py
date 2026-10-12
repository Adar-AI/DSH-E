"""提示词测试：反幻觉约束必须真的写进提示词，资料片段必须编号。"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.prompts import (
    ANSWER_SYSTEM,
    FALLBACK_TEMPLATE,
    build_answer_messages,
    build_fallback_messages,
    format_contexts,
)


@dataclass
class FakeChunk:
    citation: str
    text: str


def test_answer_system_enforces_internal_knowledge_only():
    assert "只能依据" in ANSWER_SYSTEM
    assert "内部资料中未收录" in ANSWER_SYSTEM
    assert "禁止使用你自己的知识" in ANSWER_SYSTEM
    assert "[序号]" in ANSWER_SYSTEM


def test_format_contexts_numbers_sources():
    contexts = [
        FakeChunk("产品手册.md · 第 12 页", "更换硒鼓的步骤……"),
        {"citation": "售后规则.md · 保修政策", "text": "整机保修两年。"},
    ]

    text = format_contexts(contexts)

    assert text.startswith("[1] 来源：产品手册.md · 第 12 页")
    assert "[2] 来源：售后规则.md · 保修政策" in text
    assert "整机保修两年。" in text


def test_format_contexts_empty():
    assert format_contexts([]) == "（无）"


def test_build_answer_messages_includes_history_vision_and_contexts():
    messages = build_answer_messages(
        "硒鼓怎么更换",
        [FakeChunk("产品手册.md · 第 12 页", "打开前盖，取出旧硒鼓。")],
        vision_summary="错误码：E05",
        history=[
            {"role": "user", "content": "我之前用的是 L3250"},
            {"role": "assistant", "content": "收到"},
            {"role": "system", "content": "应被忽略"},
        ],
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == ANSWER_SYSTEM
    assert [item["role"] for item in messages[1:3]] == ["user", "assistant"]

    final = messages[-1]["content"]
    assert "硒鼓怎么更换" in final
    assert "错误码：E05" in final
    assert "[1] 来源：产品手册.md · 第 12 页" in final


def test_build_answer_messages_without_vision_or_history():
    messages = build_answer_messages("保修多久", [FakeChunk("规则.md", "两年")])

    assert len(messages) == 2
    assert "客户上传图片" not in messages[-1]["content"]
    assert "两年" in messages[-1]["content"]


def test_build_fallback_messages_mentions_no_internal_record():
    messages = build_fallback_messages("火星上怎么修打印机", vision_summary="图片是一台打印机")

    content = messages[-1]["content"]
    assert "内部资料中没有检索到相关内容" in content
    assert "火星上怎么修打印机" in content
    assert "图片是一台打印机" in content
    assert "转人工" in FALLBACK_TEMPLATE
