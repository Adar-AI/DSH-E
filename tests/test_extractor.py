"""工单字段抽取测试：转录、LLM 结构化、校验、重试、规则兜底。"""

from __future__ import annotations

import json

import pytest

from app.tickets.extractor import (
    TicketFields,
    build_transcript,
    extract_ticket,
)
from fakes import GOOD_TICKET_JSON, FakeChatChain


def make_messages():
    return [
        {"role": "user", "text": "我的打印机报错E05，已经换了新硒鼓"},
        {
            "role": "assistant",
            "text": "请检查硒鼓芯片触点，并在维护菜单复位计数器。",
            "citations": ["常见故障排查表.md · 第 1 页"],
        },
        {"role": "user", "text": "好的，那保修期是多久", "image_path": "data/uploads/a.png",
         "vision": {"error_codes": ["E05"], "product_model": "L3250", "summary": "提示 E05"}},
    ]


# ------------------------------------------------------------------ 转录

def test_build_transcript_labels_roles_and_notes_images():
    transcript = build_transcript(make_messages())

    assert "客户：我的打印机报错E05，已经换了新硒鼓" in transcript
    assert "客服：请检查硒鼓芯片触点" in transcript
    assert "客户上传了图片" in transcript
    assert "错误码 E05" in transcript
    assert "型号 L3250" in transcript


def test_build_transcript_skips_empty_and_truncates():
    messages = [{"role": "user", "text": ""}, {"role": "user", "text": "有效内容"}]
    assert build_transcript(messages) == "客户：有效内容"

    long_messages = [{"role": "user", "text": "很长的内容" * 500}]
    transcript = build_transcript(long_messages, max_chars=100)
    assert len(transcript) <= 100


# ------------------------------------------------------------------ 字段模型

def test_ticket_fields_normalizes_lists_and_urgency():
    fields = TicketFields.model_validate(
        {
            "error_info": ["E05", "E06"],
            "tried_actions": ["重启", "换硒鼓"],
            "urgency": "比较紧急",
            "unknown_field": "忽略",
        }
    )

    assert fields.error_info == "E05、E06"
    assert fields.tried_actions == "重启、换硒鼓"
    assert fields.urgency == "紧急"
    assert not hasattr(fields, "unknown_field")


def test_ticket_fields_defaults():
    fields = TicketFields.model_validate({})
    assert fields.customer_request == ""
    assert fields.urgency == "普通"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("紧急", "紧急"), ("较急", "较急"), ("普通", "普通"), ("有点急", "较急"), ("", "普通"), ("随便", "普通")],
)
def test_urgency_normalization(raw, expected):
    assert TicketFields.model_validate({"urgency": raw}).urgency == expected


# ------------------------------------------------------------------ 抽取主流程

def test_extract_without_chain_uses_draft():
    result = extract_ticket(
        make_messages(), draft={"product_model": "L3250", "error_info": "E05"}, chat_chain=None
    )

    assert result.source == "draft"
    assert result.fields["product_model"] == "L3250"
    assert result.filled[0] >= 2


def test_extract_with_empty_conversation():
    result = extract_ticket([], draft={}, chat_chain=FakeChatChain(text=GOOD_TICKET_JSON))

    assert result.source == "draft"
    assert "没有可抽取的对话内容" in result.error


def test_extract_uses_llm_and_fills_gaps_from_draft():
    chain = FakeChatChain(text=GOOD_TICKET_JSON)
    # 规则草稿里有 citations，模型不负责这一项，应当被保留下来
    result = extract_ticket(
        make_messages(),
        draft={"citations": ["常见故障排查表.md · 第 1 页"], "error_info": "E05"},
        chat_chain=chain,
    )

    assert result.source == "llm"
    assert result.attempts == 1
    assert result.fields["customer_request"].startswith("打印机报错 E05")
    assert result.fields["product_model"] == "L3250"
    assert result.fields["urgency"] == "较急"
    assert result.fields["citations"] == ["常见故障排查表.md · 第 1 页"]
    assert result.ok is True

    prompt = chain.calls[0][-1]["content"]
    assert "只输出 JSON" in prompt
    assert "客户：我的打印机报错E05" in prompt


def test_extract_accepts_json_wrapped_in_prose_and_fence():
    chain = FakeChatChain(text=f"好的，结果如下：\n```json\n{GOOD_TICKET_JSON}\n```\n以上。")

    result = extract_ticket(make_messages(), chat_chain=chain)

    assert result.source == "llm"
    assert result.fields["error_info"] == "E05"


def test_extract_accepts_multiline_field_values():
    """实测本地模型会把整段回答塞进 suggestion 并带真实换行，必须能解析并压成单行。"""
    multiline = (
        '{\n  "customer_request": "报错E05",\n'
        '  "product_model": "L3250",\n'
        '  "error_info": "E05",\n'
        '  "suggestion": "第一步：检查硒鼓触点。\\n第二步：复位计数器。\n第三步：联系售后"\n}'
    )
    chain = FakeChatChain(text=multiline)

    result = extract_ticket(make_messages(), chat_chain=chain)

    assert result.source == "llm", f"多行字段值不应导致回退：{result.error}"
    assert result.attempts == 1
    assert "\n" not in result.fields["suggestion"]
    assert "第一步" in result.fields["suggestion"]
    assert "第三步" in result.fields["suggestion"]


def test_extract_retries_once_then_succeeds():
    chain = FakeChatChain(text="这不是 JSON")
    answers = iter(["这不是 JSON", GOOD_TICKET_JSON])

    def fake_chat(messages, **overrides):
        chain.calls.append(list(messages))
        from app.llm.providers import ChatOutcome

        return ChatOutcome(text=next(answers), provider="fake", model="fake-model")

    chain.chat = fake_chat  # type: ignore[assignment]
    result = extract_ticket(make_messages(), chat_chain=chain)

    assert result.source == "llm"
    assert result.attempts == 2
    assert "只输出一个 JSON 对象" in chain.calls[1][-1]["content"]


def test_extract_falls_back_to_draft_when_json_invalid_twice():
    chain = FakeChatChain(text="完全不是 JSON")
    result = extract_ticket(
        make_messages(),
        draft={"product_model": "L3250", "error_info": "E05", "customer_request": "报错"},
        chat_chain=chain,
    )

    assert result.source == "draft"
    assert result.attempts == 2
    assert "没有返回可解析的 JSON" in result.error
    assert result.fields["product_model"] == "L3250"
    assert len(chain.calls) == 2


def test_extract_falls_back_when_model_errors():
    chain = FakeChatChain(error="所有对话通道都不可用")
    result = extract_ticket(make_messages(), draft={"error_info": "E05"}, chat_chain=chain)

    assert result.source == "draft"
    assert "模型调用失败" in result.error
    assert result.fields["error_info"] == "E05"


def test_extract_handles_validation_failure():
    chain = FakeChatChain(text='{"urgency": {"nested": "dict"}}')
    result = extract_ticket(make_messages(), draft={"error_info": "E05"}, chat_chain=chain)

    # 嵌套字典会让 urgency 变成 "普通"，不会抛异常
    assert result.source in {"llm", "draft"}
    assert isinstance(result.fields, dict)


def test_extraction_as_dict_shape():
    result = extract_ticket(make_messages(), draft={"error_info": "E05"}, chat_chain=None)
    payload = result.as_dict()

    assert set(payload) == {"fields", "source", "attempts", "error", "filled"}
    assert payload["source"] == "draft"
    assert isinstance(payload["fields"], dict)


def test_good_ticket_json_matches_schema():
    data = json.loads(GOOD_TICKET_JSON)
    fields = TicketFields.model_validate(data)
    assert fields.error_info == "E05"
    assert fields.urgency == "较急"
