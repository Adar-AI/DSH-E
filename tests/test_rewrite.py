"""查询改写测试：规则变体、LLM 对齐（含失败退回）、结果合并。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.rag.confidence import EntityHint, collect_entity_pool
from app.rag.rewrite import (
    AlignmentResult,
    align_with_llm,
    build_query_variants,
    merge_alignment,
)
from fakes import FakeChatChain

GOOD_ALIGN_JSON = """{
  "intent": "售后咨询",
  "entities": ["L3250"],
  "queries": ["L3250 保修期", "云打 L3250 保修政策"],
  "confidence": "high",
  "reason": "客户上文提到过 L3250"
}"""


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None)


# ------------------------------------------------------------------ 规则变体

def test_variants_put_aligned_query_first():
    variants = build_query_variants("保修期是多少？", entities=["L3250"])

    assert variants[0] == "L3250 保修期是多少？"
    assert variants[1] == "保修期是多少？"


def test_variants_do_not_duplicate_entity_already_present():
    variants = build_query_variants("L3250 保修期", entities=["L3250"])

    assert variants == ["L3250 保修期"]


def test_variants_add_error_code_query():
    variants = build_query_variants(
        "屏幕上有个提示", entities=["L3250"], error_codes=["E05"]
    )

    assert any("E05" in item and "报错" in item for item in variants)


def test_variants_are_deduped_and_capped():
    variants = build_query_variants("问题", entities=["A", "B"], error_codes=["C"], max_variants=2)

    assert len(variants) == 2
    assert len(set(variants)) == 2


def test_variants_handle_empty_question():
    assert build_query_variants("", entities=["L3250"]) == ["L3250"]


def test_variants_collapse_whitespace_and_truncate():
    variants = build_query_variants("  问   题  ", entities=[])

    assert variants == ["问 题"]
    assert len(build_query_variants("长" * 900, entities=[])[0]) <= 400


# ------------------------------------------------------------------ LLM 对齐

def test_align_with_llm_parses_json(settings):
    chain = FakeChatChain(text=GOOD_ALIGN_JSON)

    result = align_with_llm("保修期是多少？", chat_chain=chain, settings=settings)

    assert isinstance(result, AlignmentResult)
    assert result.intent == "售后咨询"
    assert result.entities == ["L3250"]
    assert result.queries == ["L3250 保修期", "云打 L3250 保修政策"]
    assert result.confidence == "high"


def test_align_with_llm_accepts_chinese_comma_lists(settings):
    chain = FakeChatChain(
        text='{"intent": "故障报修", "entities": "L3250、E05", "queries": ["a", "b"], "confidence": "medium"}'
    )

    result = align_with_llm("报错", chat_chain=chain, settings=settings)

    assert result.entities == ["L3250", "E05"]
    assert result.intent == "故障报修"


@pytest.mark.parametrize(
    "reply",
    [
        "这里没有 JSON",
        '{"intent": "售后咨询"}',
        "```json\n{不是合法 JSON}\n```",
        "",
    ],
)
def test_align_with_llm_returns_none_on_bad_payload(reply, settings):
    chain = FakeChatChain(text=reply)

    assert align_with_llm("保修期是多少？", chat_chain=chain, settings=settings) is None


def test_align_with_llm_returns_none_on_provider_error(settings):
    chain = FakeChatChain(error="Ollama 未启动")

    assert align_with_llm("保修期是多少？", chat_chain=chain, settings=settings) is None


def test_align_with_llm_without_chain_returns_none(settings):
    assert align_with_llm("保修期是多少？", chat_chain=None, settings=settings) is None


def test_align_with_llm_normalizes_unknown_intent_and_confidence(settings):
    chain = FakeChatChain(text='{"intent": "乱写的", "confidence": "超高", "queries": ["q"]}')

    result = align_with_llm("x", chat_chain=chain, settings=settings)

    assert result.intent == "售后咨询"
    assert result.confidence == ""


def test_align_messages_include_entity_whitelist(settings):
    chain = FakeChatChain(text=GOOD_ALIGN_JSON)
    pool = collect_entity_pool(history_texts=["我用的是 L3250"])

    align_with_llm("保修期是多少？", context_entities=pool, chat_chain=chain, settings=settings)

    prompt = chain.calls[0][-1]["content"]
    assert "L3250" in prompt
    assert "禁止编造" in chain.calls[0][0]["content"]


# ------------------------------------------------------------------ 结果合并

def test_merge_alignment_appends_llm_queries_and_entities():
    queries, entities, confidence = merge_alignment(
        rule_queries=["规则查询"],
        rule_entities=["L3250"],
        alignment=AlignmentResult(
            queries=["LLM 查询", "规则查询"], entities=["L3250", "M1136"], confidence="medium"
        ),
        max_variants=3,
    )

    assert queries == ["规则查询", "LLM 查询"], "规则结果保底在前，且去重"
    assert entities == ["L3250", "M1136"]
    assert confidence == "medium"


def test_merge_alignment_without_llm_keeps_rule_result():
    queries, entities, confidence = merge_alignment(
        rule_queries=["规则查询"], rule_entities=["L3250"], alignment=None
    )

    assert queries == ["规则查询"]
    assert entities == ["L3250"]
    assert confidence == ""


def test_merge_alignment_respects_max_variants():
    queries, _, _ = merge_alignment(
        rule_queries=["a", "b", "c"],
        rule_entities=[],
        alignment=AlignmentResult(queries=["d"]),
        max_variants=2,
    )

    assert queries == ["a", "b"]


def test_entity_hint_used_by_merge():
    hint = EntityHint(name="L3250", kind="product_model")
    assert hint.as_dict()["kind"] == "product_model"
