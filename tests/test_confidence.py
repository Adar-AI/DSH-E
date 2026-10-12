"""三档置信判定测试：实体抽取、实体池、两段判定、话术。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.rag.confidence import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    EntityHint,
    build_clarify_message,
    build_reject_message,
    collect_entity_pool,
    extract_entities,
    has_inline_entity,
    is_entity_dependent,
    judge_after_recall,
    judge_before_recall,
)


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None, confidence_high=0.50, confidence_low=0.10)


# ------------------------------------------------------------------ 实体抽取

def test_extract_entities_models_and_codes():
    entities = extract_entities("我的 L3250 报错 E05，还有 M1136 也不识别")

    assert ("L3250", "product_model") in entities
    assert ("M1136", "product_model") in entities
    assert ("E05", "error_code") in entities


def test_extract_entities_without_spaces():
    # 「报错E05」是实测过的高频写法（不能用 \b 做边界）
    assert ("E05", "error_code") in extract_entities("打印机报错E05怎么办")


def test_has_inline_entity():
    assert has_inline_entity("L3250 保修多久") is True
    assert has_inline_entity("保修期是多少？") is False


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("保修期是多少？", True),
        ("这个多少钱？", True),
        ("更换周期多久？", True),
        ("怎么开发票？", False),
        ("怎么退货？", False),
        ("你们上班时间？", False),
    ],
)
def test_is_entity_dependent(question, expected):
    assert is_entity_dependent(question) is expected


# ------------------------------------------------------------------ 实体池

def test_collect_entity_pool_prefers_image_and_draft_over_history():
    pool = collect_entity_pool(
        history_texts=["我之前用的是 L3250"],
        draft={"product_model": "M1136", "error_info": "E07"},
        vision_detail={"product_model": "M1136", "error_codes": ["E07"]},
    )
    by_name = {item.name: item for item in pool}

    assert set(by_name) == {"M1136", "E07", "L3250"}
    assert by_name["M1136"].source == "image"
    assert by_name["M1136"].weight == 1.0
    assert by_name["L3250"].weight == 0.6
    # 高可信来源排在前面，且型号优先于错误码（澄清时要问的是型号）
    assert [item.name for item in pool] == ["M1136", "E07", "L3250"]


def test_collect_entity_pool_from_history_only():
    pool = collect_entity_pool(history_texts=["L3250 报错 E05", "M1136 卡纸"])

    names = {item.name for item in pool}
    assert names == {"L3250", "M1136", "E05"}


def test_collect_entity_pool_empty():
    assert collect_entity_pool() == []


# ------------------------------------------------------------------ 第一段判定

def test_inline_entity_is_high_confidence(settings):
    result = judge_before_recall("L3250 保修期是多久", settings=settings)

    assert result.level == LEVEL_HIGH
    assert result.entities == ["L3250"]
    assert result.rewritten_query == "L3250 保修期是多久"


def test_non_entity_dependent_question_is_high(settings):
    result = judge_before_recall("怎么开发票", settings=settings)

    assert result.level == LEVEL_HIGH
    assert "不依赖具体型号" in result.reason


def test_missing_subject_with_single_context_entity_aligns(settings):
    """验收场景 1：上下文已有 L3250 → 对齐后正常作答。"""
    pool = collect_entity_pool(history_texts=["我的打印机是 L3250，报错 E05"])

    result = judge_before_recall("保修期是多少？", context_entities=pool, settings=settings)

    assert result.level == LEVEL_HIGH
    assert result.context_entities == ["L3250"]
    assert result.rewritten_query == "L3250 保修期是多少？"
    assert result.should_clarify is False


def test_missing_subject_with_multiple_context_entities_clarifies(settings):
    """验收场景 2：上下文有两个型号 → 反问，且不生成答案。"""
    pool = collect_entity_pool(history_texts=["L3250 报错 E05", "另外一台 M1136 也卡纸"])

    result = judge_before_recall("保修期是多少？", context_entities=pool, settings=settings)

    assert result.level == LEVEL_MEDIUM
    assert result.should_clarify is True
    assert set(result.context_entities) == {"L3250", "M1136"}
    assert result.needs_llm_align is True
    assert "L3250" in result.clarify_question and "M1136" in result.clarify_question
    assert "不同型号" in result.clarify_question


def test_missing_subject_without_context_rejects(settings):
    """验收场景 3：全新会话 → 拒答并说明需要型号。"""
    result = judge_before_recall("保修期是多少？", context_entities=[], settings=settings)

    assert result.level == LEVEL_LOW
    assert result.should_reject is True
    assert "缺少必要信息" in result.reject_message
    assert "产品型号" in result.reject_message


def test_error_code_in_context_does_not_count_as_model(settings):
    """上下文只有错误码、没有型号 → 仍应拒答（错误码不足以确定保修政策）。"""
    pool = collect_entity_pool(history_texts=["屏幕显示 E05"])

    result = judge_before_recall("保修期是多少？", context_entities=pool, settings=settings)

    assert result.level == LEVEL_LOW


def test_single_model_plus_error_code_still_aligns(settings):
    pool = collect_entity_pool(history_texts=["L3250 报错 E05"])

    result = judge_before_recall("这个多少钱？", context_entities=pool, settings=settings)

    assert result.level == LEVEL_HIGH
    assert result.context_entities == ["L3250"]


def test_clarify_options_can_be_enriched_from_knowledge_base(settings):
    """上下文只有一个型号但有多个候选时，话术里带上知识库里的其他型号作为参考。"""
    pool = collect_entity_pool(draft={"product_model": "L3250"})

    result = judge_before_recall(
        "保修期是多少？",
        context_entities=pool,
        extra_options=["M1136"],
        settings=settings,
    )

    # 上下文唯一 → 仍然对齐，不反问
    assert result.level == LEVEL_HIGH


# ------------------------------------------------------------------ 第二段判定

def test_final_judgement_high(settings):
    result = judge_after_recall("硒鼓怎么更换", top1=0.98, kept=1, settings=settings)

    assert result.level == LEVEL_HIGH
    assert "≥ 高置信阈值" in result.reason


def test_final_judgement_medium_between_thresholds(settings):
    result = judge_after_recall(
        "保修期是多少？",
        top1=0.36,
        kept=1,
        pre=judge_before_recall("保修期是多少？", context_entities=[], settings=settings),
        settings=settings,
    )

    assert result.level == LEVEL_MEDIUM
    assert result.should_clarify is True
    assert result.reject_message == ""


def test_final_judgement_low_when_no_evidence(settings):
    result = judge_after_recall("预订国际航班机票价格", top1=0.0, kept=0, settings=settings)

    assert result.level == LEVEL_LOW
    assert result.should_reject is True
    assert "低于低置信阈值" in result.reason


def test_final_judgement_low_when_top1_below_threshold(settings):
    result = judge_after_recall("这个多少钱？", top1=0.015, kept=1, settings=settings)

    assert result.level == LEVEL_LOW


def test_final_judgement_boundaries_are_inclusive(settings):
    assert judge_after_recall("x", top1=0.50, kept=1, settings=settings).level == LEVEL_HIGH
    assert judge_after_recall("x", top1=0.10, kept=1, settings=settings).level == LEVEL_MEDIUM
    assert judge_after_recall("x", top1=0.099, kept=1, settings=settings).level == LEVEL_LOW


# ------------------------------------------------------------------ 话术

def test_clarify_message_lists_candidates_and_does_not_answer():
    message = build_clarify_message("保修期是多少？", ["L3250", "M1136"])

    assert "L3250" in message and "M1136" in message
    assert "不同型号" in message
    assert "我不猜" in message
    # 不应给出任何结论性答复
    assert "两年" not in message and "三个月" not in message


def test_clarify_message_falls_back_when_candidates_missing():
    message = build_clarify_message("保修期是多少？", [])
    assert "具体型号" in message


def test_reject_message_mentions_what_to_provide():
    message = build_reject_message("保修期是多少？")

    assert "缺少必要信息" in message
    assert "产品型号" in message
    assert "L3250" in message  # 给出填写示例
    assert "转人工" in message
    # 与"资料未收录"兜底话术区分开
    assert "未收录" not in message


def test_topic_extraction_strips_punctuation():
    message = build_reject_message("保修期是多少？？？")
    assert "「保修期是多少」" in message
