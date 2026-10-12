"""融合与动态 TopK 测试：RRF 数学性质、来源权重、候选构造、三重截断约束。"""

from __future__ import annotations

import pytest

from app.rag.fusion import (
    Candidate,
    RRF_K,
    SOURCE_INTERNAL,
    SOURCE_WEB,
    apply_rerank_scores,
    rrf_fuse,
    select_supporting,
    summarize_sources,
)


def internal(chunk_id: int, text: str, *, citation: str = "", score: float = 0.0) -> Candidate:
    return Candidate.from_retrieved(
        {
            "chunk_id": chunk_id,
            "doc_id": 1,
            "filename": "手册.md",
            "section": "耗材更换",
            "page": 12,
            "text": text,
            "citation": citation or f"手册.md · 第 12 页 · 耗材更换",
            "score": score,
        },
        channel="internal:q1",
    )


def web(url: str, title: str, snippet: str) -> Candidate:
    return Candidate.from_web(
        {"url": url, "title": title, "hostname": "example.com", "snippet": snippet},
        channel="web",
    )


# ------------------------------------------------------------------ 候选构造

def test_candidate_from_retrieved_keeps_locator_info():
    candidate = internal(7, "更换硒鼓的步骤", citation="手册.md · 第 12 页 · 耗材更换")

    assert candidate.key == "chunk:7"
    assert candidate.source == SOURCE_INTERNAL
    assert candidate.chunk_id == 7
    assert candidate.page == 12
    assert candidate.section == "耗材更换"
    assert candidate.is_web is False


def test_candidate_from_web_builds_citation():
    candidate = web("https://example.com/a", "打印机 E05 解决方法", "先检查触点")

    assert candidate.key == "web:https://example.com/a"
    assert candidate.source == SOURCE_WEB
    assert candidate.is_web is True
    assert "打印机 E05 解决方法" in candidate.citation
    assert "example.com" in candidate.citation


# ------------------------------------------------------------------ RRF 融合

def test_rrf_prefers_candidate_ranked_first_in_both_channels():
    shared = internal(1, "两路都排第一")
    only_internal = internal(2, "只在内部排第二")

    fused = rrf_fuse(
        {
            "internal:q1": [shared, only_internal],
            "internal:q2": [internal(1, "两路都排第一"), internal(3, "另一个查询变体的第二条")],
        }
    )

    assert fused[0].key == "chunk:1"
    assert fused[0].ranks == {"internal:q1": 1, "internal:q2": 1}
    by_key = {item.key: item for item in fused}
    assert fused[0].rrf > by_key["chunk:2"].rrf, "两路都第一的片段应高于只在单路第二的片段"


def test_internal_and_web_candidates_never_merge():
    """同文本但来源不同 → 保持两个独立候选，便于分别标注来源与审计。"""
    fused = rrf_fuse(
        {
            "internal:q1": [internal(1, "相同文本")],
            "web": [web("https://x/1", "网上标题", "相同文本")],
        }
    )

    assert len(fused) == 2
    assert {item.source for item in fused} == {SOURCE_INTERNAL, SOURCE_WEB}
    assert all(len(item.ranks) == 1 for item in fused)


def test_rrf_accumulates_scores_and_normalizes_to_one():
    first = internal(1, "A")
    second = internal(2, "B")

    fused = rrf_fuse({"internal:q1": [first, second]})

    # 单通道、排第一 → 理论满分即归一化 1.0
    assert fused[0].rrf_norm == pytest.approx(1.0)
    assert 0.0 < fused[1].rrf_norm < 1.0
    assert fused[1].rrf_norm == pytest.approx((1 / (RRF_K + 2)) / (1 / (RRF_K + 1)))


def test_rrf_applies_source_weights():
    internal_candidate = internal(1, "内部")
    web_candidate = web("https://x/1", "网上", "联网")

    fused = rrf_fuse(
        {"internal:q1": [internal_candidate], "web": [web_candidate]},
        weights={SOURCE_INTERNAL: 1.0, SOURCE_WEB: 0.5},
    )

    by_key = {item.key: item for item in fused}
    assert by_key[internal_candidate.key].rrf == pytest.approx(1 / (RRF_K + 1))
    assert by_key[web_candidate.key].rrf == pytest.approx(0.5 / (RRF_K + 1))
    assert fused[0].key == internal_candidate.key, "内部候选应排在联网之前"


def test_rrf_merges_same_chunk_across_query_variants():
    same = internal(5, "同一条片段")
    fused = rrf_fuse(
        {
            "internal:q1": [same],
            "internal:q2": [internal(5, "同一条片段，来自另一个查询变体")],
        }
    )

    assert len(fused) == 1
    assert fused[0].ranks == {"internal:q1": 1, "internal:q2": 1}
    assert fused[0].text == "同一条片段", "保留先出现的那份"


def test_rrf_prefers_internal_text_when_same_key_in_both_sources():
    web_candidate = web("https://x/1", "网上标题", "网上摘要")
    web_candidate.key = "chunk:9"  # 人为制造同键
    internal_candidate = internal(9, "内部更完整的正文")

    fused = rrf_fuse({"web": [web_candidate], "internal:q1": [internal_candidate]})

    assert len(fused) == 1
    assert fused[0].text == "内部更完整的正文"


def test_rrf_handles_empty_channels():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"internal:q1": []}) == []


def test_summarize_sources_counts_by_source():
    candidates = [internal(1, "a"), internal(2, "b"), web("https://x/1", "t", "s")]

    assert summarize_sources(candidates) == {SOURCE_INTERNAL: 2, SOURCE_WEB: 1}


# ------------------------------------------------------------------ 精排写回

def test_apply_rerank_scores_sorts_and_annotates():
    first = internal(1, "A")
    second = internal(2, "B")
    original = rrf_fuse({"internal:q1": [first, second]})

    reranked = apply_rerank_scores(original, [0.2, 0.9])

    assert [item.chunk_id for item in reranked] == [2, 1]
    assert reranked[0].rerank_score == 0.9
    assert reranked[0].score == 0.9, "有精排分时应优先用精排分"


def test_apply_rerank_scores_rejects_length_mismatch():
    candidates = [internal(1, "A")]

    with pytest.raises(ValueError):
        apply_rerank_scores(candidates, [0.1, 0.2])


# ------------------------------------------------------------------ 动态 TopK

def scored(values: list[float]) -> list[Candidate]:
    candidates = [internal(index + 1, f"片段{index + 1}") for index in range(len(values))]
    return apply_rerank_scores(candidates, values)


def test_select_supporting_keeps_above_absolute_floor_only():
    candidates = scored([0.61, 0.55, 0.52, 0.05, 0.01])

    result = select_supporting(candidates, min_score=0.10, relative_floor=0.35, top_k_max=6)

    assert len(result.kept) == 3
    assert result.cutoff == "floor"
    assert result.top1 == pytest.approx(0.61)


def test_select_supporting_applies_relative_floor():
    # 0.20 < 0.9*0.35=0.315 → 在第 3 条被相对地板挡住
    candidates = scored([0.90, 0.60, 0.20, 0.18])

    result = select_supporting(candidates, min_score=0.10, relative_floor=0.35)

    assert [item.score for item in result.kept] == [0.90, 0.60]
    assert result.cutoff == "relative"
    assert result.floor == pytest.approx(0.315)


def test_select_supporting_truncates_at_cliff():
    # 0.40 < 0.90*0.5 → 断崖；虽然 0.40 远高于地板，仍然截断
    candidates = scored([0.90, 0.85, 0.40, 0.38])

    result = select_supporting(candidates, min_score=0.10, relative_floor=0.35, cliff_ratio=0.5)

    assert [item.score for item in result.kept] == [0.90, 0.85]
    assert result.cutoff == "cliff"


def test_select_supporting_respects_top_k_cap():
    candidates = scored([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    result = select_supporting(candidates, min_score=0.1, relative_floor=0.35, top_k_max=3)

    assert len(result.kept) == 3
    assert result.cutoff == "cap"
    assert result.dropped == 5


def test_select_supporting_empty_when_all_below_min_score():
    candidates = scored([0.08, 0.05, 0.01])

    result = select_supporting(candidates, min_score=0.10)

    assert result.is_empty is True
    assert result.dropped == 3
    assert result.top1 == pytest.approx(0.08)


def test_select_supporting_without_candidates():
    result = select_supporting([])

    assert result.is_empty is True
    assert "没有候选" in result.reason


def test_select_supporting_uses_rrf_when_no_rerank():
    """没有精排时用归一化融合分，同一套阈值也能用。"""
    candidates = rrf_fuse({"internal:q1": [internal(1, "A"), internal(2, "B")]})

    result = select_supporting(candidates, min_score=0.10, relative_floor=0.35)

    assert result.is_empty is False
    assert result.kept[0].rerank_score is None
    assert result.top1 == pytest.approx(1.0)


def test_selection_result_as_dict_is_trace_friendly():
    result = select_supporting(scored([0.9, 0.4]))

    payload = result.as_dict()

    assert set(payload) == {"kept", "dropped", "top1", "floor", "cutoff", "reason"}
    assert payload["kept"] == 1


def test_candidate_as_dict_includes_source_and_scores():
    candidate = internal(3, "正文", score=0.5)
    candidate.rerank_score = 0.77

    payload = candidate.as_dict()

    assert payload["source"] == SOURCE_INTERNAL
    assert payload["key"] == "chunk:3"
    assert payload["score"] == 0.77
    assert payload["chunk_id"] == 3
