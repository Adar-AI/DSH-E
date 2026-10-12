"""重排模块测试：假实现、兜底覆盖率、缓存、真实模型可用性。"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.rag.reranker import (
    CrossEncoderReranker,
    LexicalFallbackReranker,
    RerankerUnavailable,
    clear_reranker_cache,
    get_reranker,
)

QUERY = "更换硒鼓的步骤"
RELEVANT = "更换硒鼓的步骤：打开前盖，取出旧硒鼓，装入新硒鼓后合上前盖。"
RELATED = "硒鼓属于易损件，保修期为三个月。"
IRRELEVANT = "无线网络连接失败时，请检查路由器是否开启了 2.4G 频段。"


# ------------------------------------------------------------------ 兜底重排

def test_lexical_fallback_scores_related_higher():
    reranker = LexicalFallbackReranker()

    scores = reranker.score(QUERY, [RELEVANT, RELATED, IRRELEVANT])

    assert scores[0] > scores[1] > scores[2]
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_lexical_fallback_handles_empty_inputs():
    reranker = LexicalFallbackReranker()

    assert reranker.score(QUERY, []) == []
    assert reranker.score("", [RELEVANT]) == [0.0]
    assert reranker.score("！！！", [RELEVANT]) == [0.0]


def test_lexical_fallback_health():
    healthy, detail = LexicalFallbackReranker().health()

    assert healthy is True
    assert "词面覆盖率" in detail


# ------------------------------------------------------------------ 模型缺失

def test_missing_model_path_raises_and_reports_health():
    reranker = CrossEncoderReranker("D:/definitely/not/a/model", device="cpu")

    healthy, detail = reranker.health()
    assert healthy is False
    assert "不存在" in detail

    with pytest.raises(RerankerUnavailable):
        reranker.score(QUERY, [RELEVANT])


def test_score_empty_input_does_not_load_model():
    reranker = CrossEncoderReranker("D:/definitely/not/a/model")
    assert reranker.score(QUERY, []) == []


# ------------------------------------------------------------------ 装配

def test_get_reranker_falls_back_when_model_missing():
    clear_reranker_cache()
    settings = Settings(_env_file=None, rerank_model_path="D:/definitely/not/a/model")

    reranker = get_reranker(settings)

    assert isinstance(reranker, LexicalFallbackReranker)
    assert get_reranker(settings) is reranker, "应当缓存"
    clear_reranker_cache()


def test_get_reranker_without_path_uses_fallback():
    clear_reranker_cache()

    reranker = get_reranker(Settings(_env_file=None, rerank_model_path=""))

    assert isinstance(reranker, LexicalFallbackReranker)
    clear_reranker_cache()


# ------------------------------------------------------------------ 真实模型

@pytest.fixture(scope="session")
def real_reranker():
    settings = get_settings()
    if not settings.rerank_model_path:
        pytest.skip("未配置 RERANK_MODEL_PATH")
    reranker = CrossEncoderReranker(
        settings.rerank_model_path,
        device=settings.rerank_device,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
    )
    healthy, detail = reranker.health()
    if not healthy:
        pytest.skip(f"本机重排模型不可用：{detail}")
    return reranker


def test_real_reranker_separates_relevant_from_irrelevant(real_reranker):
    scores = real_reranker.score(QUERY, [RELEVANT, RELATED, IRRELEVANT])

    assert len(scores) == 3
    assert scores[0] > 0.5, f"命中片段分数偏低：{scores[0]}"
    assert scores[2] < 0.1, f"无关片段分数偏高：{scores[2]}"
    assert scores[0] > scores[1] > scores[2]


def test_real_reranker_scores_stay_in_zero_one(real_reranker):
    scores = real_reranker.score("云打 L3250 报错 E05", [RELEVANT, RELATED])
    assert all(0.0 <= score <= 1.0 for score in scores)
