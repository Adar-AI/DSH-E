"""混合检索测试：关键词召回、向量通道、RRF 融合、阈值兜底、降级。"""

from __future__ import annotations

import pytest

from app.db import to_vec_blob
from app.rag.retriever import (
    MODE_HYBRID,
    MODE_KEYWORD_ONLY,
    hybrid_retrieve,
    keyword_search,
)
from fakes import (
    BrokenEmbedder,
    HashEmbedder,
    ScriptedEmbedder,
    load_corpus,
    set_document_status,
)

CORPUS = [
    ("打印机产品手册.md", "更换硒鼓的步骤：打开前盖，取出旧硒鼓，装入新硒鼓后合上前盖。", "耗材更换", 12),
    ("售后规则.md", "本机型整机保修期为两年，自购买之日起计算，人为损坏不在保修范围内。", "保修政策", None),
    ("网络设置指南.md", "无线网络连接失败时，请检查路由器是否开启了 2.4G 频段。", "网络", 3),
]


@pytest.fixture()
def corpus(conn):
    return load_corpus(conn, CORPUS, HashEmbedder(dim=64))


def test_keyword_search_recalls_chinese(conn, corpus):
    hits = keyword_search(conn, "硒鼓", 5)

    assert hits, "中文关键词检索无召回"
    assert hits[0][0] == corpus[0]


def test_keyword_search_empty_query_returns_nothing(conn, corpus):
    assert keyword_search(conn, "", 5) == []
    assert keyword_search(conn, "！！！", 5) == []


def test_hybrid_retrieve_hits_with_citation(conn, settings, corpus):
    outcome = hybrid_retrieve(
        conn, "硒鼓怎么更换", embedder=HashEmbedder(dim=64), settings=settings
    )

    assert outcome.hit is True
    assert outcome.mode == MODE_HYBRID
    assert outcome.best_score >= settings.retrieve_min_score

    top = outcome.chunks[0]
    assert top.filename == "打印机产品手册.md"
    assert top.page == 12
    assert "耗材更换" in top.citation
    assert "第 12 页" in top.citation
    assert 0.0 <= top.coverage <= 1.0
    assert top.keyword_rank == 1


def test_keyword_only_when_embedder_is_none(conn, settings, corpus):
    outcome = hybrid_retrieve(conn, "硒鼓怎么更换", embedder=None, settings=settings)

    assert outcome.mode == MODE_KEYWORD_ONLY
    assert outcome.hit is True
    assert outcome.chunks[0].vector_sim == 0.0
    assert outcome.reason == "" or "向量" in outcome.reason


def test_keyword_only_when_embedder_broken(conn, settings, corpus):
    outcome = hybrid_retrieve(
        conn, "硒鼓怎么更换", embedder=BrokenEmbedder(), settings=settings
    )

    assert outcome.mode == MODE_KEYWORD_ONLY
    assert outcome.hit is True


def test_unrelated_question_is_not_a_hit(conn, settings, corpus):
    outcome = hybrid_retrieve(
        conn, "预订国际航班机票价格", embedder=HashEmbedder(dim=64), settings=settings
    )

    assert outcome.hit is False
    assert all(chunk.score < settings.retrieve_min_score for chunk in outcome.chunks)


def test_threshold_controls_hit_decision(conn, settings, corpus):
    strict = hybrid_retrieve(
        conn,
        "硒鼓怎么更换",
        embedder=HashEmbedder(dim=64),
        settings=settings,
        min_score=0.99,
    )
    loose = hybrid_retrieve(
        conn,
        "硒鼓怎么更换",
        embedder=HashEmbedder(dim=64),
        settings=settings,
        min_score=0.0,
    )

    assert strict.hit is False
    assert strict.chunks, "低于阈值时仍应返回片段供页面展示依据"
    assert loose.hit is True


def test_top_k_limits_returned_chunks(conn, settings, corpus):
    outcome = hybrid_retrieve(
        conn, "硒鼓 保修 网络", embedder=HashEmbedder(dim=64), settings=settings, top_k=1
    )
    assert len(outcome.chunks) == 1


def test_empty_question_returns_no_hit(conn, settings, corpus):
    outcome = hybrid_retrieve(conn, "   ", embedder=HashEmbedder(dim=64), settings=settings)

    assert outcome.hit is False
    assert outcome.chunks == ()
    assert outcome.reason == "问题为空"


def test_failed_document_is_excluded(conn, settings, corpus):
    set_document_status(conn, "打印机产品手册.md", "failed")

    outcome = hybrid_retrieve(
        conn, "硒鼓怎么更换", embedder=HashEmbedder(dim=64), settings=settings
    )

    assert all(chunk.filename != "打印机产品手册.md" for chunk in outcome.chunks)
    assert outcome.hit is False


def test_vector_channel_can_surface_semantic_match(conn, settings):
    """只有向量通道命中（词面完全不同）的片段也应该被召回。"""
    dim = settings.embedding_dim
    chunk_ids = load_corpus(
        conn,
        [
            ("A.md", "硒鼓更换方法", None, None),
            ("B.md", "耗材组件替换流程说明", None, None),
        ],
        embedder=None,
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
        (chunk_ids[0], to_vec_blob([0.0, 0.0, 1.0] + [0.0] * (dim - 3))),
    )
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
        (chunk_ids[1], to_vec_blob([0.9, 0.1, 0.0] + [0.0] * (dim - 3))),
    )
    conn.commit()

    embedder = ScriptedEmbedder(
        {"硒鼓": [1.0, 0.0, 0.0] + [0.0] * (dim - 3)}, dim=dim
    )
    outcome = hybrid_retrieve(conn, "硒鼓", embedder=embedder, settings=settings)

    assert outcome.mode == MODE_HYBRID
    filenames = [chunk.filename for chunk in outcome.chunks]
    assert "B.md" in filenames, "纯语义命中的片段没有被召回"
    assert filenames[0] == "A.md", "词面命中的片段应当排在前面"
    semantic = next(chunk for chunk in outcome.chunks if chunk.filename == "B.md")
    assert semantic.vector_rank == 1
    assert semantic.coverage == 0.0
