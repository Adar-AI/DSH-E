"""嵌入模块测试：协议兼容、缓存、降级，以及本机 BGE-M3 的真实可用性。"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.embedder import (
    BGEEmbedder,
    EmbeddingUnavailable,
    clear_embedder_cache,
    get_embedder,
)
from fakes import BrokenEmbedder, HashEmbedder


def test_hash_embedder_matches_protocol_shape():
    embedder = HashEmbedder(dim=32)

    assert embedder.name == "hash-32"
    assert embedder.dim == 32
    assert callable(embedder.encode)

    vectors = embedder.encode(["更换硒鼓", "更换硒鼓", "硒鼓更换"])
    assert len(vectors) == 3
    assert all(len(vector) == 32 for vector in vectors)
    # 完全相同的文本 -> 余弦为 1
    assert sum(a * b for a, b in zip(vectors[0], vectors[1])) == pytest.approx(1.0)
    # 二元切分下「更换硒鼓」与「硒鼓更换」只共享一个 token -> 相似但不相同
    partial = sum(a * b for a, b in zip(vectors[0], vectors[2]))
    assert 0.0 < partial < 1.0


def test_get_embedder_is_cached_per_configuration():
    clear_embedder_cache()
    settings = get_settings()

    first = get_embedder(settings)
    second = get_embedder(settings)

    assert first is second
    assert isinstance(first, BGEEmbedder)
    clear_embedder_cache()


def test_missing_model_path_degrades_instead_of_crashing():
    embedder = BGEEmbedder("D:/definitely/not/a/real/model", device="cpu")

    healthy, detail = embedder.health()
    assert healthy is False
    assert "不存在" in detail

    with pytest.raises(EmbeddingUnavailable):
        embedder.encode(["任意文本"])


def test_broken_embedder_raises_unavailable():
    with pytest.raises(EmbeddingUnavailable):
        BrokenEmbedder().encode(["x"])


def test_encode_empty_input_returns_empty_without_loading_model():
    embedder = BGEEmbedder("D:/definitely/not/a/real/model", device="cpu")
    assert embedder.encode([]) == []


# ------------------------------------------------------------------ 真实模型

@pytest.fixture(scope="session")
def real_embedder():
    settings = get_settings()
    embedder = BGEEmbedder(
        settings.embedding_model_path, device="cpu", dim=settings.embedding_dim
    )
    healthy, detail = embedder.health()
    if not healthy:
        pytest.skip(f"本机嵌入模型不可用：{detail}")
    return embedder


def test_real_bge_m3_dimension_and_normalization(real_embedder):
    vectors = real_embedder.encode(["打印机报错 E05 怎么处理"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    norm = sum(value * value for value in vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_real_bge_m3_ranks_related_text_higher(real_embedder):
    vectors = real_embedder.encode(
        [
            "打印机报错 E05 怎么处理",
            "更换硒鼓的详细步骤",
            "今天天气不错，适合出门散步",
        ]
    )
    related = sum(a * b for a, b in zip(vectors[0], vectors[1]))
    unrelated = sum(a * b for a, b in zip(vectors[0], vectors[2]))

    assert related > unrelated
