"""入库管线测试：写库、去重、强制重建、降级、删除、重建索引。"""

from __future__ import annotations

import pytest

from app.db import knowledge_stats
from app.ingest.pipeline import (
    MODE_HYBRID,
    MODE_KEYWORD_ONLY,
    STATUS_FAILED,
    STATUS_INGESTED,
    STATUS_SKIPPED,
    delete_document,
    ingest_directory,
    ingest_file,
    rebuild_index,
    reembed_document,
)
from app.rag.retriever import hybrid_retrieve
from fakes import BrokenEmbedder, HashEmbedder, write_file

MANUAL = """# 打印机使用手册

## 耗材更换

打开前盖，取出旧硒鼓，装入新硒鼓后合上前盖。更换完成后请复位计数器。

## 报错处理

屏幕出现 E05 表示硒鼓未识别，请重新安装硒鼓并确认芯片接触良好。保修期内可免费更换。
"""


@pytest.fixture()
def manual_path(work_dir):
    return write_file(work_dir, "手册.md", MANUAL)


def test_ingest_creates_document_chunks_and_vectors(conn, small_chunk_settings, manual_path):
    result = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=HashEmbedder(), source="test"
    )

    assert result.status == STATUS_INGESTED
    assert result.chunks >= 2
    assert result.vectors == result.chunks
    assert result.mode == MODE_HYBRID

    stats = knowledge_stats(conn)
    assert stats["documents"] == 1
    assert stats["chunks"] == result.chunks
    assert stats["vectors"] == result.chunks

    row = conn.execute(
        "SELECT chunk_count, rel_path, status, error FROM documents WHERE id = ?",
        (result.doc_id,),
    ).fetchone()
    assert row["chunk_count"] == result.chunks
    assert row["status"] == "ready"
    assert row["error"] is None
    assert str(row["rel_path"]).endswith("手册.md")

    sections = [
        item[0]
        for item in conn.execute(
            "SELECT section FROM chunks WHERE doc_id = ? ORDER BY ordinal", (result.doc_id,)
        ).fetchall()
    ]
    assert sections[0] == "打印机使用手册 / 耗材更换"
    assert all(section is not None for section in sections)


def test_reingesting_unchanged_file_is_skipped(conn, small_chunk_settings, manual_path):
    first = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=HashEmbedder()
    )
    second = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=HashEmbedder()
    )

    assert second.status == STATUS_SKIPPED
    assert second.doc_id == first.doc_id
    assert knowledge_stats(conn)["chunks"] == first.chunks


def test_force_reingest_replaces_old_rows(conn, small_chunk_settings, manual_path):
    first = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=HashEmbedder()
    )
    forced = ingest_file(
        conn,
        manual_path,
        settings=small_chunk_settings,
        embedder=HashEmbedder(),
        force=True,
    )

    assert forced.status == STATUS_INGESTED
    stats = knowledge_stats(conn)
    assert stats["documents"] == 1
    assert stats["chunks"] == forced.chunks == first.chunks
    assert stats["vectors"] == forced.chunks


def test_ingest_without_embedder_degrades_to_keyword_only(conn, small_chunk_settings, manual_path):
    result = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=None
    )

    assert result.status == STATUS_INGESTED
    assert result.mode == MODE_KEYWORD_ONLY
    assert result.vectors == 0
    assert knowledge_stats(conn)["vectors"] == 0

    row = conn.execute(
        "SELECT error FROM documents WHERE id = ?", (result.doc_id,)
    ).fetchone()
    assert "仅关键词" in str(row["error"])

    # 检索依旧可用（纯关键词）
    outcome = hybrid_retrieve(
        conn, "硒鼓怎么更换", embedder=None, settings=small_chunk_settings
    )
    assert outcome.mode == MODE_KEYWORD_ONLY
    assert outcome.hit is True


def test_broken_embedder_records_reason(conn, small_chunk_settings, manual_path):
    result = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=BrokenEmbedder()
    )

    assert result.status == STATUS_INGESTED
    assert result.mode == MODE_KEYWORD_ONLY
    assert "模拟嵌入不可用" in result.message


def test_unsupported_and_empty_files_fail_gracefully(conn, small_chunk_settings, work_dir):
    unsupported = write_file(work_dir, "图纸.docx", "内容")
    empty = write_file(work_dir, "空文档.md", "")

    unsupported_result = ingest_file(
        conn, unsupported, settings=small_chunk_settings, embedder=HashEmbedder()
    )
    empty_result = ingest_file(
        conn, empty, settings=small_chunk_settings, embedder=HashEmbedder()
    )

    assert unsupported_result.status == STATUS_FAILED
    assert "不支持" in unsupported_result.message
    assert empty_result.status == STATUS_FAILED
    assert "未解析出文本" in empty_result.message
    assert knowledge_stats(conn)["documents"] == 0


def test_missing_file_fails_gracefully(conn, small_chunk_settings, work_dir):
    result = ingest_file(
        conn, work_dir / "不存在.md", settings=small_chunk_settings, embedder=HashEmbedder()
    )

    assert result.status == STATUS_FAILED
    assert "文件不存在" in result.message


def test_delete_document_removes_chunks_vectors_and_index(conn, small_chunk_settings, manual_path):
    result = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=HashEmbedder()
    )

    assert delete_document(conn, result.doc_id) is True

    stats = knowledge_stats(conn)
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["vectors"] == 0

    outcome = hybrid_retrieve(
        conn, "硒鼓怎么更换", embedder=HashEmbedder(), settings=small_chunk_settings
    )
    assert outcome.hit is False
    assert delete_document(conn, result.doc_id) is False


def test_reembed_document_after_keyword_only_ingest(conn, small_chunk_settings, manual_path):
    ingested = ingest_file(conn, manual_path, settings=small_chunk_settings, embedder=None)
    assert knowledge_stats(conn)["vectors"] == 0

    rebuilt = reembed_document(
        conn, ingested.doc_id, embedder=HashEmbedder(), settings=small_chunk_settings
    )

    assert rebuilt.vectors == rebuilt.chunks == ingested.chunks
    assert knowledge_stats(conn)["vectors"] == ingested.chunks
    # 重复重建不应产生重复向量
    reembed_document(
        conn, ingested.doc_id, embedder=HashEmbedder(), settings=small_chunk_settings
    )
    assert knowledge_stats(conn)["vectors"] == ingested.chunks


def test_ingest_directory_and_rebuild_index(conn, small_chunk_settings, work_dir):
    samples = work_dir / "docs" / "samples"
    write_file(samples, "手册.md", MANUAL)
    write_file(samples, "规则.txt", "保修期为两年。\n\n人为损坏不在保修范围内。")
    write_file(samples, "忽略.png", "not a document")

    results = ingest_directory(
        conn,
        samples,
        settings=small_chunk_settings,
        embedder=HashEmbedder(),
        source="sample",
    )

    assert [result.filename for result in results] == ["手册.md", "规则.txt"]
    assert all(result.status == STATUS_INGESTED for result in results)
    assert knowledge_stats(conn)["documents"] == 2

    rebuilt = rebuild_index(conn, settings=small_chunk_settings, embedder=HashEmbedder())

    assert len(rebuilt) == 2
    assert all(result.status == STATUS_INGESTED for result in rebuilt)
    assert knowledge_stats(conn)["documents"] == 2


def test_rebuild_index_reembeds_when_source_file_is_gone(conn, small_chunk_settings, manual_path):
    ingested = ingest_file(
        conn, manual_path, settings=small_chunk_settings, embedder=None
    )
    manual_path.unlink()

    results = rebuild_index(conn, settings=small_chunk_settings, embedder=HashEmbedder())

    assert len(results) == 1
    assert results[0].doc_id == ingested.doc_id
    assert results[0].vectors == ingested.chunks
