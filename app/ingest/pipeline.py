"""入库管线：解析 → 分块 → 嵌入 → 写库。

特性：
* **幂等**：按文件 sha256 去重，内容没变直接跳过；`force=True` 时先删旧数据再入库；
* **可降级**：嵌入模型不可用时照样入库，只是没有向量（检索退化为纯关键词），
  并把原因写进返回值与 documents.error，页面上能看到；
* **可重建**：原文还在就重新解析入库；原文丢了就只用现有片段重算向量。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import BASE_DIR, Settings, get_settings
from app.db import has_vec_table, to_vec_blob
from app.ingest.loader import (
    SUPPORTED_SUFFIXES,
    UnsupportedDocument,
    load_document,
    sha256_of,
)
from app.ingest.splitter import split_document
from app.rag.embedder import Embedder, EmbeddingUnavailable
from app.rag.tokenize import segment
from app.utils import now_iso

logger = logging.getLogger(__name__)

STATUS_INGESTED = "ingested"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

MODE_HYBRID = "hybrid"
MODE_KEYWORD_ONLY = "keyword_only"


@dataclass(frozen=True)
class IngestResult:
    filename: str
    status: str
    doc_id: int | None = None
    chunks: int = 0
    vectors: int = 0
    mode: str = MODE_HYBRID
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAILED

    def as_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "status": self.status,
            "doc_id": self.doc_id,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "mode": self.mode,
            "message": self.message,
        }


def _relative_path(path: Path, settings: Settings) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def _store_vectors(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[int],
    texts: Sequence[str],
    embedder: Embedder | None,
) -> tuple[int, str, str]:
    """写入向量，返回 (向量条数, 模式, 说明)。"""
    if embedder is None:
        return 0, MODE_KEYWORD_ONLY, "未启用嵌入模型，仅关键词检索"
    if not has_vec_table(conn):
        return 0, MODE_KEYWORD_ONLY, "向量表不可用，仅关键词检索"

    try:
        embeddings = embedder.encode(texts)
    except EmbeddingUnavailable as exc:
        return 0, MODE_KEYWORD_ONLY, f"嵌入不可用，仅关键词检索：{exc}"

    if len(embeddings) != len(chunk_ids):
        return 0, MODE_KEYWORD_ONLY, "嵌入返回数量与分块数不一致，仅关键词检索"

    conn.executemany(
        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
        [(chunk_id, to_vec_blob(vector)) for chunk_id, vector in zip(chunk_ids, embeddings)],
    )
    return len(chunk_ids), MODE_HYBRID, ""


def ingest_file(
    conn: sqlite3.Connection,
    path: Path | str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    source: str = "upload",
    force: bool = False,
) -> IngestResult:
    """把一个文件入库。"""
    settings = settings or get_settings()
    path = Path(path)

    if not path.is_file():
        return IngestResult(path.name, STATUS_FAILED, message="文件不存在")

    try:
        digest = sha256_of(path)
    except OSError as exc:
        return IngestResult(path.name, STATUS_FAILED, message=f"读取文件失败：{exc}")

    existing = conn.execute(
        "SELECT id FROM documents WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing is not None and not force:
        return IngestResult(
            path.name,
            STATUS_SKIPPED,
            doc_id=int(existing[0]),
            message="内容未变化（sha256 相同），跳过",
        )
    if existing is not None:
        delete_document(conn, int(existing[0]))

    try:
        document = load_document(path)
    except UnsupportedDocument as exc:
        return IngestResult(path.name, STATUS_FAILED, message=str(exc))
    except Exception as exc:  # noqa: BLE001 - 坏文件不应让接口崩
        logger.warning("解析失败 %s：%s", path, exc)
        return IngestResult(path.name, STATUS_FAILED, message=f"解析失败：{exc}")

    pieces = split_document(
        document, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not pieces:
        return IngestResult(
            path.name,
            STATUS_FAILED,
            message="未解析出文本（可能是扫描版 PDF，需要先做 OCR）",
        )

    conn.execute(
        """
        INSERT INTO documents(filename, rel_path, sha256, mime, page_count,
                              chunk_count, source, status, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?)
        """,
        (
            path.name,
            _relative_path(path, settings),
            digest,
            document.mime,
            document.page_count,
            len(pieces),
            source,
            now_iso(),
        ),
    )
    doc_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    chunk_ids: list[int] = []
    texts: list[str] = []
    for piece in pieces:
        conn.execute(
            """
            INSERT INTO chunks(doc_id, ordinal, section, page, text, text_tokens, char_len)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                piece.ordinal,
                piece.section,
                piece.page,
                piece.text,
                segment(piece.text),
                len(piece.text),
            ),
        )
        chunk_ids.append(int(conn.execute("SELECT last_insert_rowid()").fetchone()[0]))
        texts.append(piece.text)

    vectors, mode, message = _store_vectors(conn, chunk_ids, texts, embedder)

    conn.execute(
        "UPDATE documents SET chunk_count = ?, error = ? WHERE id = ?",
        (len(pieces), message or None, doc_id),
    )
    conn.commit()

    logger.info(
        "入库完成 %s：%d 块 / %d 向量（%s）", path.name, len(pieces), vectors, mode
    )
    return IngestResult(
        path.name,
        STATUS_INGESTED,
        doc_id=doc_id,
        chunks=len(pieces),
        vectors=vectors,
        mode=mode,
        message=message,
    )


def ingest_directory(
    conn: sqlite3.Connection,
    directory: Path | str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    source: str = "sample",
    force: bool = False,
    recursive: bool = True,
) -> list[IngestResult]:
    """把一个目录下所有受支持的文件入库（按文件名排序，结果稳定）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    files = sorted(
        candidate
        for candidate in directory.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [
        ingest_file(
            conn,
            candidate,
            settings=settings,
            embedder=embedder,
            source=source,
            force=force,
        )
        for candidate in files
    ]


def delete_document(conn: sqlite3.Connection, doc_id: int) -> bool:
    """删除文档及其分块、向量、全文索引。"""
    row = conn.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return False

    chunk_ids = [
        int(item[0])
        for item in conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
    ]
    if chunk_ids and has_vec_table(conn):
        conn.executemany(
            "DELETE FROM vec_chunks WHERE chunk_id = ?", [(cid,) for cid in chunk_ids]
        )

    # chunks 由外键级联删除，FTS5 索引由触发器同步
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    logger.info("已删除文档 %s（%d 个分块）", doc_id, len(chunk_ids))
    return True


def reembed_document(
    conn: sqlite3.Connection,
    doc_id: int,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """原文丢失或换了嵌入模型时，只用库里的片段重算向量。"""
    row = conn.execute(
        "SELECT filename FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return IngestResult(f"doc#{doc_id}", STATUS_FAILED, message="文档不存在")
    filename = str(row["filename"])

    chunks = conn.execute(
        "SELECT id, text FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
    ).fetchall()
    if not chunks:
        return IngestResult(filename, STATUS_FAILED, doc_id=doc_id, message="没有分块可重建")

    chunk_ids = [int(item["id"]) for item in chunks]
    texts = [str(item["text"]) for item in chunks]

    if has_vec_table(conn):
        conn.executemany(
            "DELETE FROM vec_chunks WHERE chunk_id = ?", [(cid,) for cid in chunk_ids]
        )

    vectors, mode, message = _store_vectors(conn, chunk_ids, texts, embedder)
    conn.execute("UPDATE documents SET error = ? WHERE id = ?", (message or None, doc_id))
    conn.commit()
    return IngestResult(
        filename,
        STATUS_INGESTED,
        doc_id=doc_id,
        chunks=len(chunk_ids),
        vectors=vectors,
        mode=mode,
        message=message,
    )


def rebuild_index(
    conn: sqlite3.Connection,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> list[IngestResult]:
    """重建全部索引：原文在就重新解析，原文不在就重算向量。"""
    settings = settings or get_settings()
    documents = conn.execute(
        "SELECT id, rel_path, filename FROM documents ORDER BY id"
    ).fetchall()

    results: list[IngestResult] = []
    for document in documents:
        doc_id = int(document["id"])
        rel_path = document["rel_path"]
        path = (BASE_DIR / rel_path) if rel_path else None
        if path is not None and path.is_file():
            results.append(
                ingest_file(
                    conn,
                    path,
                    settings=settings,
                    embedder=embedder,
                    source="rebuild",
                    force=True,
                )
            )
        else:
            results.append(
                reembed_document(conn, doc_id, embedder=embedder, settings=settings)
            )
    return results
