"""混合检索：FTS5 关键词 + sqlite-vec 向量 + RRF 融合 + 词汇覆盖率打分。

打分设计（三个信号都归一到 0~1，缺哪个通道该项为 0）：

    score = 0.45 · 覆盖率 + 0.35 · 向量相似度 + 0.20 · RRF 排名分

* **覆盖率**：问题分词后有多少 token 出现在片段里。中文按二元切分后，
  「打印机报错E05怎么处理」这类问题会命中「打印/印机/机报/报错/e05」，
  与问题无关的资料覆盖不了几个词，天然被压分；
* **向量相似度**：归一化向量的 L2 距离换算成余弦，再用
  `[VEC_FLOOR, VEC_CEIL]` 线性拉伸。BGE-M3 的不相关文本相似度普遍在 0.35~0.45，
  直接当分数用会把无关片段抬得太高，所以设了地板；
* **RRF**：Reciprocal Rank Fusion，把两路的排名融合成与量纲无关的分值，再按
  通道数归一，避免只有一路时满分被稀释。

当最高分低于阈值（默认 0.35）时判定为「未命中」，由上层走兜底话术而不是让模型
凭记忆编答案——这是需求 F5 的验收点。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Sequence

from app.config import Settings, get_settings
from app.db import has_vec_table, knn_search
from app.rag.embedder import Embedder, EmbeddingUnavailable
from app.rag.tokenize import build_match_query, tokens_of

logger = logging.getLogger(__name__)

RRF_K = 60
W_COVERAGE = 0.45
W_VECTOR = 0.35
W_RRF = 0.20
VEC_FLOOR = 0.35
VEC_CEIL = 0.85

MODE_HYBRID = "hybrid"
MODE_KEYWORD_ONLY = "keyword_only"


@dataclass(frozen=True)
class RetrievedChunk:
    """一条召回片段（含打分明细，便于页面展示「本轮依据」）。"""

    chunk_id: int
    doc_id: int
    filename: str
    section: str | None
    page: int | None
    text: str
    score: float
    coverage: float
    vector_sim: float
    rrf: float
    keyword_rank: int | None = None
    vector_rank: int | None = None

    @property
    def citation(self) -> str:
        parts = [self.filename]
        if self.page:
            parts.append(f"第 {self.page} 页")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "section": self.section,
            "page": self.page,
            "text": self.text,
            "score": self.score,
            "coverage": self.coverage,
            "vector_sim": self.vector_sim,
            "rrf": self.rrf,
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class RetrievalOutcome:
    """一次检索的完整结果：命中片段 + 是否达标 + 使用的模式。"""

    chunks: tuple[RetrievedChunk, ...]
    hit: bool
    best_score: float
    mode: str
    reason: str = ""
    keyword_candidates: int = 0
    vector_candidates: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "hit": self.hit,
            "best_score": self.best_score,
            "mode": self.mode,
            "reason": self.reason,
            "keyword_candidates": self.keyword_candidates,
            "vector_candidates": self.vector_candidates,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }


# ------------------------------------------------------------------ 单路检索

def keyword_search(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[int, float]]:
    """FTS5/BM25 检索，返回 [(chunk_id, bm25 分值), ...]（分值越小越相关）。"""
    expression = build_match_query(query)
    if not expression:
        return []

    sql = """
        SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts) AS rank_score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank_score
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, (expression, int(limit))).fetchall()
    except sqlite3.OperationalError as exc:  # 例如 FTS 表达式非法
        logger.warning("关键词检索失败：%s", exc)
        return []
    return [(int(row[0]), float(row[1])) for row in rows]


def vector_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder | None,
    limit: int,
) -> tuple[list[tuple[int, float]], str | None]:
    """向量检索，返回 ([(chunk_id, 余弦相似度)], 失败原因)。"""
    if embedder is None:
        return [], "未配置嵌入模型"
    if not has_vec_table(conn):
        return [], "向量表不可用"

    try:
        vectors = embedder.encode([query])
    except EmbeddingUnavailable as exc:
        return [], str(exc)
    if not vectors:
        return [], "嵌入结果为空"

    hits = knn_search(conn, vectors[0], limit=limit)
    results: list[tuple[int, float]] = []
    for chunk_id, distance in hits:
        # 归一化向量的 L2 距离 -> 余弦相似度：d² = 2 - 2cos
        cosine = 1.0 - (distance**2) / 2.0
        results.append((chunk_id, max(min(cosine, 1.0), -1.0)))
    return results, None


# ------------------------------------------------------------------ 融合打分

def _fetch_chunks(conn: sqlite3.Connection, chunk_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    sql = f"""
        SELECT c.id, c.doc_id, c.section, c.page, c.text, c.text_tokens, d.filename
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE c.id IN ({placeholders}) AND d.status = 'ready'
    """
    rows = conn.execute(sql, list(chunk_ids)).fetchall()
    return {int(row["id"]): row for row in rows}


def _coverage(query_tokens: Sequence[str], text_tokens: str | None) -> float:
    if not query_tokens:
        return 0.0
    indexed = set((text_tokens or "").split())
    if not indexed:
        return 0.0
    matched = sum(1 for token in query_tokens if token in indexed)
    return matched / len(query_tokens)


def _vector_normalized(similarity: float) -> float:
    if similarity <= VEC_FLOOR:
        return 0.0
    return min((similarity - VEC_FLOOR) / (VEC_CEIL - VEC_FLOOR), 1.0)


def hybrid_retrieve(
    conn: sqlite3.Connection,
    query: str,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
    candidates: int | None = None,
    min_score: float | None = None,
) -> RetrievalOutcome:
    """混合检索主入口。embedder 为 None 或不可用时自动退化为纯关键词。"""
    settings = settings or get_settings()
    top_k = int(top_k or settings.retrieve_top_k)
    candidates = int(candidates or settings.retrieve_candidates)
    threshold = float(settings.retrieve_min_score if min_score is None else min_score)

    query = (query or "").strip()
    if not query:
        return RetrievalOutcome((), False, 0.0, MODE_KEYWORD_ONLY, "问题为空")

    keyword_hits = keyword_search(conn, query, candidates)
    vector_hits, vector_error = vector_search(conn, query, embedder, candidates)

    if vector_hits:
        mode, reason = MODE_HYBRID, ""
    else:
        mode = MODE_KEYWORD_ONLY
        reason = vector_error or "向量通道无结果"

    ordered_ids = list(dict.fromkeys([cid for cid, _ in keyword_hits] + [cid for cid, _ in vector_hits]))
    rows = _fetch_chunks(conn, ordered_ids)
    query_tokens = tokens_of(query)

    keyword_rank = {cid: index + 1 for index, (cid, _) in enumerate(keyword_hits)}
    vector_rank = {cid: index + 1 for index, (cid, _) in enumerate(vector_hits)}
    vector_similarity = dict(vector_hits)

    channels = (1 if keyword_hits else 0) + (1 if vector_hits else 0)
    rrf_ceiling = (channels / (RRF_K + 1)) if channels else 1.0

    scored: list[RetrievedChunk] = []
    for chunk_id in ordered_ids:
        row = rows.get(chunk_id)
        if row is None:
            continue

        raw_rrf = 0.0
        if chunk_id in keyword_rank:
            raw_rrf += 1.0 / (RRF_K + keyword_rank[chunk_id])
        if chunk_id in vector_rank:
            raw_rrf += 1.0 / (RRF_K + vector_rank[chunk_id])
        rrf = min(raw_rrf / rrf_ceiling, 1.0) if rrf_ceiling else 0.0

        coverage = _coverage(query_tokens, row["text_tokens"])
        similarity = float(vector_similarity.get(chunk_id, 0.0))
        score = (
            W_COVERAGE * coverage
            + W_VECTOR * _vector_normalized(similarity)
            + W_RRF * rrf
        )

        scored.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=int(row["doc_id"]),
                filename=str(row["filename"]),
                section=row["section"],
                page=row["page"],
                text=str(row["text"]),
                score=round(score, 4),
                coverage=round(coverage, 4),
                vector_sim=round(similarity, 4),
                rrf=round(rrf, 4),
                keyword_rank=keyword_rank.get(chunk_id),
                vector_rank=vector_rank.get(chunk_id),
            )
        )

    scored.sort(key=lambda item: (-item.score, item.chunk_id))
    top = tuple(scored[:top_k])
    best = top[0].score if top else 0.0
    hit = bool(top) and best >= threshold

    return RetrievalOutcome(
        chunks=top,
        hit=hit,
        best_score=round(best, 4),
        mode=mode,
        reason=reason if not hit else "",
        keyword_candidates=len(keyword_hits),
        vector_candidates=len(vector_hits),
    )
