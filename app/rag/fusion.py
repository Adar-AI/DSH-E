"""多路召回的融合（RRF）与支撑集选择（动态 TopK）。

数据流：各召回通道各自排好序 → `rrf_fuse` 融合成一个候选池 →
（重排器打分）→ `select_supporting` 用三重约束挑出最终支撑片段。

为什么要 RRF：不同通道的分数量纲完全不同（BM25 负分、余弦 0~1、网页搜索无分数），
只有"排名"是可比的；RRF 用 `Σ weight/(k+rank)` 把排名融成分数，再归一化到 0~1，
使得"没有重排模型时"这套分数也能直接套用同一组阈值。

动态 TopK 的三重约束（对应需求 F17）：
1. **绝对下限**：低于 `min_score` 的一律不要（资料不相关就别硬塞进提示词）；
2. **相对地板**：低于 `top1 × relative_floor` 的不要（同批候选里明显拉胯的）；
3. **断崖截断**：自第 2 条起，若分数突然掉到上一条的 `cliff_ratio` 以下，立即停止。
再加一个条数硬上限，防止提示词被塞爆。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

RRF_K = 60  # 与 app/rag/retriever.py 保持一致

SOURCE_INTERNAL = "internal"
SOURCE_WEB = "web"

DEFAULT_SOURCE_WEIGHTS: Mapping[str, float] = {
    SOURCE_INTERNAL: 1.0,
    SOURCE_WEB: 0.6,  # 联网只做补充，不允许压过内部资料
}


@dataclass
class Candidate:
    """融合前的一条候选：内部知识片段或联网搜索结果。"""

    key: str
    text: str
    source: str = SOURCE_INTERNAL
    citation: str = ""
    url: str = ""
    title: str = ""
    hostname: str = ""
    chunk_id: int | None = None
    doc_id: int | None = None
    filename: str = ""
    section: str | None = None
    page: int | None = None
    base_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    rrf: float = 0.0
    rrf_norm: float = 0.0
    rerank_score: float | None = None

    # ------------------------------------------------------------------ 分数

    @property
    def score(self) -> float:
        """当前可用分数：有精排用精排，否则用归一化融合分。"""
        return self.rerank_score if self.rerank_score is not None else self.rrf_norm

    @property
    def is_web(self) -> bool:
        return self.source == SOURCE_WEB

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "source": self.source,
            "citation": self.citation,
            "url": self.url,
            "title": self.title,
            "hostname": self.hostname,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "section": self.section,
            "page": self.page,
            "score": round(self.score, 4),
            "rrf": round(self.rrf, 6),
            "rrf_norm": round(self.rrf_norm, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "ranks": dict(self.ranks),
        }

    # -------------------------------------------------------------- 构造入口

    @classmethod
    def from_retrieved(cls, chunk: Mapping[str, Any], *, channel: str) -> "Candidate":
        """由检索结果（`RetrievedChunk.as_dict()`）构造。"""
        chunk_id = int(chunk.get("chunk_id") or 0)
        return cls(
            key=f"chunk:{chunk_id}",
            text=str(chunk.get("text") or ""),
            source=SOURCE_INTERNAL,
            citation=str(chunk.get("citation") or chunk.get("filename") or ""),
            chunk_id=chunk_id,
            doc_id=int(chunk.get("doc_id") or 0) or None,
            filename=str(chunk.get("filename") or ""),
            section=chunk.get("section"),
            page=chunk.get("page"),
            base_score=float(chunk.get("score") or 0.0),
            ranks={channel: 0},
        )

    @classmethod
    def from_web(cls, item: Mapping[str, Any], *, channel: str = "web") -> "Candidate":
        """由联网搜索结果（MCP `search_pro` 的一条 page）构造。"""
        url = str(item.get("url") or "")
        hostname = str(item.get("hostname") or "")
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        citation = " · ".join(part for part in (title, hostname) if part) or url
        return cls(
            key=f"web:{url or title or snippet[:32]}",
            text=snippet,
            source=SOURCE_WEB,
            citation=citation,
            url=url,
            title=title,
            hostname=hostname,
            ranks={channel: 0},
        )


def rrf_fuse(
    channels: Mapping[str, Sequence[Candidate]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
) -> list[Candidate]:
    """多路 RRF 融合。

    `channels` 形如 `{"internal:q1": [候选...], "web": [候选...]}`，**每个通道内部必须已按相关性降序**。
    同一条候选命中多个通道时分数相加（`ranks` 记录各通道名次），
    这样"多路都排前面"的结果会自然浮到最前。
    """
    effective = dict(DEFAULT_SOURCE_WEIGHTS)
    if weights:
        effective.update(weights)

    def weight_of(name: str, sample: Candidate | None) -> float:
        # 通道自带来源权重：channel 名形如 "internal:q1" / "web:q2"
        if name in effective:
            return float(effective[name])
        if name.split(":", 1)[0] in effective:
            return float(effective[name.split(":", 1)[0]])
        if sample is not None:
            return float(effective.get(sample.source, 1.0))
        return 1.0

    merged: dict[str, Candidate] = {}
    max_possible = 0.0
    for name, items in channels.items():
        sample = items[0] if items else None
        max_possible += weight_of(name, sample) / (k + 1)
        for rank, candidate in enumerate(items, start=1):
            weight = weight_of(name, candidate)
            existing = merged.get(candidate.key)
            if existing is None:
                candidate.ranks = {name: rank}
                candidate.rrf = weight / (k + rank)
                merged[candidate.key] = candidate
                continue

            existing.ranks[name] = rank
            existing.rrf += weight / (k + rank)
            # 内部片段优先保留更完整的信息
            if existing.is_web and not candidate.is_web:
                existing.text = candidate.text or existing.text
                existing.citation = candidate.citation or existing.citation
                existing.filename = candidate.filename or existing.filename
                existing.section = candidate.section or existing.section
                existing.page = candidate.page or existing.page
                existing.chunk_id = candidate.chunk_id or existing.chunk_id

    result = list(merged.values())
    for candidate in result:
        candidate.rrf_norm = (
            min(candidate.rrf / max_possible, 1.0) if max_possible > 0 else 0.0
        )
    result.sort(key=lambda item: (-item.score, item.key))
    return result


def apply_rerank_scores(
    candidates: Sequence[Candidate], scores: Sequence[float]
) -> list[Candidate]:
    """把精排分数写到候选上并重新排序（返回新列表，元素与原对象同一份）。"""
    if len(scores) != len(candidates):
        raise ValueError(
            f"精排分数数量与候选数量不一致：{len(scores)} vs {len(candidates)}"
        )
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(candidates, key=lambda item: (-item.score, item.key))


@dataclass(frozen=True)
class SelectionResult:
    """动态 TopK 的结果与依据（进 trace，便于解释"为什么只用了这几条"）。"""

    kept: tuple[Candidate, ...]
    dropped: int = 0
    top1: float = 0.0
    floor: float = 0.0
    cutoff: str = "none"  # floor | relative | cliff | cap | none
    reason: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.kept

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": len(self.kept),
            "dropped": self.dropped,
            "top1": round(self.top1, 4),
            "floor": round(self.floor, 4),
            "cutoff": self.cutoff,
            "reason": self.reason,
        }


def select_supporting(
    candidates: Sequence[Candidate],
    *,
    min_score: float = 0.10,
    relative_floor: float = 0.35,
    cliff_ratio: float = 0.5,
    top_k_max: int = 6,
) -> SelectionResult:
    """动态 TopK：挑出最终进入提示词的支撑片段。"""
    ordered = sorted(candidates, key=lambda item: (-item.score, item.key))
    if not ordered:
        return SelectionResult((), 0, 0.0, float(min_score), "none", "没有候选片段")

    top1 = ordered[0].score
    floor = max(float(min_score), top1 * float(relative_floor))

    kept: list[Candidate] = []
    cutoff = "none"
    for index, candidate in enumerate(ordered):
        if candidate.score < float(min_score):
            cutoff = "floor"
            break
        if index > 0 and candidate.score < floor:
            cutoff = "relative"
            break
        if kept and candidate.score < kept[-1].score * float(cliff_ratio):
            cutoff = "cliff"
            break
        if len(kept) >= int(top_k_max):
            cutoff = "cap"
            break
        kept.append(candidate)

    kept_count = len(kept)
    reason = (
        f"top1={top1:.3f} 门槛={floor:.3f} 保留 {kept_count}/{len(ordered)} 条"
        f"（截断：{cutoff}）"
    )
    return SelectionResult(
        kept=tuple(kept),
        dropped=len(ordered) - kept_count,
        top1=top1,
        floor=floor,
        cutoff=cutoff,
        reason=reason,
    )


def summarize_sources(candidates: Iterable[Candidate]) -> dict[str, int]:
    """统计候选里的来源分布（进 trace）。"""
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.source] = counts.get(candidate.source, 0) + 1
    return counts


__all__ = [
    "Candidate",
    "DEFAULT_SOURCE_WEIGHTS",
    "RRF_K",
    "SOURCE_INTERNAL",
    "SOURCE_WEB",
    "SelectionResult",
    "apply_rerank_scores",
    "rrf_fuse",
    "select_supporting",
    "summarize_sources",
]
