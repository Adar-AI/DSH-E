"""重排（Rerank）：把粗召回的候选按"与问题的真实相关性"重新排序。

两种实现，接口一致：

* `CrossEncoderReranker`：本地 `bge-reranker-large`（sentence-transformers CrossEncoder），
  输出 sigmoid 相关性分（0~1）。实测分离度极好：命中片段 0.97、相关但不同章节 0.38、
  无关 ≤0.035 —— 这正是三档置信阈值能直接落在 0.50 / 0.10 的依据；
* `LexicalFallbackReranker`：权重缺失或加载失败时的兜底，用**查询词覆盖率**（0~1）当分数。
  保证"重排"这一步永远可用，不会因为少一个模型就让整条链路中断（见架构说明 13.4 降级矩阵）。

模型体积 2.1GB、CPU 加载约 6.5s，因此**懒加载**：第一次真正需要重排时才载入，
进程内只加载一次；可用 `WARMUP_RERANKER=true` 让服务启动时后台预热。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from app.config import Settings, get_settings
from app.rag.tokenize import segment, tokens_of

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[tuple[str, str, int, int], "Reranker"] = {}
_CACHE_LOCK = threading.Lock()


class RerankerUnavailable(RuntimeError):
    """重排模型不可用（路径缺失、依赖缺失、加载失败）。"""


@runtime_checkable
class Reranker(Protocol):
    """重排器协议：测试里用假实现替换。"""

    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...

    def health(self) -> tuple[bool, str]:
        ...


class CrossEncoderReranker:
    """基于 sentence-transformers CrossEncoder 的本地精排。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = f"cross-encoder({Path(self.model_path).name})"
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 内部

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:  # pragma: no cover - 依赖缺失分支
                raise RerankerUnavailable("未安装 sentence-transformers") from exc

            if not Path(self.model_path).exists():
                raise RerankerUnavailable(f"重排模型路径不存在：{self.model_path}")

            try:
                model = CrossEncoder(
                    self.model_path, device=self.device, max_length=self.max_length
                )
            except Exception as exc:  # noqa: BLE001 - 任何加载失败都要能降级
                raise RerankerUnavailable(f"加载重排模型失败：{exc}") from exc

            self._model = model
            logger.info("重排模型已加载：%s（device=%s）", self.model_path, self.device)
            return self._model

    # ------------------------------------------------------------------ 对外

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        items = [text if text else " " for text in texts]
        if not items:
            return []

        model = self._ensure_model()
        try:
            raw = model.predict(
                [(query, text) for text in items],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - 推理失败同样降级
            raise RerankerUnavailable(f"重排推理失败：{exc}") from exc

        return [float(value) for value in raw]

    def health(self) -> tuple[bool, str]:
        try:
            self._ensure_model()
        except RerankerUnavailable as exc:
            return False, str(exc)
        return True, f"{self.name} 就绪（device={self.device}）"


class LexicalFallbackReranker:
    """兜底重排：用查询词覆盖率打分（0~1），无需任何模型。"""

    name = "lexical-coverage"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        tokens = tokens_of(query)
        if not tokens:
            return [0.0 for _ in texts]

        scores: list[float] = []
        for text in texts:
            indexed = set(segment(text or "").split())
            hit = sum(1 for token in tokens if token in indexed)
            scores.append(hit / len(tokens))
        return scores

    def health(self) -> tuple[bool, str]:
        return True, f"{self.name}（词面覆盖率兜底，无需模型）"


def get_reranker(settings: Settings | None = None) -> Reranker:
    """按配置返回缓存的重排器；模型不可用时自动降级为词面覆盖率。"""
    settings = settings or get_settings()
    key = (
        str(settings.rerank_model_path),
        settings.rerank_device,
        settings.rerank_batch_size,
        settings.rerank_max_length,
    )

    with _CACHE_LOCK:
        cached = _RERANKER_CACHE.get(key)
        if cached is not None:
            return cached

        if settings.rerank_model_path:
            candidate: Reranker = CrossEncoderReranker(
                settings.rerank_model_path,
                device=settings.rerank_device,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
            )
            healthy, detail = candidate.health()
            if healthy:
                cached = candidate
            else:
                logger.warning("重排模型不可用，降级为词面覆盖率兜底：%s", detail)
                cached = LexicalFallbackReranker()
        else:
            cached = LexicalFallbackReranker()

        _RERANKER_CACHE[key] = cached
        return cached


def clear_reranker_cache() -> None:
    """测试用：清空缓存。"""
    with _CACHE_LOCK:
        _RERANKER_CACHE.clear()


__all__ = [
    "CrossEncoderReranker",
    "LexicalFallbackReranker",
    "Reranker",
    "RerankerUnavailable",
    "clear_reranker_cache",
    "get_reranker",
]
