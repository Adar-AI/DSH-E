"""本地嵌入模型封装（BGE-M3，1024 维）。

要点：
* **懒加载**：进程启动不加载 4GB 模型，第一次 `encode()` 时才加载，避免拖慢启动；
* **可降级**：依赖缺失、模型路径错误、显存不足等一律抛 `EmbeddingUnavailable`，
  由上层（入库 / 检索）捕获后切换成纯关键词模式，绝不因此让接口 500；
* **归一化**：输出 L2 归一化向量，点积即余弦相似度。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_EMBEDDER_CACHE: dict[tuple[str, str], "Embedder"] = {}
_CACHE_LOCK = threading.Lock()


class EmbeddingUnavailable(RuntimeError):
    """嵌入模型不可用。"""


@runtime_checkable
class Embedder(Protocol):
    """嵌入模型协议：测试里用假实现替换。"""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class BGEEmbedder:
    """基于 sentence-transformers 的本地 BGE-M3 嵌入。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        dim: int = 1024,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = batch_size
        self.dim = dim
        self.name = f"bge-m3({Path(self.model_path).name},{device})"
        self._model = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- 内部

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover - 依赖缺失时的分支
                raise EmbeddingUnavailable("未安装 sentence-transformers") from exc

            if not Path(self.model_path).exists():
                raise EmbeddingUnavailable(f"嵌入模型路径不存在：{self.model_path}")

            try:
                model = SentenceTransformer(self.model_path, device=self.device)
            except Exception as exc:  # noqa: BLE001 - 任何加载失败都要能降级
                raise EmbeddingUnavailable(f"加载嵌入模型失败：{exc}") from exc

            # 记录真实维度（transformers 5.x 改了方法名，两种都兼容）
            getter = getattr(model, "get_embedding_dimension", None) or getattr(
                model, "get_sentence_embedding_dimension", None
            )
            if callable(getter):
                try:
                    detected = int(getter())
                    if detected > 0:
                        self.dim = detected
                except Exception:  # noqa: BLE001 - 维度探测失败不影响使用
                    pass

            self._model = model
            logger.info("嵌入模型已加载：%s（dim=%s, device=%s）", self.model_path, self.dim, self.device)
            return self._model

    # -------------------------------------------------------------- 对外

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        items = [text if text else " " for text in texts]
        if not items:
            return []

        model = self._ensure_model()
        try:
            matrix = model.encode(
                items,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001 - 推理失败同样降级
            raise EmbeddingUnavailable(f"嵌入推理失败：{exc}") from exc

        return [[float(value) for value in row] for row in matrix]

    def health(self) -> tuple[bool, str]:
        try:
            self._ensure_model()
        except EmbeddingUnavailable as exc:
            return False, str(exc)
        return True, f"{self.name} 就绪（dim={self.dim}）"


def get_embedder(settings: Settings | None = None) -> Embedder:
    """按配置返回缓存的嵌入器（同一进程内模型只加载一次）。"""
    settings = settings or get_settings()
    key = (str(settings.embedding_model_path), settings.embedding_device)
    with _CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is None:
            cached = BGEEmbedder(
                settings.embedding_model_path,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                dim=settings.embedding_dim,
            )
            _EMBEDDER_CACHE[key] = cached
        return cached


def clear_embedder_cache() -> None:
    """测试用：清空缓存。"""
    with _CACHE_LOCK:
        _EMBEDDER_CACHE.clear()
