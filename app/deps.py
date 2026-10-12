"""进程级依赖装配：模型客户端缓存、checkpointer、启动预热。

Web 层每次请求调用 `build_deps(conn)` 拿到一份绑定了本次请求数据库连接的依赖；
模型客户端、嵌入模型、checkpointer 都是进程级单例（构造便宜、加载昂贵）。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from app.db import connect as db_connect
from app.llm.providers import (
    ChatChain,
    OpenAICompatProvider,
    build_chat_chain,
    build_vision_provider,
)
from app.rag.embedder import Embedder, EmbeddingUnavailable, get_embedder
from app.store import ConversationStore

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], None]


def _resolve_stream_emitter() -> EmitFn | None:
    """拿到 LangGraph 的自定义流写入器（必须在图运行期间调用）。"""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - 不在图运行上下文里（例如单测直接调节点）
        return None

    if writer is None:
        return None

    def emit(payload: dict[str, Any]) -> None:
        try:
            writer(payload)
        except Exception:  # noqa: BLE001 - 推送失败不应影响业务
            logger.debug("自定义流事件推送失败", exc_info=True)

    return emit


@dataclass
class GraphDeps:
    """一次请求内的全部依赖。"""

    settings: Settings
    conn: sqlite3.Connection
    store: ConversationStore
    chat_chain: ChatChain
    embedder: Embedder | None = None
    vision_provider: OpenAICompatProvider | None = None
    emit: EmitFn | None = None
    _recorded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def emitter(self) -> EmitFn:
        """事件推送器：优先用注入的（测试），否则用 LangGraph 流写入器，再否则丢弃。"""
        if self.emit is not None:
            return self.emit
        writer = _resolve_stream_emitter()
        if writer is not None:
            return writer
        return lambda payload: None

    def record(self, payload: dict[str, Any]) -> None:
        """测试辅助：把事件记进列表，便于断言。"""
        self._recorded.append(payload)

    @property
    def recorded(self) -> list[dict[str, Any]]:
        return self._recorded


# ------------------------------------------------------------------ 单例缓存

_LOCK = threading.Lock()
_CHAIN: ChatChain | None = None
_VISION: OpenAICompatProvider | None = None
_CHECKPOINTERS: dict[str, Any] = {}


def get_chat_chain(settings: Settings | None = None) -> ChatChain:
    global _CHAIN
    settings = settings or get_settings()
    with _LOCK:
        if _CHAIN is None:
            _CHAIN = build_chat_chain(settings)
        return _CHAIN


def get_vision_provider(settings: Settings | None = None) -> OpenAICompatProvider:
    global _VISION
    settings = settings or get_settings()
    with _LOCK:
        if _VISION is None:
            _VISION = build_vision_provider(settings)
        return _VISION


def get_checkpointer(settings: Settings | None = None) -> Any:
    """按数据库路径缓存的 SqliteSaver（LangGraph 的持久化记忆）。"""
    settings = settings or get_settings()
    key = str(settings.db_file)
    with _LOCK:
        saver = _CHECKPOINTERS.get(key)
        if saver is not None:
            return saver

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = db_connect(settings=settings, check_same_thread=False)
        saver = SqliteSaver(conn)
        setup = getattr(saver, "setup", None)
        if callable(setup):
            setup()
        _CHECKPOINTERS[key] = saver
        logger.info("LangGraph checkpointer 就绪：%s", key)
        return saver


def build_deps(
    conn: sqlite3.Connection,
    settings: Settings | None = None,
    *,
    chat_chain: ChatChain | None = None,
    embedder: Embedder | None = None,
    vision_provider: OpenAICompatProvider | None = None,
    emit: EmitFn | None = None,
) -> GraphDeps:
    settings = settings or get_settings()
    return GraphDeps(
        settings=settings,
        conn=conn,
        store=ConversationStore(conn),
        chat_chain=chat_chain or get_chat_chain(settings),
        embedder=embedder if embedder is not None else get_embedder(settings),
        vision_provider=vision_provider or get_vision_provider(settings),
        emit=emit,
    )


def reset_caches() -> None:
    """测试用：清空进程级单例，并关闭 checkpointer 的连接（否则临时目录删不掉）。"""
    global _CHAIN, _VISION
    with _LOCK:
        _CHAIN = None
        _VISION = None
        for saver in _CHECKPOINTERS.values():
            conn = getattr(saver, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响测试
                    pass
        _CHECKPOINTERS.clear()


# ---------------------------------------------------------------------- 预热

def warmup(settings: Settings | None = None) -> dict[str, str]:
    """预热对话模型与嵌入模型，把冷启动耗时从「用户第一次提问」里挪走。

    实测：本地 qwen2.5 冷启动一次要 70s+，预热后单轮约 7s。
    """
    settings = settings or get_settings()
    report: dict[str, str] = {}

    chain = get_chat_chain(settings)
    for provider in chain.providers:
        ok, detail = provider.ping()
        report[provider.name] = "ok" if ok else detail
        if ok:
            break

    embedder = get_embedder(settings)
    try:
        embedder.encode(["预热"])
        report["embedding"] = "ok"
    except EmbeddingUnavailable as exc:
        report["embedding"] = str(exc)

    logger.info("预热完成：%s", report)
    return report


def start_warmup(settings: Settings | None = None) -> threading.Thread:
    """后台线程预热，不阻塞服务启动。"""

    def run() -> None:
        try:
            warmup(settings)
        except Exception:  # noqa: BLE001 - 预热失败不能影响服务
            logger.warning("预热失败（不影响服务启动）", exc_info=True)

    thread = threading.Thread(target=run, name="warmup", daemon=True)
    thread.start()
    return thread


__all__ = [
    "GraphDeps",
    "build_deps",
    "get_chat_chain",
    "get_checkpointer",
    "get_vision_provider",
    "reset_caches",
    "start_warmup",
    "warmup",
]
