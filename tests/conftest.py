"""测试夹具。

注意：刻意不使用 pytest 的 `tmp_path` / `tmpdir` 夹具——它们底层用
`tempfile.mkdtemp()`，会把目录 chmod 成 0700，在 DSH 沙箱里当前令牌对该目录
没有访问权（写文件直接 PermissionError）。这里改成在项目内 `data/test_tmp/`
下自建目录（普通权限），用完手动清理。
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

from app import db as dbmod
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
TEST_TMP_ROOT = ROOT / "data" / "test_tmp"

# 让测试文件可以直接 `from fakes import ...`
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture()
def work_dir() -> Path:
    """一个位于工作区内、用完即删的临时目录。"""
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"t-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def settings(work_dir: Path) -> Settings:
    """隔离到临时目录的配置：不读 .env，不碰真实的 data/app.db。

    `embedding_dim` 取 64，与测试用假嵌入器（HashEmbedder）的维度一致；
    真实的 BGE-M3（1024 维）由 tests/test_embedder.py 单独验证。
    """
    return Settings(
        _env_file=None,
        db_path=work_dir / "app.db",
        upload_dir=work_dir / "uploads",
        docs_dir=work_dir / "docs",
        embedding_dim=64,
    )


@pytest.fixture()
def small_chunk_settings(settings: Settings) -> Settings:
    """小块配置 + 打开联网通道开关（测试都注入假 web_search，不会真的联网）。

    为什么要给 MCP 配上假地址/假密钥：`should_search()` 会先检查通道是否可用，
    没配就永远不会触发联网，联网相关的分支也就测不到。
    """
    return settings.model_copy(
        update={
            "chunk_size": 60,
            "chunk_overlap": 10,
            "mcp_enabled": True,
            "mcp_base_url": "https://example.com/mcp",
            # 必须是短占位符：仓库有密钥扫描测试，`sk-` 后超过 16 位会被判为疑似真实密钥
            "mcp_api_key": "sk-fake-test",
        }
    )


@pytest.fixture()
def conn(settings: Settings):
    """已建表的临时数据库连接。"""
    connection = dbmod.connect(settings=settings)
    dbmod.init_db(connection, settings=settings)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def api_env(work_dir: Path, monkeypatch):
    """把配置指向临时目录（含假模型依赖所需的 64 维嵌入）。"""
    from app.config import get_settings
    from app.deps import reset_caches

    monkeypatch.setenv("DB_PATH", str(work_dir / "api.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(work_dir / "uploads"))
    monkeypatch.setenv("DOCS_DIR", str(work_dir / "docs"))
    monkeypatch.setenv("EMBEDDING_DIM", "64")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("RETRIEVE_TOP_K", "4")
    monkeypatch.setenv("RETRIEVE_MIN_SCORE", "0.35")
    monkeypatch.setenv("CHUNK_SIZE", "60")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("WARMUP_ON_START", "false")
    monkeypatch.setenv("FALLBACK_LLM_PROVIDER", "none")
    monkeypatch.setenv("HISTORY_TURNS", "4")
    # v2.1：接口测试保持离线与快速——不加载 2.1GB 重排模型、不联网
    monkeypatch.setenv("RERANK_MODEL_PATH", "")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("MCP_BASE_URL", "")
    monkeypatch.setenv("MCP_API_KEY", "")

    get_settings.cache_clear()
    reset_caches()
    settings = get_settings()
    settings.ensure_dirs()
    connection = dbmod.connect(settings=settings)
    try:
        dbmod.init_db(connection, settings=settings)
    finally:
        connection.close()

    try:
        yield settings
    finally:
        get_settings.cache_clear()
        reset_caches()


@pytest.fixture()
def events() -> list[dict]:
    """节点事件收集器（模拟 SSE 的 custom 流）。"""
    return []


@pytest.fixture()
def make_deps(conn, small_chunk_settings: Settings, events: list[dict]):
    """构造 GraphDeps 的工厂：所有外部依赖都可替换。

    默认注入 `StubReranker`（可控假重排器）——否则候选只能拿排名分，
    无法体现「重排后拉开差距再由动态 TopK 取舍」的行为。
    """
    from app.deps import GraphDeps
    from app.store import ConversationStore
    from fakes import FakeChatChain, StubReranker

    def factory(
        *,
        chat_chain=None,
        embedder=None,
        vision_provider=None,
        reranker="default",
        web_search=None,
        settings: Settings | None = None,
    ) -> GraphDeps:
        return GraphDeps(
            settings=settings or small_chunk_settings,
            conn=conn,
            store=ConversationStore(conn),
            chat_chain=chat_chain or FakeChatChain(),
            embedder=embedder,
            vision_provider=vision_provider,
            reranker=StubReranker() if reranker == "default" else reranker,
            web_search=web_search,
            emit=events.append,
        )

    return factory
