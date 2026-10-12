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
    """小块配置：让短文档也能产生多个分块，便于验证分块与覆盖逻辑。"""
    return settings.model_copy(update={"chunk_size": 60, "chunk_overlap": 10})


@pytest.fixture()
def conn(settings: Settings):
    """已建表的临时数据库连接。"""
    connection = dbmod.connect(settings=settings)
    dbmod.init_db(connection, settings=settings)
    try:
        yield connection
    finally:
        connection.close()
