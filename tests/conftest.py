"""测试夹具。

注意：刻意不使用 pytest 的 `tmp_path` / `tmpdir` 夹具——它们底层用
`tempfile.mkdtemp()`，会把目录 chmod 成 0700，在 DSH 沙箱里当前令牌对该目录
没有访问权（写文件直接 PermissionError）。这里改成在项目内 `data/test_tmp/`
下自建目录（普通权限），用完手动清理。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
TEST_TMP_ROOT = ROOT / "data" / "test_tmp"


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
    """隔离到临时目录的配置：不读 .env，不碰真实的 data/app.db。"""
    return Settings(
        _env_file=None,
        db_path=work_dir / "app.db",
        upload_dir=work_dir / "uploads",
        docs_dir=work_dir / "docs",
    )
