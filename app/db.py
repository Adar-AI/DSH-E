"""SQLite 数据层：连接、sqlite-vec 装载、建表与统计。

设计要点：
* 每个调用方拿自己的连接（`connect()`），Web 层按请求创建，避免跨线程共享；
* `journal_mode=WAL` + `foreign_keys=ON` 是硬要求，建表与级联删除都依赖它们；
* sqlite-vec 属于可选扩展：装载失败时业务功能不受影响，检索自动退化为纯关键词。
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Sequence

from app.config import Settings, get_settings

APP_DIR = Path(__file__).resolve().parent
CORE_SCHEMA_PATH = APP_DIR / "schema.sql"
VEC_SCHEMA_PATH = APP_DIR / "schema_vec.sql"

SCHEMA_VERSION = 1

# 建库后应当存在的业务表（校验用）
EXPECTED_TABLES = {
    "documents",
    "chunks",
    "chunks_fts",
    "conversations",
    "messages",
    "tickets",
    "ticket_events",
}
VEC_TABLE = "vec_chunks"


# --------------------------------------------------------------------- 连接

def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """尝试装载 sqlite-vec 扩展，成功返回 True（失败不抛异常）。"""
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False

    if not hasattr(conn, "enable_load_extension"):
        return False

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (sqlite3.OperationalError, sqlite3.DatabaseError, AttributeError):
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except (sqlite3.OperationalError, AttributeError):
            pass
    return vec_version(conn) is not None


def vec_version(conn: sqlite3.Connection) -> str | None:
    """返回 sqlite-vec 版本号；未装载时返回 None。"""
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return str(row[0])


def connect(
    db_file: Path | str | None = None,
    *,
    settings: Settings | None = None,
    load_vec: bool = True,
) -> sqlite3.Connection:
    """打开（必要时创建）数据库连接，并完成基础 PRAGMA 设置。"""
    settings = settings or get_settings()
    path = Path(db_file) if db_file is not None else settings.db_file
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    if load_vec:
        load_sqlite_vec(conn)
    return conn


# --------------------------------------------------------------------- 建表

def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    return {str(row[0]) for row in rows}


def init_db(
    conn: sqlite3.Connection,
    *,
    embedding_dim: int | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """建表（幂等）。返回本次新建的表名列表。"""
    settings = settings or get_settings()
    dim = int(embedding_dim or settings.embedding_dim)
    before = table_names(conn)

    conn.executescript(CORE_SCHEMA_PATH.read_text(encoding="utf-8"))

    if vec_version(conn) is not None:
        vec_sql = VEC_SCHEMA_PATH.read_text(encoding="utf-8").replace(
            "{embedding_dim}", str(dim)
        )
        conn.executescript(vec_sql)

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()

    return sorted(table_names(conn) - before)


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def has_vec_table(conn: sqlite3.Connection) -> bool:
    return VEC_TABLE in table_names(conn)


# ------------------------------------------------------------------ 向量辅助

def to_vec_blob(vector: Sequence[float]) -> bytes:
    """把向量序列转成 sqlite-vec 需要的 float32 小端字节串。"""
    values = [float(x) for x in vector]
    try:
        import sqlite_vec  # type: ignore

        return sqlite_vec.serialize_float32(values)
    except (ImportError, AttributeError):
        return struct.pack(f"<{len(values)}f", *values)


def knn_search(
    conn: sqlite3.Connection,
    query_vector: Sequence[float],
    *,
    limit: int = 10,
    chunk_ids: Iterable[int] | None = None,
) -> list[tuple[int, float]]:
    """向量近邻检索，返回 [(chunk_id, distance), ...]，距离越小越相似。"""
    if not has_vec_table(conn):
        return []

    blob = to_vec_blob(query_vector)
    sql = (
        "SELECT chunk_id, distance FROM vec_chunks "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
    )
    rows = conn.execute(sql, (blob, int(limit))).fetchall()

    allowed = set(chunk_ids) if chunk_ids is not None else None
    result: list[tuple[int, float]] = []
    for row in rows:
        chunk_id = int(row[0])
        if allowed is not None and chunk_id not in allowed:
            continue
        result.append((chunk_id, float(row[1])))
    return result


# --------------------------------------------------------------------- 统计

def knowledge_stats(conn: sqlite3.Connection) -> dict[str, int | str | None]:
    """知识库与业务数据概览，供 /api/health 与初始化脚本展示。"""

    def scalar(sql: str, default: int = 0) -> int:
        try:
            row = conn.execute(sql).fetchone()
        except sqlite3.OperationalError:
            return default
        return int(row[0]) if row and row[0] is not None else default

    stats: dict[str, int | str | None] = {
        "documents": scalar("SELECT COUNT(*) FROM documents"),
        "chunks": scalar("SELECT COUNT(*) FROM chunks"),
        "conversations": scalar("SELECT COUNT(*) FROM conversations"),
        "messages": scalar("SELECT COUNT(*) FROM messages"),
        "tickets": scalar("SELECT COUNT(*) FROM tickets"),
        "vec_version": vec_version(conn),
        "schema_version": schema_version(conn),
    }
    stats["vectors"] = (
        scalar(f"SELECT COUNT(*) FROM {VEC_TABLE}") if has_vec_table(conn) else 0
    )
    return stats
