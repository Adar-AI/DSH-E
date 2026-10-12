"""数据层测试：建表、向量扩展、FTS5 同步、级联删除、统计。"""

from __future__ import annotations

import pytest

from app import db as dbmod
from app.rag.tokenize import build_match_query, segment
from app.utils import now_iso


@pytest.fixture()
def conn(settings):
    connection = dbmod.connect(settings=settings)
    dbmod.init_db(connection, settings=settings)
    try:
        yield connection
    finally:
        connection.close()


def _insert_document(connection, filename="产品手册.pdf", sha="sha-1", source="sample"):
    connection.execute(
        "INSERT INTO documents(filename, sha256, source, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, sha, source, now_iso()),
    )
    return int(
        connection.execute(
            "SELECT id FROM documents WHERE sha256 = ?", (sha,)
        ).fetchone()[0]
    )


def _insert_chunk(connection, doc_id, ordinal, text, section=None, page=None):
    connection.execute(
        """
        INSERT INTO chunks(doc_id, ordinal, section, page, text, text_tokens, char_len)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, ordinal, section, page, text, segment(text), len(text)),
    )
    return int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])


def _fts_match(connection, question):
    expression = build_match_query(question)
    rows = connection.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (expression,)
    ).fetchall()
    return [int(row[0]) for row in rows]


# ------------------------------------------------------------------ 建表

def test_init_creates_expected_tables(conn):
    assert dbmod.EXPECTED_TABLES <= dbmod.table_names(conn)
    assert dbmod.schema_version(conn) == dbmod.SCHEMA_VERSION


def test_init_is_idempotent(conn):
    created = dbmod.init_db(conn)
    assert created == []


def test_connection_pragmas(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


# ------------------------------------------------------------ 向量检索

def test_sqlite_vec_is_loaded(conn):
    assert dbmod.vec_version(conn) is not None, "sqlite-vec 未装载，向量检索不可用"
    assert dbmod.has_vec_table(conn)


def test_knn_search_returns_nearest_first(conn, settings):
    dim = settings.embedding_dim
    vectors = {
        1: [1.0] + [0.0] * (dim - 1),
        2: [0.0, 1.0] + [0.0] * (dim - 2),
        3: [0.0, 0.0, 1.0] + [0.0] * (dim - 3),
    }
    for chunk_id, vector in vectors.items():
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, dbmod.to_vec_blob(vector)),
        )
    conn.commit()

    hits = dbmod.knn_search(conn, vectors[2], limit=3)

    assert [chunk_id for chunk_id, _ in hits][0] == 2
    assert hits[0][1] == pytest.approx(0.0, abs=1e-5)


def test_knn_search_can_be_filtered_by_chunk_ids(conn, settings):
    dim = settings.embedding_dim
    for chunk_id in (1, 2):
        vector = [1.0] + [0.0] * (dim - 1)
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, dbmod.to_vec_blob(vector)),
        )
    conn.commit()

    hits = dbmod.knn_search(conn, [1.0] + [0.0] * (dim - 1), limit=5, chunk_ids=[2])

    assert [chunk_id for chunk_id, _ in hits] == [2]


# ------------------------------------------------------------ 全文检索

def test_fts_triggers_keep_index_in_sync(conn):
    doc_id = _insert_document(conn)
    chunk_id = _insert_chunk(
        conn, doc_id, 1, "更换硒鼓的步骤：打开前盖，取出旧硒鼓。", section="耗材更换", page=12
    )
    conn.commit()

    assert chunk_id in _fts_match(conn, "硒鼓怎么换")
    assert chunk_id in _fts_match(conn, "更换硒鼓")

    # 更新：旧内容不再命中，新内容可以命中
    conn.execute(
        "UPDATE chunks SET text = ?, text_tokens = ?, char_len = ? WHERE id = ?",
        ("复位加热组件的方法", segment("复位加热组件的方法"), 10, chunk_id),
    )
    conn.commit()
    assert _fts_match(conn, "硒鼓") == []
    assert chunk_id in _fts_match(conn, "加热组件")

    # 删除：索引同步清除
    conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    conn.commit()
    assert _fts_match(conn, "加热组件") == []


def test_fts_recall_is_usable_for_chinese(conn):
    doc_id = _insert_document(conn)
    target = _insert_chunk(conn, doc_id, 1, "本机型保修期为两年，自购买之日起计算。")
    _insert_chunk(conn, doc_id, 2, "无线网络连接失败时请检查路由器频段。")
    conn.commit()

    assert target in _fts_match(conn, "保修期多久")


# ------------------------------------------------------------ 级联与统计

def test_deleting_document_cascades_chunks_and_index(conn):
    doc_id = _insert_document(conn)
    for ordinal in range(1, 4):
        _insert_chunk(conn, doc_id, ordinal, f"第{ordinal}段关于打印机报错E0{ordinal}的说明")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3

    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert _fts_match(conn, "打印机报错") == []


def test_document_sha256_is_unique(conn):
    _insert_document(conn, sha="dup")
    with pytest.raises(Exception):
        _insert_document(conn, filename="另一份.pdf", sha="dup")


def test_ticket_defaults_and_event_cascade(conn):
    conn.execute(
        "INSERT INTO tickets(ticket_no, created_at, updated_at) VALUES (?, ?, ?)",
        ("T-20261012-0001", now_iso(), now_iso()),
    )
    row = conn.execute(
        "SELECT id, status, urgency FROM tickets WHERE ticket_no = ?",
        ("T-20261012-0001",),
    ).fetchone()
    assert row["status"] == "待处理"
    assert row["urgency"] == "普通"

    conn.execute(
        """
        INSERT INTO ticket_events(ticket_id, from_status, to_status, note, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (row["id"], "待处理", "处理中", "客服已接单", now_iso()),
    )
    conn.commit()

    conn.execute("DELETE FROM tickets WHERE id = ?", (row["id"],))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM ticket_events").fetchone()[0] == 0


def test_messages_cascade_with_conversation(conn):
    conn.execute(
        "INSERT INTO conversations(id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("conv-1", "sess-1", now_iso(), now_iso()),
    )
    conn.execute(
        "INSERT INTO messages(conv_id, role, text, created_at) VALUES (?, ?, ?, ?)",
        ("conv-1", "user", "打印机报错E05", now_iso()),
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1

    conn.execute("DELETE FROM conversations WHERE id = ?", ("conv-1",))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_knowledge_stats_reports_counts(conn, settings):
    stats = dbmod.knowledge_stats(conn)
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["vectors"] == 0
    assert stats["schema_version"] == dbmod.SCHEMA_VERSION

    doc_id = _insert_document(conn)
    _insert_chunk(conn, doc_id, 1, "保修期为两年")
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
        (1, dbmod.to_vec_blob([0.0] * settings.embedding_dim)),
    )
    conn.commit()

    stats = dbmod.knowledge_stats(conn)
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["vectors"] == 1
