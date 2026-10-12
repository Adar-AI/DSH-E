-- 多模态企业智能客服系统 · 核心表结构（sqlite）
--
-- 约定：
--  * 所有时间字段统一存 ISO8601 字符串（UTC），便于直接比较与展示；
--  * chunks.text 存原文，chunks.text_tokens 存分词后的检索用文本（中文按二元切分，
--    见 app/rag/tokenize.py），FTS5 只索引 text_tokens；
--  * 向量表在 schema_vec.sql（依赖 sqlite-vec 扩展，加载失败时自动跳过）。

-- ---------------------------------------------------------------- 知识库

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    rel_path    TEXT,
    sha256      TEXT    NOT NULL UNIQUE,
    mime        TEXT,
    page_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'upload',   -- upload | sample | user_dir
    status      TEXT    NOT NULL DEFAULT 'ready',    -- ready | failed
    error       TEXT,
    uploaded_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    section     TEXT,
    page        INTEGER,
    text        TEXT    NOT NULL,
    text_tokens TEXT    NOT NULL,
    char_len    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- 全文检索：external content 指向 chunks，索引 text_tokens 列
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text_tokens,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text_tokens) VALUES (new.id, new.text_tokens);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text_tokens)
    VALUES ('delete', old.id, old.text_tokens);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text_tokens)
    VALUES ('delete', old.id, old.text_tokens);
    INSERT INTO chunks_fts(rowid, text_tokens) VALUES (new.id, new.text_tokens);
END;

-- ---------------------------------------------------------------- 会话

CREATE TABLE IF NOT EXISTS conversations (
    id                TEXT PRIMARY KEY,
    session_id        TEXT,
    title             TEXT,
    status            TEXT NOT NULL DEFAULT 'active',  -- active | closed
    ticket_draft_json TEXT,
    -- V3 转人工：none | requested（客户要求人工）| claimed（客服已接入）
    handoff_state     TEXT NOT NULL DEFAULT 'none',
    handoff_reason    TEXT,                            -- 触发原因（话术命中词 / 客户点击按钮）
    handoff_at        TEXT,                            -- 客户请求人工的时间
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
-- 注意：handoff_state 的索引不写在这里。
-- 老库执行本文件时 CREATE TABLE IF NOT EXISTS 是空操作，而那时还没有 handoff_state 列，
-- 直接建索引会报 "no such column" 导致服务起不来。索引统一放到 db.migrate() 里补完列之后再建。

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);

CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id        TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role           TEXT NOT NULL,                      -- user | assistant | system
    text           TEXT NOT NULL DEFAULT '',
    image_path     TEXT,
    vision_json    TEXT,
    citations_json TEXT,
    trace_json     TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);

-- ---------------------------------------------------------------- 工单

CREATE TABLE IF NOT EXISTS tickets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no         TEXT NOT NULL UNIQUE,
    conv_id           TEXT,
    customer_request  TEXT,
    product_model     TEXT,
    fault_phenomenon  TEXT,
    error_info        TEXT,
    tried_actions     TEXT,
    urgency           TEXT NOT NULL DEFAULT '普通',   -- 普通 | 较急 | 紧急
    suggestion        TEXT,
    citations_json    TEXT,
    status            TEXT NOT NULL DEFAULT '待处理', -- 待处理 | 处理中 | 已解决 | 已关闭
    source            TEXT NOT NULL DEFAULT 'auto',   -- auto | manual
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_conv ON tickets(conv_id);

CREATE TABLE IF NOT EXISTS ticket_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   INTEGER NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticket_events_ticket ON ticket_events(ticket_id, id);
