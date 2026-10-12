-- 向量表（依赖 sqlite-vec 扩展；{embedding_dim} 由 app/db.py 按配置替换）
-- 与 chunks 一一对应：chunk_id 即 chunks.id。删除文档时由 db 层显式清理。

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[{embedding_dim}]
);
