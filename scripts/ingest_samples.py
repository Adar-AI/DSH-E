"""把企业资料入库到本地知识库。

用法：
    # 入库演示资料（data/docs/samples）
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\ingest_samples.py

    # 入库你自己的资料（放进 data/docs/user 后执行）
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\ingest_samples.py --user

    # 内容变了要重新入库（默认按 sha256 跳过未变化的文件）
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\ingest_samples.py --force

    # 换了嵌入模型或想整体重算向量
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\ingest_samples.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db as dbmod  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest.pipeline import ingest_directory, rebuild_index  # noqa: E402
from app.rag.embedder import get_embedder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 PDF/MD/TXT 资料入库到本地知识库")
    parser.add_argument("--user", action="store_true", help="改为入库 data/docs/user 里的私有资料")
    parser.add_argument("--force", action="store_true", help="即使内容没变也重新入库")
    parser.add_argument("--rebuild", action="store_true", help="重建全部索引（换嵌入模型后用）")
    parser.add_argument("--no-embed", action="store_true", help="只做关键词索引，不生成向量")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()

    conn = dbmod.connect(settings=settings)
    try:
        dbmod.init_db(conn, settings=settings)
        embedder = None if args.no_embed else get_embedder(settings)

        if args.rebuild:
            results = rebuild_index(conn, settings=settings, embedder=embedder)
        else:
            directory = settings.user_docs_path if args.user else settings.samples_path
            source = "user_dir" if args.user else "sample"
            if not directory.is_dir():
                print(f"目录不存在：{directory}")
                return 1
            results = ingest_directory(
                conn,
                directory,
                settings=settings,
                embedder=embedder,
                source=source,
                force=args.force,
            )

        if not results:
            print("没有找到可入库的文件（支持 .pdf / .md / .markdown / .txt）")
        for item in results:
            mark = {"ingested": "✅", "skipped": "⏭️ ", "failed": "❌"}.get(item.status, "•")
            detail = f"{item.chunks} 块 / {item.vectors} 向量（{item.mode}）"
            suffix = f" — {item.message}" if item.message else ""
            print(f"{mark} {item.filename}: {detail}{suffix}")

        stats = dbmod.knowledge_stats(conn)
        print(
            f"\n知识库现状：文档 {stats['documents']} 份 / 分块 {stats['chunks']} 个 / "
            f"向量 {stats['vectors']} 条（sqlite-vec {stats['vec_version']}）"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
