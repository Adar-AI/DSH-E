"""初始化本地数据库：建表、装载向量扩展并打印概览。

用法：
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\init_db.py
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\init_db.py --reset   # 谨慎：清空重建
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="初始化多模态智能客服的本地数据库")
    parser.add_argument("--db", default=None, help="指定数据库文件（默认取配置 DB_PATH）")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="删除已有数据库文件后重建（会丢失会话与工单数据）",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    target = Path(args.db) if args.db else settings.db_file

    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(target) + suffix)
            if candidate.exists():
                candidate.unlink()
                print(f"已删除 {candidate}")

    conn = dbmod.connect(target, settings=settings)
    try:
        created = dbmod.init_db(conn, settings=settings)
        stats = dbmod.knowledge_stats(conn)
    finally:
        conn.close()

    # FTS5 与 vec0 会派生一批影子表，这里只报告业务表
    business = set(dbmod.EXPECTED_TABLES) | {dbmod.VEC_TABLE}
    created_business = [name for name in created if name in business]

    print(f"数据库：{target}")
    print(
        "本次新建表："
        + ("、".join(created_business) if created_business else "（已存在，无需新建）")
    )
    print("概览：")
    for key, value in stats.items():
        print(f"  {key:16} = {value}")

    if stats["vec_version"] is None:
        print("[提示] sqlite-vec 未装载：检索会自动退化为纯关键词模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
