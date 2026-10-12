"""检索打分与阈值标定脚本。

用真实 BGE-M3 在示例语料上跑一批问题，打印「命中类」与「未命中类」的分数分布，
用来确认 RETRIEVE_MIN_SCORE 是否落在两类之间的安全间隔里。
本轮标定结论记录在 docs/架构说明.md 第 5.1 节。

用法：
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\calibrate.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db as dbmod  # noqa: E402
from app.config import BASE_DIR, Settings  # noqa: E402
from app.ingest.pipeline import ingest_directory  # noqa: E402
from app.rag.embedder import BGEEmbedder  # noqa: E402
from app.rag.retriever import hybrid_retrieve  # noqa: E402

WORK = BASE_DIR / "data" / "test_tmp" / "calib"
CORPUS = WORK / "corpus"

MANUAL = """# 打印机使用手册

## 耗材更换

更换硒鼓的步骤：先关闭打印机电源，打开前盖，取出旧硒鼓，装入新硒鼓后合上前盖，
最后开机并复位硒鼓计数器。更换完成后可在设置菜单中查看剩余寿命。

## 报错处理

屏幕出现 E05 表示硒鼓未被识别。请重新安装硒鼓并确认芯片触点清洁，
若仍报错则说明硒鼓芯片损坏，需要更换新的硒鼓。

## 网络设置

无线网络连接失败时，请确认路由器已开启 2.4G 频段，并在打印机设置中重新输入密码。
"""

RULES = """# 售后服务规则

## 保修政策

本机型整机保修期为两年，自购买之日起计算。硒鼓、墨盒等耗材属于易损件，保修期为三个月。
人为损坏、私自拆机、进水受潮均不在免费保修范围内。

## 退换货

签收后七日内出现非人为性能故障可申请退货，十五日内可申请换货，需保留原包装与发票。
"""

QUESTIONS = [
    "硒鼓怎么更换",
    "打印机显示E05是什么意思",
    "保修期是多久",
    "无线网络连不上怎么办",
    "预订国际航班机票价格",
    "今天北京天气怎么样",
]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    (CORPUS / "打印机使用手册.md").write_text(MANUAL, encoding="utf-8")
    (CORPUS / "售后服务规则.md").write_text(RULES, encoding="utf-8")

    settings = Settings(
        _env_file=None,
        db_path=WORK / "app.db",
        upload_dir=WORK / "uploads",
        docs_dir=WORK / "docs",
        embedding_dim=1024,
    )
    conn = dbmod.connect(settings=settings)
    dbmod.init_db(conn, settings=settings)

    embedder = BGEEmbedder(settings.embedding_model_path, device="cpu", dim=1024)
    results = ingest_directory(
        conn, CORPUS, settings=settings, embedder=embedder, source="calib"
    )
    for item in results:
        print(f"入库 {item.filename}: {item.chunks} 块 / {item.vectors} 向量（{item.mode}）")

    print(f"\n阈值 = {settings.retrieve_min_score}\n")
    for question in QUESTIONS:
        outcome = hybrid_retrieve(conn, question, embedder=embedder, settings=settings)
        print(f"Q: {question}")
        print(
            f"   命中={outcome.hit} 最高分={outcome.best_score} "
            f"模式={outcome.mode} 关键词候选={outcome.keyword_candidates} "
            f"向量候选={outcome.vector_candidates}"
        )
        for chunk in outcome.chunks[:2]:
            print(
                f"   - {chunk.score:6.3f} | 覆盖={chunk.coverage:.2f} "
                f"向量={chunk.vector_sim:.3f} rrf={chunk.rrf:.2f} | {chunk.citation}"
            )
            print(f"     {chunk.text[:40].replace(chr(10), ' ')}...")
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
