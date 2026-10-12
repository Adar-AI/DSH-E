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
from app.config import BASE_DIR, Settings, get_settings  # noqa: E402
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

# 语义不全 / 缺主语的问题：靠 rerank 分数本身区分不出"问得不清楚"，
# 因此这些用例是用来证明「第一段实体粗判」必需性的（见文档 13.2 两段式判定）。
INCOMPLETE_QUESTIONS = [
    "保修期是多少？",
    "这个多少钱？",
    "坏了怎么办？",
]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    (CORPUS / "打印机使用手册.md").write_text(MANUAL, encoding="utf-8")
    (CORPUS / "售后服务规则.md").write_text(RULES, encoding="utf-8")

    # 检索/嵌入用固定维度的小语料库；重排与置信阈值沿用 .env 里的真实配置
    base = get_settings()
    settings = Settings(
        _env_file=None,
        db_path=WORK / "app.db",
        upload_dir=WORK / "uploads",
        docs_dir=WORK / "docs",
        embedding_dim=1024,
        rerank_model_path=base.rerank_model_path,
        rerank_device=base.rerank_device,
        rerank_batch_size=base.rerank_batch_size,
        rerank_max_length=base.rerank_max_length,
        rerank_min_score=base.rerank_min_score,
        rerank_relative_floor=base.rerank_relative_floor,
        rerank_cliff_ratio=base.rerank_cliff_ratio,
        rerank_top_k_max=base.rerank_top_k_max,
        confidence_high=base.confidence_high,
        confidence_low=base.confidence_low,
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

    calibrate_rerank(conn, embedder, settings)

    conn.close()
    return 0


def calibrate_rerank(conn, embedder, settings) -> None:
    """v2.1：用真实重排模型标定「三档置信」与「动态 TopK」的阈值。"""
    from app.rag.fusion import (
        Candidate,
        apply_rerank_scores,
        rrf_fuse,
        select_supporting,
    )
    from app.rag.reranker import get_reranker

    reranker = get_reranker(settings)
    healthy, detail = reranker.health()
    print(f"\n==== ② 重排分与动态 TopK 标定 ====")
    print(f"重排器：{reranker.name}｜{detail}")
    if not healthy:
        print("⚠️ 重排模型不可用，跳过本段（会退化为词面覆盖率兜底）")
        return

    def probe(question: str) -> tuple[float, list, object]:
        # min_score=0 拿到完整候选，便于观察重排分布
        outcome = hybrid_retrieve(
            conn, question, embedder=embedder, settings=settings, min_score=0.0
        )
        candidates = [
            Candidate.from_retrieved(chunk.as_dict(), channel="internal:q1")
            for chunk in outcome.chunks
        ]
        if not candidates:
            return 0.0, [], None
        fused = rrf_fuse({"internal:q1": candidates})
        scores = reranker.score(question, [item.text for item in fused])
        ranked = apply_rerank_scores(fused, scores)
        selection = select_supporting(
            ranked,
            min_score=settings.rerank_min_score,
            relative_floor=settings.rerank_relative_floor,
            cliff_ratio=settings.rerank_cliff_ratio,
            top_k_max=settings.rerank_top_k_max,
        )
        return selection.top1, ranked, selection

    print("\n--- 可回答问题（期望：高置信，top1 应明显高于 CONFIDENCE_HIGH）---")
    for question in QUESTIONS:
        top1, ranked, selection = probe(question)
        if not ranked:
            print(f"Q: {question} → 无候选")
            continue
        kept = len(selection.kept) if selection else 0
        print(f"Q: {question}")
        print(
            f"   top1={top1:.3f} 保留 {kept}/{len(ranked)} 条"
            f"（截断={selection.cutoff if selection else '-'}）"
        )
        for item in ranked[:3]:
            marker = "✔" if selection and item in selection.kept else "✘"
            print(f"   {marker} {item.rerank_score:.3f} | {item.citation}")

    print("\n--- 语义不全的问题（rerank 分数区分不出「问得清不清楚」→ 必须靠第一段实体粗判）---")
    for question in INCOMPLETE_QUESTIONS:
        top1, ranked, selection = probe(question)
        if not ranked:
            print(f"Q: {question} → 无候选")
            continue
        if top1 >= settings.confidence_high:
            verdict = "分数不低，但问题缺主语 → 第一段应判 medium/low"
        elif top1 >= settings.confidence_low:
            verdict = "落在中置信区间 → 反问澄清"
        else:
            verdict = "落在低置信区间 → 拒答"
        print(f"Q: {question}\n   top1={top1:.3f}（{verdict}）")

    print(f"\n当前配置：CONFIDENCE_HIGH={settings.confidence_high} "
          f"CONFIDENCE_LOW={settings.confidence_low} "
          f"RERANK_MIN_SCORE={settings.rerank_min_score} "
          f"RERANK_RELATIVE_FLOOR={settings.rerank_relative_floor}")


if __name__ == "__main__":
    raise SystemExit(main())
