"""健康检查：数据库、知识库规模、检索模式、各模型通道状态（密钥脱敏）。"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app import db as dbmod
from app.config import get_settings, mask_secret
from app.deps import get_chat_chain, get_vision_provider

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(probe: bool = Query(False, description="是否真实调用各通道做连通性探测")):
    settings = get_settings()
    settings.ensure_dirs()

    conn = dbmod.connect(settings=settings)
    try:
        stats = dbmod.knowledge_stats(conn)
    finally:
        conn.close()

    chain = get_chat_chain(settings)
    chat_channels = []
    for provider in chain.providers:
        item = {
            "name": provider.name,
            "model": provider.model,
            "base_url": provider.base_url,
            "available": provider.available()[0],
            "reason": provider.available()[1],
        }
        if probe:
            ok, detail = provider.ping()
            item["reachable"] = ok
            item["probe_detail"] = detail
        chat_channels.append(item)

    vision = get_vision_provider(settings)
    vision_available, vision_reason = vision.available()

    return {
        "status": "ok",
        "db": {"path": str(settings.db_file), "schema_version": stats["schema_version"]},
        "knowledge": {
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "vectors": stats["vectors"],
            "vec_version": stats["vec_version"],
        },
        "retrieval": {
            "mode": "hybrid" if stats["vectors"] else "keyword_only",
            "min_score": settings.retrieve_min_score,
            "top_k": settings.retrieve_top_k,
        },
        "chat": {"channels": chat_channels, "max_tokens": settings.chat_max_tokens},
        "vision": {
            "model": vision.model,
            "base_url": getattr(vision, "base_url", ""),
            "available": vision_available,
            "reason": vision_reason,
            "api_key": mask_secret(settings.qwen_api_key),
        },
        "embedding": {
            "model_path": settings.embedding_model_path,
            "device": settings.embedding_device,
            "dim": settings.embedding_dim,
        },
    }
