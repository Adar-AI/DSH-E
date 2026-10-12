"""健康检查：数据库、知识库规模、检索模式、各模型通道状态（密钥脱敏）。

对话与视觉通道都会列出来，并标注「本地 Ollama / 云端网关」——这是排查
"到底哪一步在出网"最直接的依据。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app import db as dbmod
from app.config import get_settings, is_local_url, mask_secret
from app.deps import get_chat_chain, get_vision_providers

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
    for provider in getattr(chain, "providers", []):
        available, reason = provider.available()
        item = {
            "name": provider.name,
            "model": provider.model,
            "base_url": provider.base_url,
            "local": is_local_url(provider.base_url),
            "kind": "本地 Ollama" if is_local_url(provider.base_url) else "云端网关",
            "available": available,
            "reason": reason,
        }
        if probe:
            ok, detail = provider.ping()
            item["reachable"] = ok
            item["probe_detail"] = detail
        chat_channels.append(item)

    vision_channels = []
    for index, provider in enumerate(get_vision_providers(settings)):
        base_url = getattr(provider, "base_url", "")
        local = is_local_url(base_url)
        available, reason = provider.available()
        item = {
            "name": getattr(provider, "name", "vision"),
            "model": getattr(provider, "model", ""),
            "base_url": base_url,
            "local": local,
            "kind": "本地 Ollama" if local else "云端网关",
            "role": "主通道" if index == 0 else "回退通道",
            "available": available,
            "reason": reason,
        }
        if probe:
            ok, detail = provider.ping()
            item["reachable"] = ok
            item["probe_detail"] = detail
        vision_channels.append(item)

    primary_vision = vision_channels[0] if vision_channels else {}

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
            "model": primary_vision.get("model", ""),
            "base_url": primary_vision.get("base_url", ""),
            "kind": primary_vision.get("kind", ""),
            "local": primary_vision.get("local", False),
            "available": primary_vision.get("available", False),
            "reason": primary_vision.get("reason", ""),
            "max_side": settings.vision_max_side,
            "api_key": (
                "（本地通道无需密钥）"
                if primary_vision.get("local")
                else mask_secret(settings.effective_vision_api_key)
            ),
            "channels": vision_channels,
        },
        "embedding": {
            "model_path": settings.embedding_model_path,
            "device": settings.embedding_device,
            "dim": settings.embedding_dim,
        },
    }
