"""会话接口：创建、列表、详情、历史消息、删除。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import db as dbmod
from app.config import get_settings
from app.store import ConversationStore

router = APIRouter(prefix="/api", tags=["conversations"])


def _with_store(handler):
    """小工具：每个请求开一个连接，用完关掉（避免跨线程共享 sqlite 连接）。"""

    def wrapper(*args, **kwargs):
        settings = get_settings()
        settings.ensure_dirs()
        conn = dbmod.connect(settings=settings)
        try:
            return handler(ConversationStore(conn), *args, **kwargs)
        finally:
            conn.close()

    return wrapper


@router.post("/conversations")
def create_conversation(session_id: str | None = Query(None), title: str | None = Query(None)):
    """新建会话（前端也可以直接用 /api/chat 自动建会话）。"""

    @_with_store
    def handler(store: ConversationStore):
        return store.create(session_id=session_id, title=title)

    return handler()


@router.get("/conversations")
def list_conversations(limit: int = Query(50, ge=1, le=200)):
    @_with_store
    def handler(store: ConversationStore):
        return {"items": store.list(limit=limit)}

    return handler()


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    @_with_store
    def handler(store: ConversationStore):
        conversation = store.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        conversation["message_count"] = store.message_count(conversation_id)
        return conversation

    return handler()


@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, limit: int | None = Query(None, ge=1, le=500)):
    @_with_store
    def handler(store: ConversationStore):
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"items": store.messages(conversation_id, limit=limit)}

    return handler()


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    @_with_store
    def handler(store: ConversationStore):
        return {"deleted": store.delete(conversation_id)}

    return handler()
