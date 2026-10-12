"""会话接口：创建、列表、详情、历史消息、删除、结束会话定稿工单。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import db as dbmod
from app.config import get_settings
from app.deps import get_chat_chain
from app.handoff import HandoffService
from app.store import ConversationStore
from app.tickets.service import TicketService

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
def get_messages(
    conversation_id: str,
    limit: int | None = Query(None, ge=1, le=500),
    after_id: int | None = Query(None, ge=0, description="只取该消息之后的增量（客户页轮询用）"),
):
    @_with_store
    def handler(store: ConversationStore):
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"items": store.messages(conversation_id, limit=limit, after_id=after_id)}

    return handler()


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    @_with_store
    def handler(store: ConversationStore):
        return {"deleted": store.delete(conversation_id)}

    return handler()


# ------------------------------------------------------------------ 转人工（V3）


def _with_handoff(handler):
    """转人工相关接口：一个请求一个连接，顺带带上工单服务。"""

    def wrapper(*args, **kwargs):
        settings = get_settings()
        settings.ensure_dirs()
        conn = dbmod.connect(settings=settings)
        try:
            store = ConversationStore(conn)
            return handler(HandoffService(conn, settings=settings, store=store), store, *args, **kwargs)
        finally:
            conn.close()

    return wrapper


@router.post("/conversations/{conversation_id}/handoff")
def request_handoff(conversation_id: str, reason: str = Query("客户点击了「转人工客服」")):
    """客户要求转人工：标记状态 + 立即建单（紧急度较急）+ 系统提示（需求 F24）。"""

    @_with_handoff
    def handler(handoff: HandoffService, store: ConversationStore):
        try:
            outcome = handoff.request(conversation_id, reason=reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        outcome["messages"] = store.messages(conversation_id)
        return outcome

    return handler()


@router.get("/handoffs")
def list_handoffs():
    """工单台用：待接入 / 人工中的会话列表（需求 F24）。"""

    @_with_handoff
    def handler(handoff: HandoffService, store: ConversationStore):
        items = handoff.pending()
        return {
            "items": items,
            "waiting": sum(1 for item in items if item.get("handoff_state") == "requested"),
            "serving": sum(1 for item in items if item.get("handoff_state") == "claimed"),
        }

    return handler()


@router.post("/conversations/{conversation_id}/agent-messages")
def agent_reply(conversation_id: str, text: str = Query(..., min_length=1)):
    """客服发言：写入 role=agent 的消息，首次发言自动接入（需求 F25）。"""

    @_with_handoff
    def handler(handoff: HandoffService, store: ConversationStore):
        try:
            return handoff.reply(conversation_id, text)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return handler()


@router.post("/conversations/{conversation_id}/close")
def close_conversation(
    conversation_id: str,
    force: bool = Query(False, description="同一会话已生成过工单时，是否再生成一张"),
    note: str = Query("", description="定稿备注"),
):
    """结束会话：抽取结构化字段并生成工单（需求 F7）。"""
    settings = get_settings()
    settings.ensure_dirs()
    conn = dbmod.connect(settings=settings)
    try:
        service = TicketService(conn)
        try:
            ticket = service.create_from_conversation(
                conversation_id,
                chat_chain=get_chat_chain(settings),
                settings=settings,
                force=force,
                note=note,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        ticket["events"] = service.events(int(ticket["id"]))
        return ticket
    finally:
        conn.close()
