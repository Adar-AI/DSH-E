"""对话接口：`POST /api/chat` 以 SSE 流式返回处理过程与回答。

事件序列（前端按事件名渲染）：

| 事件 | 时机 | 数据 |
|---|---|---|
| `meta` | 会话确定后立刻 | `conversation_id`、标题 |
| `stage` | 每个耗时阶段开始 | `stage`、`message` |
| `vision` | 图片识别完成 | `ok`、`summary`、`detail`、`error` |
| `retrieval` | 检索完成 | `hit`、`best_score`、`mode`、`chunks`（含分数明细） |
| `token` | 模型逐段产出 | `text` |
| `draft` | 工单草稿更新 | `draft` |
| `done` | 全部完成 | `answer`、`citations`、`ticket_draft`、`trace` |
| `error` | 异常 | `message` |

用同步生成器：FastAPI 会把它放到线程池里迭代，因此里面的 sqlite 与模型调用
（都是阻塞式）不会卡住事件循环。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import db as dbmod
from app.config import Settings, get_settings
from app.deps import build_deps, get_checkpointer
from app.graph.agent import build_graph
from app.llm.vision import SUPPORTED_IMAGE_SUFFIXES
from app.store import ConversationStore
from app.utils import new_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse(event: str, data: dict[str, Any]) -> str:
    """按 SSE 规范格式化一条事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_image(payload: bytes, suffix: str, settings: Settings) -> str:
    settings.ensure_dirs()
    target = settings.upload_path / f"{new_id('img')}{suffix}"
    target.write_bytes(payload)
    return str(target)


def stream_chat(
    *,
    text: str,
    conversation_id: str | None,
    session_id: str | None,
    image_path: str | None,
    settings: Settings,
) -> Iterator[str]:
    """执行一轮对话并把过程以 SSE 形式产出。

    连接用 `check_same_thread=False`：LangGraph 会在自己的工作线程里执行节点，
    而这条连接要跟着请求走。本图是线性流程（无并行分支），节点之间不会并发访问，
    加上 SQLite 处于 WAL 模式，因此跨线程复用是安全的。
    """
    conn = dbmod.connect(settings=settings, check_same_thread=False)
    try:
        store = ConversationStore(conn)
        conversation = store.ensure(
            conversation_id, session_id=session_id, first_text=text
        )
        conv_id = str(conversation["id"])

        yield sse(
            "meta",
            {
                "conversation_id": conv_id,
                "title": conversation.get("title"),
                "status": conversation.get("status"),
            },
        )
        # 先落用户消息：即使后面模型挂了，这轮提问也不会丢
        store.add_message(conv_id, "user", text, image_path=image_path)

        final_state: dict[str, Any] = {}
        try:
            deps = build_deps(conn, settings)
            graph = build_graph(deps, checkpointer=get_checkpointer(settings))
            initial = {"conversation_id": conv_id, "text": text, "image_path": image_path}

            for mode, chunk in graph.stream(
                initial,
                config={"configurable": {"thread_id": conv_id}},
                stream_mode=["custom", "values"],
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    yield sse(str(chunk.get("type") or "event"), chunk)
                elif mode == "values" and isinstance(chunk, dict):
                    final_state = chunk
        except Exception as exc:  # noqa: BLE001 - 任何异常都要变成可读事件
            logger.exception("对话图执行失败")
            yield sse("error", {"message": f"服务异常：{exc}"})
            return

        answer = str(final_state.get("answer") or "")
        citations = list(final_state.get("citations") or [])
        trace = {
            "provider": final_state.get("provider"),
            "retrieval_mode": final_state.get("retrieval_mode"),
            "retrieval_reason": final_state.get("retrieval_reason"),
            "best_score": final_state.get("best_score"),
            "hit": final_state.get("hit"),
            "vision_ok": final_state.get("vision_ok"),
            "fallback_used": final_state.get("fallback_used"),
            "answer_error": final_state.get("answer_error"),
        }

        store.add_message(
            conv_id,
            "assistant",
            answer,
            vision=final_state.get("vision_detail") or None,
            citations=citations,
            trace=trace,
        )
        draft = final_state.get("ticket_draft") or store.get_draft(conv_id)

        yield sse(
            "done",
            {
                "conversation_id": conv_id,
                "answer": answer,
                "citations": citations,
                "chunks": final_state.get("retrieved") or [],
                "ticket_draft": draft,
                "trace": trace,
            },
        )
    finally:
        conn.close()


@router.post("/chat")
async def chat(
    text: str = Form(""),
    conversation_id: str | None = Form(None),
    session_id: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    """一轮对话（文字 + 可选图片），返回 SSE 流。"""
    settings = get_settings()
    settings.ensure_dirs()

    image_path: str | None = None
    if image is not None and (image.filename or "").strip():
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图片类型 {suffix or '(无扩展名)'}",
            )
        payload = await image.read()
        if not payload:
            raise HTTPException(status_code=400, detail="上传的图片为空")
        image_path = _save_image(payload, suffix, settings)

    text = (text or "").strip()
    if not text and not image_path:
        raise HTTPException(status_code=400, detail="请提供文字问题或上传图片")

    return StreamingResponse(
        stream_chat(
            text=text,
            conversation_id=conversation_id,
            session_id=session_id,
            image_path=image_path,
            settings=settings,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
