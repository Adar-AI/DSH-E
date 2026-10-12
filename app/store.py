"""会话与消息的读写（仓储层）。

这一层只负责数据库读写与 JSON 编解码，不含任何模型调用，便于单测与复用。
约定：时间字段统一用 `now_iso()`，JSON 字段统一用 `_dumps`/`_loads` 处理。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from app.utils import new_id, now_iso

DEFAULT_TITLE = "新会话"
MAX_TITLE_LENGTH = 24


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _loads(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class ConversationStore:
    """会话、消息、工单草稿的数据库访问。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------ 会话

    def create(
        self,
        *,
        session_id: str | None = None,
        title: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        conv_id = conversation_id or new_id("conv")
        timestamp = now_iso()
        self.conn.execute(
            """
            INSERT INTO conversations(id, session_id, title, status, created_at, updated_at)
            VALUES (?, ?, ?, 'active', ?, ?)
            """,
            (conv_id, session_id, title or DEFAULT_TITLE, timestamp, timestamp),
        )
        self.conn.commit()
        return self.get(conv_id) or {}

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        data = _row_to_dict(row)
        if data is not None:
            data["ticket_draft"] = _loads(data.pop("ticket_draft_json", None), {})
        return data

    def ensure(
        self,
        conversation_id: str | None,
        *,
        session_id: str | None = None,
        first_text: str = "",
    ) -> dict[str, Any]:
        """拿到会话；不存在就新建（用首句问题当标题）。"""
        if conversation_id:
            existing = self.get(conversation_id)
            if existing is not None:
                return existing

        title = (first_text or "").strip().replace("\n", " ")[:MAX_TITLE_LENGTH] or DEFAULT_TITLE
        return self.create(
            session_id=session_id, title=title, conversation_id=conversation_id
        )

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conv_id = c.id) AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["ticket_draft"] = _loads(data.pop("ticket_draft_json", None), {})
            result.append(data)
        return result

    def update(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        draft: dict[str, Any] | None = None,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [now_iso()]
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if draft is not None:
            sets.append("ticket_draft_json = ?")
            params.append(_dumps(draft))
        params.append(conversation_id)
        self.conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", params
        )
        self.conn.commit()

    def get_draft(self, conversation_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT ticket_draft_json FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return {}
        return _loads(row["ticket_draft_json"], {}) or {}

    def set_draft(self, conversation_id: str, draft: dict[str, Any]) -> None:
        self.update(conversation_id, draft=draft)

    def delete(self, conversation_id: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ 消息

    def add_message(
        self,
        conversation_id: str,
        role: str,
        text: str = "",
        *,
        image_path: str | None = None,
        vision: dict[str, Any] | None = None,
        citations: Iterable[Any] | None = None,
        trace: dict[str, Any] | None = None,
    ) -> int:
        timestamp = now_iso()
        cursor = self.conn.execute(
            """
            INSERT INTO messages(conv_id, role, text, image_path, vision_json,
                                 citations_json, trace_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                text or "",
                image_path,
                _dumps(vision),
                _dumps(list(citations) if citations is not None else None),
                _dumps(trace),
                timestamp,
            ),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def messages(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE conv_id = ? ORDER BY id"
        params: list[Any] = [conversation_id]
        if limit:
            sql = (
                "SELECT * FROM (SELECT * FROM messages WHERE conv_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            params.append(int(limit))

        result: list[dict[str, Any]] = []
        for row in self.conn.execute(sql, params).fetchall():
            data = dict(row)
            data["vision"] = _loads(data.pop("vision_json", None), {})
            data["citations"] = _loads(data.pop("citations_json", None), [])
            data["trace"] = _loads(data.pop("trace_json", None), {})
            result.append(data)
        return result

    def history_before_last_user(
        self, conversation_id: str, *, turns: int = 6
    ) -> list[dict[str, str]]:
        """取多轮对话历史（给提示词用），并排除最新那条用户消息（它就是当前问题）。"""
        rows = self.conn.execute(
            "SELECT role, text FROM messages WHERE conv_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, int(turns) * 2 + 1),
        ).fetchall()

        items = [dict(row) for row in rows]
        if items and items[0]["role"] == "user":
            items = items[1:]
        items = items[: int(turns) * 2]
        items.reverse()
        return [
            {"role": str(item["role"]), "content": str(item["text"] or "")}
            for item in items
            if item["role"] in {"user", "assistant"} and (item["text"] or "").strip()
        ]

    def message_count(self, conversation_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conv_id = ?", (conversation_id,)
        ).fetchone()
        return int(row[0]) if row else 0
