"""工单服务：会话定稿生成工单、查询筛选、状态流转、CSV 导出。

状态机（只允许这些流转，非法流转直接被拒绝，对应需求 F8）：

```
待处理 ──→ 处理中 ──→ 已解决 ──→ 已关闭
   │          ↑  ↓        │
   └──────────┘  └────────┘（允许回退到处理中/待处理）
```

每次流转都会写一条 `ticket_events`，工单台可看到完整历史。
删除会话**不会**删除工单：工单是留档记录，需要独立于会话存在。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from app.config import Settings
from app.llm.providers import ChatChain
from app.store import ConversationStore
from app.tickets.extractor import TicketExtraction, extract_ticket
from app.utils import now_iso

logger = logging.getLogger(__name__)

STATUS_PENDING = "待处理"
STATUS_WORKING = "处理中"
STATUS_RESOLVED = "已解决"
STATUS_CLOSED = "已关闭"

STATUS_ORDER = (STATUS_PENDING, STATUS_WORKING, STATUS_RESOLVED, STATUS_CLOSED)

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATUS_PENDING: {STATUS_WORKING, STATUS_CLOSED},
    STATUS_WORKING: {STATUS_RESOLVED, STATUS_CLOSED, STATUS_PENDING},
    STATUS_RESOLVED: {STATUS_CLOSED, STATUS_WORKING},
    STATUS_CLOSED: {STATUS_WORKING},
}

# 导出用的列顺序与中文表头
CSV_COLUMNS: Sequence[tuple[str, str]] = (
    ("ticket_no", "工单号"),
    ("status", "状态"),
    ("urgency", "紧急程度"),
    ("customer_request", "客户诉求"),
    ("product_model", "产品型号"),
    ("fault_phenomenon", "故障现象"),
    ("error_info", "报错信息"),
    ("tried_actions", "已尝试操作"),
    ("suggestion", "处理建议"),
    ("citations", "资料来源"),
    ("conv_id", "会话ID"),
    ("created_at", "创建时间"),
    ("updated_at", "更新时间"),
)


class TicketService:
    """工单的读写与状态管理。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.store = ConversationStore(conn)

    # ------------------------------------------------------------------ 编号

    def next_ticket_no(self, *, now: datetime | None = None) -> str:
        """生成工单号 `T-YYYYMMDD-NNNN`（当天顺序递增）。"""
        moment = now or datetime.now(timezone.utc)
        prefix = f"T-{moment.strftime('%Y%m%d')}-"
        row = self.conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE ticket_no LIKE ?", (f"{prefix}%",)
        ).fetchone()
        return f"{prefix}{int(row[0]) + 1:04d}"

    # ------------------------------------------------------------------ 定稿

    def create_from_conversation(
        self,
        conversation_id: str,
        *,
        chat_chain: ChatChain | None = None,
        settings: Settings | None = None,
        force: bool = False,
        extractor: Callable[..., TicketExtraction] = extract_ticket,
        note: str = "",
    ) -> dict[str, Any]:
        """把一次会话定稿成工单（同一会话默认只生成一张，force=True 可再生成一张）。"""
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise LookupError("会话不存在")

        if not force:
            existing = self.conn.execute(
                "SELECT id FROM tickets WHERE conv_id = ? ORDER BY id LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if existing is not None:
                ticket = self.get(int(existing[0])) or {}
                ticket["extraction"] = {"source": "existing", "attempts": 0, "error": ""}
                return ticket

        messages = self.store.messages(conversation_id)
        draft = self.store.get_draft(conversation_id)
        extraction = extractor(
            messages, draft=draft, chat_chain=chat_chain, settings=settings
        )
        fields = dict(extraction.fields)
        citations = fields.pop("citations", None) or draft.get("citations") or []

        timestamp = now_iso()
        cursor = self.conn.execute(
            """
            INSERT INTO tickets(ticket_no, conv_id, customer_request, product_model,
                                fault_phenomenon, error_info, tried_actions, urgency,
                                suggestion, citations_json, status, source,
                                created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.next_ticket_no(),
                conversation_id,
                str(fields.get("customer_request") or ""),
                str(fields.get("product_model") or ""),
                str(fields.get("fault_phenomenon") or ""),
                str(fields.get("error_info") or ""),
                str(fields.get("tried_actions") or ""),
                str(fields.get("urgency") or "普通"),
                str(fields.get("suggestion") or ""),
                _dumps(citations),
                STATUS_PENDING,
                "auto",
                timestamp,
                timestamp,
            ),
        )
        ticket_id = int(cursor.lastrowid or 0)

        event_note = note or f"会话定稿自动生成（字段来源：{extraction.source}）"
        if extraction.error:
            event_note += f"；抽取提示：{extraction.error}"
        self._add_event(ticket_id, None, STATUS_PENDING, note=event_note)
        self.store.update(conversation_id, status="closed")
        self.conn.commit()

        ticket = self.get(ticket_id) or {}
        ticket["extraction"] = extraction.as_dict()
        logger.info(
            "工单已生成：%s（来源 %s，填充 %s/%s）",
            ticket.get("ticket_no"),
            extraction.source,
            extraction.filled[0],
            extraction.filled[1],
        )
        return ticket

    # ------------------------------------------------------------------ 查询

    def get(self, ticket_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (int(ticket_id),)
        ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_by_no(self, ticket_no: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM tickets WHERE ticket_no = ?", (ticket_no,)
        ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        conv_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """工单列表（支持状态筛选与关键字搜索）。"""
        where: list[str] = []
        params: list[Any] = []

        if status:
            where.append("status = ?")
            params.append(status)
        if conv_id:
            where.append("conv_id = ?")
            params.append(conv_id)
        if q and q.strip():
            like = f"%{q.strip()}%"
            where.append(
                "(ticket_no LIKE ? OR customer_request LIKE ? OR product_model LIKE ?"
                " OR fault_phenomenon LIKE ? OR error_info LIKE ? OR suggestion LIKE ?)"
            )
            params.extend([like] * 6)

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        total = int(
            self.conn.execute(
                f"SELECT COUNT(*) FROM tickets {clause}", params
            ).fetchone()[0]
        )
        rows = self.conn.execute(
            f"SELECT * FROM tickets {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        ).fetchall()

        return {
            "items": [self._row_to_dict(row) for row in rows],
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "stats": self.stats(),
        }

    def stats(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUS_ORDER}
        for row in self.conn.execute(
            "SELECT status, COUNT(*) FROM tickets GROUP BY status"
        ).fetchall():
            counts[str(row[0])] = int(row[1])
        counts["全部"] = sum(
            value for key, value in counts.items() if key != "全部"
        )
        return counts

    def events(self, ticket_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM ticket_events WHERE ticket_id = ? ORDER BY id", (int(ticket_id),)
        ).fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------- 状态流转

    def update_status(
        self,
        ticket_id: int,
        to_status: str,
        *,
        note: str = "",
        operator: str = "客服",
    ) -> dict[str, Any]:
        ticket = self.get(ticket_id)
        if ticket is None:
            raise LookupError("工单不存在")

        current = str(ticket["status"])
        to_status = str(to_status or "").strip()
        if to_status not in STATUS_ORDER:
            raise ValueError(f"未知状态：{to_status or '(空)'}")
        if to_status == current:
            raise ValueError(f"工单已经是「{current}」状态")
        if to_status not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(f"不允许从「{current}」直接变更为「{to_status}」")

        self.conn.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
            (to_status, now_iso(), int(ticket_id)),
        )
        self._add_event(
            ticket_id,
            current,
            to_status,
            note=note or f"{operator} 变更状态",
        )
        self.conn.commit()
        return self.get(int(ticket_id)) or {}

    def _add_event(
        self, ticket_id: int, from_status: str | None, to_status: str, *, note: str = ""
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ticket_events(ticket_id, from_status, to_status, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(ticket_id), from_status, to_status, note or None, now_iso()),
        )

    # ------------------------------------------------------------------ 导出

    def export_csv(
        self, *, status: str | None = None, q: str | None = None
    ) -> str:
        """导出 CSV 文本（带 UTF-8 BOM，Excel 直接双击不乱码）。"""
        data = self.list(status=status, q=q, limit=10000, offset=0)

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow([label for _, label in CSV_COLUMNS])
        for item in data["items"]:
            row = []
            for key, _ in CSV_COLUMNS:
                value = item.get(key)
                if isinstance(value, (list, tuple)):
                    value = "；".join(str(entry) for entry in value)
                row.append("" if value is None else value)
            writer.writerow(row)

        return "\ufeff" + buffer.getvalue()

    # ------------------------------------------------------------------ 内部

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        citations = data.pop("citations_json", None)
        if citations:
            try:
                data["citations"] = json.loads(citations)
            except (TypeError, ValueError):
                data["citations"] = []
        else:
            data["citations"] = []
        return data


def _dumps(value: Any) -> str | None:
    if not value:
        return None
    return json.dumps(value, ensure_ascii=False)
