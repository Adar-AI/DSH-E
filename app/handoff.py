"""转人工（handoff）：识别、建单提醒、客服接入与双向对话。

流程（对应需求规格 V3 的 F23-F27）：

```
客户说"转人工"/点按钮
   → HandoffService.request()：会话标记 requested + 立即建单（紧急度较急）+ 记事件 + 系统提示
   → 工单台 GET /api/handoffs 看到待接入提醒
   → 客服点击 → GET /messages 看历史 → POST /agent-messages 发言（角色 agent）
   → 首次发言自动置 claimed；人工期间后端不再调用模型（AI 静默）
```

为什么转人工要**立即建单**：工单原本是"结束会话"时才生成的，而客户要求人工时工单可能还不存在，
客服台就没有可提醒的载体。立即建单后既能在工单台看到，也能在结束会话时复用（不重复建单）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.store import ConversationStore, _loads
from app.tickets.extractor import TicketExtraction
from app.tickets.service import TicketService

logger = logging.getLogger(__name__)

STATE_NONE = "none"
STATE_REQUESTED = "requested"
STATE_CLAIMED = "claimed"
# V4：客服处理完毕后结束人工介入（会话从「转人工提醒」栏移除）
STATE_DONE = "done"
# 仍在人工环节中的状态（提醒栏只看这两个）
ACTIVE_STATES = (STATE_REQUESTED, STATE_CLAIMED)

# 客户要求人工的话术（保守：只认明确表达，避免"人工"出现在无关句子里误判）
HANDOFF_PHRASES = (
    "转人工",
    "人工客服",
    "找人工",
    "要人工",
    "人工服务",
    "转客服",
    "真人客服",
    "联系客服",
    "找客服",
    "叫客服",
    "投诉",
    "找你们领导",
    "要个说法",
)

HANDOFF_NOTICE = "已为你转接人工客服，客服会尽快在这里回复你，请稍等。（人工服务期间机器人不再自动回答）"

# 客服结束会话时同步给客户的系统提示（客户页轮询可见，状态与提示保持一致）
FINISH_NOTICE = (
    "本次人工服务已结束。如果还有其他问题，随时在这里继续提问即可（机器人会重新为你服务）。"
)

# 人工阶段新增的对话需要重新抽取进工单的字段
SYNC_FIELDS = (
    "customer_request",
    "product_model",
    "fault_phenomenon",
    "error_info",
    "tried_actions",
    "urgency",
    "suggestion",
)


@dataclass(frozen=True)
class HandoffDecision:
    """是否转人工 + 命中原因。"""

    requested: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"requested": self.requested, "reason": self.reason}


def detect_handoff_request(text: str) -> HandoffDecision:
    """从客户话术里识别"要求转人工"。"""
    content = str(text or "")
    for phrase in HANDOFF_PHRASES:
        if phrase in content:
            return HandoffDecision(True, f"话术命中「{phrase}」")
    return HandoffDecision(False)


def draft_only_extractor(
    messages: list[dict[str, Any]],
    *,
    draft: dict[str, Any] | None = None,
    chat_chain: Any | None = None,
    settings: Settings | None = None,
) -> TicketExtraction:
    """只用工单草稿建单（不调模型）。

    转人工是客户在等的操作，不能为了建单再等一次 LLM 抽取（本机约 10s）；
    草稿里已有的字段足够让客服看懂，缺的字段由"结束会话"时的正式抽取补齐。
    """
    draft = draft or {}
    fields = {
        key: value
        for key, value in draft.items()
        if key != "citations" and value not in (None, "", [], {})
    }
    if draft.get("citations"):
        fields["citations"] = draft["citations"]
    return TicketExtraction(fields=fields, source="draft", attempts=0)


class HandoffService:
    """转人工状态与建单提醒。"""

    def __init__(
        self,
        conn,
        *,
        settings: Settings | None = None,
        store: ConversationStore | None = None,
        tickets: TicketService | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.store = store or ConversationStore(conn)
        self.tickets = tickets or TicketService(conn)

    # ------------------------------------------------------------------ 查询

    def state(self, conversation_id: str) -> str:
        return self.store.handoff_state(conversation_id)

    def is_waiting_human(self, conversation_id: str) -> bool:
        """人工待接入 / 人工中：此时 AI 必须静默。"""
        return self.state(conversation_id) in {STATE_REQUESTED, STATE_CLAIMED}

    def pending(self) -> list[dict[str, Any]]:
        return self.store.pending_handoffs()

    def finished(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """最近已结束人工服务的会话（审计/回看用；提醒栏不展示）。"""
        rows = self.conn.execute(
            """
            SELECT c.*,
                   (SELECT t.ticket_no FROM tickets t WHERE t.conv_id = c.id ORDER BY t.id LIMIT 1) AS ticket_no,
                   (SELECT t.status FROM tickets t WHERE t.conv_id = c.id ORDER BY t.id LIMIT 1) AS ticket_status
              FROM conversations c
             WHERE c.handoff_state = ?
             ORDER BY c.updated_at DESC, c.id
             LIMIT ?
            """,
            (STATE_DONE, int(limit)),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["ticket_draft"] = _loads(data.pop("ticket_draft_json", None), {})
            result.append(data)
        return result

    # ------------------------------------------------------------------ 请求

    def request(
        self,
        conversation_id: str,
        *,
        reason: str = "客户点击了「转人工客服」",
        notify: str = HANDOFF_NOTICE,
    ) -> dict[str, Any]:
        """客户要求人工：标记状态 + 立即建单（紧急度较急）+ 系统提示。"""
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise LookupError("会话不存在")

        current = str(conversation.get("handoff_state") or STATE_NONE)
        if current != STATE_NONE:
            # 已经是 requested/claimed：不重复建单，也不重复提示
            return {
                "conversation_id": conversation_id,
                "handoff_state": current,
                "ticket": self._existing_ticket(conversation_id),
                "duplicated": True,
                "notice": "",
            }

        self.store.set_handoff(
            conversation_id, STATE_REQUESTED, reason=reason, at=_now()
        )
        self.store.add_message(conversation_id, "system", notify)

        ticket = self._create_ticket(conversation_id, reason=reason)
        self._record_event(ticket, reason)
        logger.info("转人工：会话 %s 已建单 %s", conversation_id, ticket.get("ticket_no"))

        return {
            "conversation_id": conversation_id,
            "handoff_state": STATE_REQUESTED,
            "ticket": ticket,
            "duplicated": False,
            "notice": notify,
        }

    # ------------------------------------------------------------------ 接入

    def claim(self, conversation_id: str) -> dict[str, Any]:
        """客服接入（首次发言时自动调用）。"""
        state = self.state(conversation_id)
        if state == STATE_NONE:
            raise LookupError("该会话没有转人工请求")
        if state == STATE_CLAIMED:
            return self.store.get(conversation_id) or {}
        return self.store.set_handoff(conversation_id, STATE_CLAIMED)

    def reply(self, conversation_id: str, text: str) -> dict[str, Any]:
        """客服发言：写入 role=agent 的消息，首次发言自动接入。"""
        content = str(text or "").strip()
        if not content:
            raise ValueError("回复内容不能为空")

        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise LookupError("会话不存在")

        claimed_now = False
        if str(conversation.get("handoff_state") or STATE_NONE) != STATE_CLAIMED:
            self.claim(conversation_id)
            claimed_now = True

        message_id = self.store.add_message(conversation_id, "agent", content)
        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "handoff_state": STATE_CLAIMED,
            "claimed_now": claimed_now,
        }

    # ------------------------------------------------------------------ 结束

    def finish(
        self,
        conversation_id: str,
        *,
        operator: str = "客服",
        chat_chain: Any | None = None,
        sync_fields: bool = True,
    ) -> dict[str, Any]:
        """客服处理完毕：结束人工介入 + 工单置已解决 + 会话结束 + 通知客户（V4）。

        * 会话从「转人工提醒」栏移除（handoff_state=done）；
        * 工单按合法流转顺序置「已解决」（待处理需先经处理中，状态机不允许跳步）；
        * **把人工阶段的对话重新抽取进工单字段**（可选，需要模型；失败不影响结束流程）；
        * 写一条系统提示给客户，并在工单时间线留痕，保证三处信息一致。
        """
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise LookupError("会话不存在")

        state = str(conversation.get("handoff_state") or STATE_NONE)
        if state not in ACTIVE_STATES:
            raise LookupError("该会话没有进行中的人工服务")

        ticket = self._existing_ticket(conversation_id)
        extraction: dict[str, Any] | None = None
        if ticket and sync_fields:
            extraction = self._sync_ticket_fields(
                ticket, conversation_id, operator=operator, chat_chain=chat_chain
            )
        if ticket:
            ticket = self._resolve_ticket(ticket, operator=operator)

        self.store.set_handoff(
            conversation_id, STATE_DONE, reason=f"{operator}结束人工服务"
        )
        self.store.update(conversation_id, status="closed")
        self.store.add_message(conversation_id, "system", FINISH_NOTICE)
        logger.info(
            "结束人工服务：会话 %s（工单 %s）",
            conversation_id,
            (ticket or {}).get("ticket_no"),
        )

        return {
            "conversation_id": conversation_id,
            "handoff_state": STATE_DONE,
            "conversation_status": "closed",
            "ticket": ticket,
            "extraction": extraction,
            "notice": FINISH_NOTICE,
        }

    # ------------------------------------------------------------------ 清理

    def clear_active(
        self, conversation_id: str, *, reason: str = "会话已结束，自动清理人工提醒"
    ) -> bool:
        """会话被结束（定稿）时自动清理人工提醒。

        为什么需要：客户在等待人工时自己点了「结束会话并生成工单」，
        如果不清理，这条会话会**永远挂在转人工提醒栏**里（实测踩到过）。
        只清提醒、不改工单状态（工单仍走正常定稿流程）。
        """
        if self.state(conversation_id) not in ACTIVE_STATES:
            return False
        self.store.set_handoff(conversation_id, STATE_DONE, reason=reason)
        ticket = self._existing_ticket(conversation_id)
        if ticket:
            self.tickets.add_timeline_note(int(ticket["id"]), reason)
        logger.info("会话 %s 已结束，自动清理人工提醒", conversation_id)
        return True

    # ------------------------------------------------------------------ 内部

    def _sync_ticket_fields(
        self,
        ticket: dict[str, Any],
        conversation_id: str,
        *,
        operator: str,
        chat_chain: Any | None,
    ) -> dict[str, Any]:
        """把整段对话（含人工阶段）重新抽取到工单字段。

        抽取失败不阻断结束流程：只记一条时间线说明，客服仍可手动补字段。
        """
        from app.tickets.extractor import extract_ticket

        try:
            messages = self.store.messages(conversation_id)
            draft = self.store.get_draft(conversation_id)
            result = extract_ticket(
                messages, draft=draft, chat_chain=chat_chain, settings=self.settings
            )
        except Exception as exc:  # noqa: BLE001 - 抽取失败不能阻断结束
            logger.warning("结束会话时同步工单字段失败：%s", exc)
            self.tickets.add_timeline_note(
                int(ticket["id"]), f"人工阶段字段同步失败：{exc}"
            )
            return {"source": "error", "error": str(exc), "updated": []}

        fields = {
            key: value
            for key, value in (result.fields or {}).items()
            if key in SYNC_FIELDS and value not in (None, "", [], {})
        }

        current = self.tickets.get(int(ticket["id"])) or {}
        changed = {
            key: value
            for key, value in fields.items()
            if str(current.get(key) or "") != str(value)
        }
        if changed:
            self.tickets.update_fields(
                int(ticket["id"]),
                changed,
                note="根据人工服务阶段的对话同步",
                operator=operator,
            )
        return {
            "source": result.source,
            "attempts": result.attempts,
            "error": result.error,
            "updated": sorted(changed),
        }

    def _resolve_ticket(self, ticket: dict[str, Any], *, operator: str) -> dict[str, Any]:
        """把工单置「已解决」，遵守状态机的合法路径（待处理必须先到处理中）。"""
        from app.tickets.service import (
            STATUS_PENDING,
            STATUS_RESOLVED,
            STATUS_WORKING,
        )

        current = str(ticket.get("status") or "")
        if current == STATUS_RESOLVED:
            self.tickets.add_timeline_note(
                int(ticket["id"]), f"{operator}结束人工服务（工单已是已解决）"
            )
            return self.tickets.get(int(ticket["id"])) or ticket

        if current == STATUS_PENDING:
            self.tickets.update_status(
                int(ticket["id"]),
                STATUS_WORKING,
                note="客服接入处理",
                operator=operator,
            )
        if current in {STATUS_PENDING, STATUS_WORKING}:
            return self.tickets.update_status(
                int(ticket["id"]),
                STATUS_RESOLVED,
                note="人工服务完成，会话结束",
                operator=operator,
            )

        # 已关闭等其它状态：不擅自重开，只留痕
        self.tickets.add_timeline_note(
            int(ticket["id"]), f"{operator}结束人工服务（工单状态：{current}，未改动）"
        )
        return self.tickets.get(int(ticket["id"])) or ticket

    # ------------------------------------------------------------------ 内部

    def _create_ticket(self, conversation_id: str, *, reason: str) -> dict[str, Any]:
        ticket = self.tickets.create_from_conversation(
            conversation_id,
            settings=self.settings,
            extractor=draft_only_extractor,
            note="",  # 建单事件由 create_from_conversation 记；转人工事件单独记一条，避免重复
        )
        # 转人工的工单要更醒目：紧急度至少"较急"，状态进入"处理中"前的"待处理"保持可筛选
        if str(ticket.get("urgency") or "") == "普通":
            self.conn.execute(
                "UPDATE tickets SET urgency = ?, updated_at = ? WHERE id = ?",
                ("较急", _now(), ticket["id"]),
            )
            self.conn.commit()
            ticket = self.tickets.get(int(ticket["id"])) or ticket
        return ticket

    def _record_event(self, ticket: dict[str, Any], reason: str) -> None:
        ticket_id = ticket.get("id")
        if not ticket_id:
            return
        self.tickets.add_timeline_note(
            int(ticket_id),
            f"客户请求转人工（{reason}），请在工单台接入会话",
        )

    def _existing_ticket(self, conversation_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id FROM tickets WHERE conv_id = ? ORDER BY id LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return self.tickets.get(int(row["id"])) if row else None


def _now() -> str:
    from app.utils import now_iso

    return now_iso()


__all__ = [
    "ACTIVE_STATES",
    "FINISH_NOTICE",
    "HANDOFF_NOTICE",
    "HANDOFF_PHRASES",
    "HandoffDecision",
    "HandoffService",
    "STATE_CLAIMED",
    "STATE_DONE",
    "STATE_NONE",
    "STATE_REQUESTED",
    "SYNC_FIELDS",
    "detect_handoff_request",
    "draft_only_extractor",
]
