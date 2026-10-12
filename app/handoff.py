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
from app.store import ConversationStore
from app.tickets.extractor import TicketExtraction
from app.tickets.service import TicketService

logger = logging.getLogger(__name__)

STATE_NONE = "none"
STATE_REQUESTED = "requested"
STATE_CLAIMED = "claimed"

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
    "HANDOFF_NOTICE",
    "HANDOFF_PHRASES",
    "HandoffDecision",
    "HandoffService",
    "STATE_CLAIMED",
    "STATE_NONE",
    "STATE_REQUESTED",
    "detect_handoff_request",
    "draft_only_extractor",
]
