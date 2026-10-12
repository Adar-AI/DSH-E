"""转人工测试：话术识别、迁移、建单提醒、客服接入、AI 静默、增量消息。"""

from __future__ import annotations

import sqlite3

import pytest

from app import db as dbmod
from app.config import Settings
from app.handoff import (
    HANDOFF_NOTICE,
    STATE_CLAIMED,
    STATE_NONE,
    STATE_REQUESTED,
    HandoffService,
    detect_handoff_request,
    draft_only_extractor,
)
from app.store import ConversationStore
from app.tickets.service import TicketService


@pytest.fixture()
def service(conn, settings: Settings) -> HandoffService:
    return HandoffService(conn, settings=settings)


@pytest.fixture()
def conversation(conn) -> str:
    store = ConversationStore(conn)
    conv_id = store.create(title="转人工测试")["id"]
    store.add_message(conv_id, "user", "我的打印机报错了")
    return conv_id


# ------------------------------------------------------------------ 话术识别

@pytest.mark.parametrize(
    "text",
    ["我要转人工", "请转人工客服", "能不能找人工", "我要投诉", "转客服"],
)
def test_detect_handoff_positive(text):
    decision = detect_handoff_request(text)
    assert decision.requested is True
    assert "话术命中" in decision.reason


@pytest.mark.parametrize(
    "text",
    ["打印机报错E05怎么处理", "保修期多久", "人工湖怎么走", "", "谢谢"],
)
def test_detect_handoff_negative(text):
    assert detect_handoff_request(text).requested is False


def test_detect_handoff_reports_reason():
    assert "「转人工」" in detect_handoff_request("我要转人工").reason


# ------------------------------------------------------------------ 迁移

def test_migration_adds_handoff_columns_to_old_db(work_dir):
    """老库（v1 建表语句）没有 handoff 列，migrate 必须补上，且可重复执行。"""
    legacy = work_dir / "legacy.db"
    raw = sqlite3.connect(legacy)
    raw.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            ticket_draft_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    raw.commit()

    applied = dbmod.migrate(raw)
    columns = {row[1] for row in raw.execute("PRAGMA table_info(conversations)")}

    assert {"handoff_state", "handoff_reason", "handoff_at"} <= columns
    assert set(applied) == {
        "conversations.handoff_state",
        "conversations.handoff_reason",
        "conversations.handoff_at",
    }
    assert dbmod.migrate(raw) == [], "第二次执行不应重复迁移"
    raw.close()


def test_init_db_sets_schema_version(conn):
    assert dbmod.schema_version(conn) >= 2


def test_new_conversation_defaults_to_none(conn, conversation):
    assert ConversationStore(conn).handoff_state(conversation) == STATE_NONE


# ------------------------------------------------------------------ 请求转人工

def test_request_sets_state_ticket_notice_and_event(conn, service, conversation):
    outcome = service.request(conversation, reason="话术命中「转人工」")

    assert outcome["handoff_state"] == STATE_REQUESTED
    assert outcome["duplicated"] is False
    ticket = outcome["ticket"]
    assert ticket["ticket_no"].startswith("T-")
    assert ticket["urgency"] == "较急", "转人工工单要更醒目"
    assert ticket["conv_id"] == conversation

    # 客户能看到系统提示
    roles = [(item["role"], item["text"]) for item in ConversationStore(conn).messages(conversation)]
    assert ("system", HANDOFF_NOTICE) in roles

    # 工单时间线里有一条转人工说明
    events = TicketService(conn).events(int(ticket["id"]))
    assert any("转人工" in (event.get("note") or "") for event in events)

    assert service.is_waiting_human(conversation) is True


def test_request_is_idempotent(conn, service, conversation):
    first = service.request(conversation)
    second = service.request(conversation)

    assert second["duplicated"] is True
    assert second["ticket"]["ticket_no"] == first["ticket"]["ticket_no"]
    assert second["notice"] == "", "重复请求不应再插一条系统提示"

    count = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE conv_id = ?", (conversation,)
    ).fetchone()[0]
    assert count == 1, "同一会话只应有一张工单"


def test_request_unknown_conversation(service):
    with pytest.raises(LookupError):
        service.request("conv-does-not-exist")


def test_pending_handoffs_lists_waiting_conversation(service, conversation):
    service.request(conversation)

    items = service.pending()

    assert len(items) == 1
    assert items[0]["id"] == conversation
    assert items[0]["handoff_state"] == STATE_REQUESTED
    assert items[0]["ticket_no"].startswith("T-")
    assert items[0]["waiting_seconds"] >= 0
    assert items[0]["message_count"] >= 2


# ------------------------------------------------------------------ 客服接入

def test_reply_claims_and_stores_agent_message(conn, service, conversation):
    service.request(conversation)

    result = service.reply(conversation, "我是客服小李，请问型号是？")

    assert result["handoff_state"] == STATE_CLAIMED
    assert result["claimed_now"] is True
    messages = ConversationStore(conn).messages(conversation)
    assert messages[-1]["role"] == "agent"
    assert messages[-1]["text"] == "我是客服小李，请问型号是？"
    assert service.state(conversation) == STATE_CLAIMED


def test_reply_without_handoff_raises(service, conversation):
    with pytest.raises(LookupError):
        service.reply(conversation, "你好")


def test_reply_rejects_empty_text(service, conversation):
    service.request(conversation)
    with pytest.raises(ValueError):
        service.reply(conversation, "   ")


def test_pending_excludes_claimed_from_waiting_count(service, conversation):
    service.request(conversation)
    service.reply(conversation, "客服已接入")

    items = service.pending()
    assert [item["handoff_state"] for item in items] == [STATE_CLAIMED]


# ------------------------------------------------------------------ 草稿建单

def test_draft_only_extractor_does_not_call_model():
    calls: list[object] = []

    class Boom:
        def chat(self, *args, **kwargs):  # pragma: no cover - 不应被调用
            calls.append(args)
            raise AssertionError("建单不应调用模型")

    extraction = draft_only_extractor(
        [{"role": "user", "text": "L3250 报错 E05"}],
        draft={"customer_request": "报错", "product_model": "L3250", "citations": ["手册.md"]},
        chat_chain=Boom(),
    )

    assert extraction.source == "draft"
    assert extraction.fields["product_model"] == "L3250"
    assert extraction.fields["citations"] == ["手册.md"]
    assert calls == []


def test_draft_only_extractor_skips_empty_values():
    extraction = draft_only_extractor([], draft={"product_model": "L3250", "error_info": ""})
    assert "error_info" not in extraction.fields


# ------------------------------------------------------------------ 增量消息

def test_messages_after_id_returns_only_new(conn, service, conversation):
    store = ConversationStore(conn)
    service.request(conversation)
    service.reply(conversation, "客服回复")
    all_messages = store.messages(conversation)

    incremental = store.messages(conversation, after_id=all_messages[0]["id"])

    assert len(incremental) == len(all_messages) - 1
    assert incremental[-1]["role"] == "agent"


def test_messages_after_id_with_limit_keeps_order(conn, conversation):
    store = ConversationStore(conn)
    for index in range(5):
        store.add_message(conversation, "user", f"第{index}条")

    result = store.messages(conversation, after_id=0, limit=3)

    assert [item["text"] for item in result] == ["我的打印机报错了", "第0条", "第1条"]
