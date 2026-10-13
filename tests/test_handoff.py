"""转人工测试：话术识别、迁移、建单提醒、客服接入、AI 静默、增量消息。"""

from __future__ import annotations

import sqlite3

import pytest

from app import db as dbmod
from app.config import Settings
from app.handoff import (
    HANDOFF_NOTICE,
    STATE_DONE,
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
    assert {
        "conversations.handoff_state",
        "conversations.handoff_reason",
        "conversations.handoff_at",
    } <= set(applied)
    assert any("idx_conversations_handoff" in item for item in applied)
    assert dbmod.migrate(raw) == [], "第二次执行不应重复迁移"
    raw.close()


def test_init_db_upgrades_legacy_database_without_crashing(work_dir):
    """回归：老库直接跑 init_db 也必须成功。

    真机踩过的坑：把 handoff_state 的索引写在 schema.sql 里，
    老库的 CREATE TABLE IF NOT EXISTS 是空操作（没有该列），建索引直接报
    "no such column: handoff_state"，服务启动即失败。
    """
    legacy = work_dir / "legacy_init.db"
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
        INSERT INTO conversations(id, title, status, created_at, updated_at)
        VALUES ('conv-old', '老会话', 'active', '2026-01-01T00:00:00', '2026-01-01T00:00:00');
        """
    )
    raw.commit()

    settings = Settings(
        _env_file=None,
        db_path=legacy,
        upload_dir=work_dir / "u",
        docs_dir=work_dir / "d",
        embedding_dim=64,
    )
    dbmod.init_db(raw, settings=settings)  # 不应抛异常

    columns = {row[1] for row in raw.execute("PRAGMA table_info(conversations)")}
    assert "handoff_state" in columns
    indexes = {row[1] for row in raw.execute("PRAGMA index_list(conversations)")}
    assert "idx_conversations_handoff" in indexes
    # 老数据仍在，且默认状态正确
    row = raw.execute("SELECT title, handoff_state FROM conversations WHERE id = 'conv-old'").fetchone()
    assert tuple(row) == ("老会话", "none")
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


# ---------------------------------------------------------------- 结束会话（V4）

TICKET_JSON = """{
  "customer_request": "L3250 报错 E05",
  "product_model": "L3250",
  "fault_phenomenon": "屏幕显示 E05",
  "error_info": "E05",
  "tried_actions": "已重启",
  "urgency": "较急",
  "suggestion": "检查硒鼓芯片触点"
}"""


def test_finish_moves_out_of_bar_and_closes_everything(conn, service, conversation):
    """V4 核心：结束人工服务后从提醒栏移除，工单置已解决，会话结束，客户收到提示。"""
    service.request(conversation)
    service.reply(conversation, "我是客服小李，这就帮你处理")
    assert len(service.pending()) == 1

    outcome = service.finish(conversation)

    assert outcome["handoff_state"] == STATE_DONE
    assert outcome["conversation_status"] == "closed"
    assert service.pending() == [], "结束后不应再占用转人工提醒栏"
    assert service.is_waiting_human(conversation) is False

    store = ConversationStore(conn)
    assert store.handoff_state(conversation) == STATE_DONE
    assert (store.get(conversation) or {}).get("status") == "closed"

    messages = store.messages(conversation)
    assert messages[-1]["role"] == "system"
    assert "人工服务已结束" in messages[-1]["text"]

    ticket = outcome["ticket"]
    assert ticket["status"] == "已解决", "工单应同步为已解决"
    notes = [event.get("note") or "" for event in TicketService(conn).events(ticket["id"])]
    assert any("人工服务完成" in note for note in notes)


def test_finish_requires_active_handoff(service, conversation):
    with pytest.raises(LookupError):
        service.finish(conversation)


def test_finish_twice_is_rejected(conn, service, conversation):
    service.request(conversation)
    service.finish(conversation)

    with pytest.raises(LookupError):
        service.finish(conversation)


def test_finish_syncs_fields_from_agent_conversation(conn, service, conversation):
    """把人工阶段的对话重新抽取进工单字段（这里用假模型保证确定）。"""
    from fakes import FakeChatChain

    service.request(conversation)
    ConversationStore(conn).add_message(conversation, "user", "型号是 L3250，屏幕显示 E05")
    service.reply(conversation, "已帮你确认，是硒鼓未识别")

    outcome = service.finish(conversation, chat_chain=FakeChatChain(text=TICKET_JSON))

    assert outcome["extraction"]["source"] == "llm"
    assert "product_model" in outcome["extraction"]["updated"]
    assert outcome["ticket"]["product_model"] == "L3250"
    assert outcome["ticket"]["error_info"] == "E05"


def test_finish_without_field_sync_still_closes(conn, service, conversation):
    service.request(conversation)

    outcome = service.finish(conversation, sync_fields=False)

    assert outcome["extraction"] is None
    assert outcome["ticket"]["status"] == "已解决"
    assert service.pending() == []


def test_finish_degrades_to_rules_when_model_fails(conn, service, conversation):
    """模型不可用时抽取器会退化为规则抽取，结束流程照常完成。"""
    from fakes import FakeChatChain

    service.request(conversation)
    ConversationStore(conn).add_message(conversation, "user", "型号 L3250 报错 E05")

    outcome = service.finish(conversation, chat_chain=FakeChatChain(error="模型挂了"))

    assert outcome["ticket"]["status"] == "已解决"
    assert service.pending() == []
    assert outcome["extraction"]["source"] in {"draft", "rules"}, (
        f"应退化为规则抽取，实际 {outcome['extraction']['source']}"
    )


def test_finish_survives_extractor_crash(conn, service, conversation, monkeypatch):
    """抽取过程整体抛异常时也不能阻断结束：只记一条时间线说明。"""
    import app.tickets.extractor as extractor_module

    def boom(*args, **kwargs):
        raise RuntimeError("抽取器崩了")

    monkeypatch.setattr(extractor_module, "extract_ticket", boom)
    service.request(conversation)

    outcome = service.finish(conversation)

    assert outcome["ticket"]["status"] == "已解决"
    assert service.pending() == []
    assert outcome["extraction"]["source"] == "error"
    notes = [event.get("note") or "" for event in TicketService(conn).events(outcome["ticket"]["id"])]
    assert any("同步失败" in note for note in notes)


def test_finished_history_keeps_record(conn, service, conversation):
    service.request(conversation)
    service.finish(conversation, sync_fields=False)

    history = service.finished()

    assert [item["id"] for item in history] == [conversation]
    assert history[0]["handoff_state"] == STATE_DONE
    assert history[0]["ticket_status"] == "已解决"


def test_customer_can_continue_after_finish(conn, service, conversation):
    """结束后机器人恢复服务：不再是人工环节。"""
    service.request(conversation)
    service.finish(conversation, sync_fields=False)

    assert service.is_waiting_human(conversation) is False


def test_clear_active_removes_bar_entry(conn, service, conversation):
    """会话被结束时自动清理人工提醒（客户自己去点「结束会话」的情形）。"""
    service.request(conversation)
    assert len(service.pending()) == 1

    cleared = service.clear_active(conversation)

    assert cleared is True
    assert service.pending() == []
    assert service.state(conversation) == STATE_DONE
    # 不影响工单状态，只留一条时间线
    ticket = service._existing_ticket(conversation)
    notes = [event.get("note") or "" for event in service.tickets.events(int(ticket["id"]))]
    assert any("自动清理人工提醒" in note for note in notes)


def test_clear_active_noop_without_handoff(service, conversation):
    assert service.clear_active(conversation) is False


# ---------------------------------------------------------------- V4 修订

def test_cancel_removes_entry_without_touching_ticket(conn, service, conversation):
    """手动移出提醒：只清提醒，工单与消息都不动。"""
    service.request(conversation)
    ticket_before = service._existing_ticket(conversation)
    status_before = ticket_before["status"]

    outcome = service.cancel(conversation)

    assert outcome["cleared"] is True
    assert outcome["handoff_state"] == STATE_DONE
    assert service.pending() == []
    assert (service._existing_ticket(conversation) or {})["status"] == status_before
    assert ConversationStore(conn).messages(conversation)


def test_cancel_is_noop_when_not_in_bar(service, conversation):
    outcome = service.cancel(conversation)

    assert outcome["cleared"] is False
    assert "不在人工提醒栏" in outcome["reason"]


def test_cancel_unknown_conversation(service):
    with pytest.raises(LookupError):
        service.cancel("conv-missing")


def test_finish_notice_tells_customer_ai_takes_over():
    from app.handoff import FINISH_NOTICE

    assert "AI" in FINISH_NOTICE
    assert "接管" in FINISH_NOTICE
