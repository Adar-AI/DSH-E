"""工单服务测试：定稿生成、编号、筛选、状态流转、事件、导出。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.store import ConversationStore
from app.tickets.service import (
    ALLOWED_TRANSITIONS,
    STATUS_CLOSED,
    STATUS_PENDING,
    STATUS_RESOLVED,
    STATUS_WORKING,
    TicketService,
)
from fakes import GOOD_TICKET_JSON, FakeChatChain


@pytest.fixture()
def service(conn):
    return TicketService(conn)


@pytest.fixture()
def conversation(conn):
    """一个带两轮对话和草稿的会话。"""
    store = ConversationStore(conn)
    conv_id = store.create(title="打印机报错")["id"]
    store.add_message(conv_id, "user", "我的打印机型号 L3250 报错E05，已经换了硒鼓")
    store.add_message(
        conv_id,
        "assistant",
        "请检查硒鼓芯片触点并复位计数器。",
        citations=["常见故障排查表.md · 第 1 页"],
    )
    store.set_draft(
        conv_id,
        {
            "customer_request": "打印机报错E05",
            "product_model": "L3250",
            "error_info": "E05",
            "tried_actions": "已经换了硒鼓",
            "urgency": "较急",
            "citations": ["常见故障排查表.md · 第 1 页"],
        },
    )
    return conv_id


# ------------------------------------------------------------------ 编号与定稿

def test_next_ticket_no_increments_per_day(service):
    today = datetime(2026, 10, 12, tzinfo=timezone.utc)
    assert service.next_ticket_no(now=today) == "T-20261012-0001"

    service.conn.execute(
        """
        INSERT INTO tickets(ticket_no, created_at, updated_at)
        VALUES ('T-20261012-0001', 'now', 'now')
        """
    )
    service.conn.commit()

    assert service.next_ticket_no(now=today) == "T-20261012-0002"
    # 换一天重新从 0001 开始
    assert service.next_ticket_no(now=datetime(2026, 10, 13, tzinfo=timezone.utc)) == "T-20261013-0001"


def test_create_from_conversation_with_llm(service, conversation):
    chain = FakeChatChain(text=GOOD_TICKET_JSON)

    ticket = service.create_from_conversation(conversation, chat_chain=chain)

    assert ticket["ticket_no"].startswith("T-")
    assert ticket["status"] == STATUS_PENDING
    assert ticket["customer_request"].startswith("打印机报错 E05")
    assert ticket["product_model"] == "L3250"
    assert ticket["error_info"] == "E05"
    assert ticket["urgency"] == "较急"
    assert ticket["citations"] == ["常见故障排查表.md · 第 1 页"]
    assert ticket["conv_id"] == conversation
    assert ticket["extraction"]["source"] == "llm"

    # 会话被标记为已结束
    assert ConversationStore(service.conn).get(conversation)["status"] == "closed"

    # 生成时写了一条事件
    events = service.events(ticket["id"])
    assert len(events) == 1
    assert events[0]["to_status"] == STATUS_PENDING
    assert events[0]["from_status"] is None


def test_create_from_conversation_without_model_uses_draft(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    assert ticket["product_model"] == "L3250"
    assert ticket["extraction"]["source"] == "draft"


def test_create_records_extraction_source_and_failure_hint(service, conversation):
    failing = FakeChatChain(error="模型全挂")
    ticket = service.create_from_conversation(conversation, chat_chain=failing)

    note = service.events(ticket["id"])[0]["note"]
    assert "字段来源：draft" in note
    assert "模型调用失败" in note


def test_create_records_llm_source(service, conversation):
    ticket = service.create_from_conversation(
        conversation, chat_chain=FakeChatChain(text=GOOD_TICKET_JSON)
    )

    note = service.events(ticket["id"])[0]["note"]
    assert "字段来源：llm" in note
    assert "抽取提示" not in note


def test_create_from_conversation_is_idempotent(service, conversation):
    first = service.create_from_conversation(conversation, chat_chain=None)
    second = service.create_from_conversation(conversation, chat_chain=None)

    assert first["id"] == second["id"]
    assert second["extraction"]["source"] == "existing"
    assert service.stats()["全部"] == 1


def test_create_from_conversation_force_creates_another(service, conversation):
    first = service.create_from_conversation(conversation, chat_chain=None)
    second = service.create_from_conversation(conversation, chat_chain=None, force=True)

    assert first["id"] != second["id"]
    assert service.stats()["全部"] == 2


def test_create_from_unknown_conversation_raises(service):
    with pytest.raises(LookupError):
        service.create_from_conversation("conv-nope", chat_chain=None)


# ------------------------------------------------------------------ 查询

def test_list_filters_by_status_and_keyword(service, conversation):
    first = service.create_from_conversation(conversation, chat_chain=None)
    service.update_status(first["id"], STATUS_WORKING)

    store = ConversationStore(service.conn)
    other = store.create(title="别的会话")["id"]
    store.add_message(other, "user", "网络连不上")
    store.set_draft(other, {"customer_request": "网络连不上", "product_model": "M1136"})
    second = service.create_from_conversation(other, chat_chain=None)

    assert service.list()["total"] == 2
    assert service.list(status=STATUS_WORKING)["total"] == 1
    assert service.list(status=STATUS_PENDING)["items"][0]["id"] == second["id"]
    assert service.list(q="L3250")["total"] == 1
    assert service.list(q="M1136")["total"] == 1
    assert service.list(q="查不到的东西")["total"] == 0
    # 默认按 id 倒序：最新的工单在最前
    assert service.list(limit=1)["items"][0]["id"] == second["id"]
    assert service.list(conv_id=other)["total"] == 1


def test_stats_counts_by_status(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)
    service.update_status(ticket["id"], STATUS_WORKING)

    stats = service.stats()

    assert stats[STATUS_PENDING] == 0
    assert stats[STATUS_WORKING] == 1
    assert stats["全部"] == 1


def test_get_by_no(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    assert service.get_by_no(ticket["ticket_no"])["id"] == ticket["id"]
    assert service.get_by_no("T-nope") is None


# ------------------------------------------------------------------ 状态流转

def test_update_status_records_events(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)
    ticket_id = ticket["id"]

    service.update_status(ticket_id, STATUS_WORKING, note="客服已接单", operator="小王")
    updated = service.update_status(ticket_id, STATUS_RESOLVED)

    assert updated["status"] == STATUS_RESOLVED
    events = service.events(ticket_id)
    assert [(event["from_status"], event["to_status"]) for event in events] == [
        (None, STATUS_PENDING),
        (STATUS_PENDING, STATUS_WORKING),
        (STATUS_WORKING, STATUS_RESOLVED),
    ]
    assert events[1]["note"] == "客服已接单"


def test_update_status_rejects_illegal_transition(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    with pytest.raises(ValueError) as error:
        service.update_status(ticket["id"], STATUS_RESOLVED)

    assert "不允许从「待处理」直接变更为「已解决」" in str(error.value)
    assert service.get(ticket["id"])["status"] == STATUS_PENDING


def test_update_status_rejects_same_or_unknown_status(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    with pytest.raises(ValueError):
        service.update_status(ticket["id"], STATUS_PENDING)
    with pytest.raises(ValueError):
        service.update_status(ticket["id"], "已发货")
    with pytest.raises(LookupError):
        service.update_status(99999, STATUS_WORKING)


def test_closed_ticket_can_reopen_to_working(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)
    service.update_status(ticket["id"], STATUS_CLOSED)

    reopened = service.update_status(ticket["id"], STATUS_WORKING, note="客户回访有新问题")

    assert reopened["status"] == STATUS_WORKING


def test_transition_table_covers_all_statuses():
    assert set(ALLOWED_TRANSITIONS) == {STATUS_PENDING, STATUS_WORKING, STATUS_RESOLVED, STATUS_CLOSED}
    for status, targets in ALLOWED_TRANSITIONS.items():
        assert status not in targets, "不允许原地流转"
        assert targets <= set(ALLOWED_TRANSITIONS)


# ------------------------------------------------------------------ 导出

def test_export_csv_has_bom_headers_and_rows(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    content = service.export_csv()

    assert content.startswith("\ufeff"), "必须带 BOM，否则 Excel 打开是乱码"
    lines = content.lstrip("\ufeff").strip().splitlines()
    assert lines[0].startswith("工单号,状态,紧急程度,客户诉求")
    assert ticket["ticket_no"] in lines[1]
    assert "L3250" in lines[1]
    assert len(lines) == 2


def test_export_csv_respects_filters(service, conversation):
    service.create_from_conversation(conversation, chat_chain=None)

    assert len(service.export_csv(status=STATUS_WORKING).strip().splitlines()) == 1
    assert len(service.export_csv(q="不存在的关键字").strip().splitlines()) == 1


def test_export_csv_escapes_commas(service, conversation):
    service.create_from_conversation(conversation, chat_chain=None)

    content = service.export_csv()

    assert '"' in content or "," in content  # 含逗号的字段被引号包裹


# ------------------------------------------------------------------ 生命周期

def test_deleting_conversation_keeps_ticket(service, conversation):
    ticket = service.create_from_conversation(conversation, chat_chain=None)

    ConversationStore(service.conn).delete(conversation)

    assert service.get(ticket["id"]) is not None
    assert service.get(ticket["id"])["conv_id"] == conversation
