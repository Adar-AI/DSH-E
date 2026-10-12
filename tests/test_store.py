"""会话仓储层测试：会话、消息、历史、草稿。"""

from __future__ import annotations

from app.store import ConversationStore


def test_create_get_and_list(conn):
    store = ConversationStore(conn)

    created = store.create(session_id="sess-1", title="打印机报错")
    assert created["id"].startswith("conv-")
    assert created["title"] == "打印机报错"
    assert created["status"] == "active"
    assert store.get(created["id"])["session_id"] == "sess-1"

    store.create(session_id="sess-1", title="第二个会话")
    items = store.list()
    assert len(items) == 2
    assert items[0]["message_count"] == 0


def test_ensure_creates_with_title_from_first_question(conn):
    store = ConversationStore(conn)
    question = "我的打印机报错 E05 怎么办？换了硒鼓还是不识别，麻烦尽快处理一下"

    conversation = store.ensure(None, first_text=question)

    assert conversation["title"] == question[:24]
    assert len(conversation["title"]) == 24
    assert store.get(conversation["id"]) is not None


def test_ensure_uses_default_title_for_empty_question(conn):
    store = ConversationStore(conn)

    conversation = store.ensure(None, first_text="   ")

    assert conversation["title"] == "新会话"


def test_ensure_reuses_existing_conversation(conn):
    store = ConversationStore(conn)
    first = store.ensure(None, first_text="第一次提问")

    second = store.ensure(first["id"], first_text="第二次提问")

    assert second["id"] == first["id"]
    assert len(store.list()) == 1


def test_ensure_with_unknown_id_creates_new_one(conn):
    store = ConversationStore(conn)

    conversation = store.ensure("conv-不存在", first_text="提问")

    assert conversation["id"] == "conv-不存在"
    assert store.get("conv-不存在") is not None


def test_add_message_and_read_back_json_fields(conn):
    store = ConversationStore(conn)
    conversation = store.create()
    conv_id = conversation["id"]

    store.add_message(conv_id, "user", "报错 E05", image_path="data/uploads/a.png")
    store.add_message(
        conv_id,
        "assistant",
        "请检查触点",
        vision={"detail": {"error_codes": ["E05"]}},
        citations=["手册.md · 第 12 页"],
        trace={"provider": "ollama", "hit": True},
    )

    messages = store.messages(conv_id)

    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["image_path"] == "data/uploads/a.png"
    assert messages[1]["citations"] == ["手册.md · 第 12 页"]
    assert messages[1]["vision"]["detail"]["error_codes"] == ["E05"]
    assert messages[1]["trace"]["hit"] is True
    assert store.message_count(conv_id) == 2


def test_messages_limit_keeps_latest(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]
    for index in range(5):
        store.add_message(conv_id, "user", f"第{index}句")

    messages = store.messages(conv_id, limit=2)

    assert [item["text"] for item in messages] == ["第3句", "第4句"]


def test_history_before_last_user_excludes_current_question(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]
    store.add_message(conv_id, "user", "第一轮问题")
    store.add_message(conv_id, "assistant", "第一轮回答")
    store.add_message(conv_id, "user", "第二轮问题")

    history = store.history_before_last_user(conv_id, turns=4)

    assert history == [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
    ]


def test_history_respects_turns_limit(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]
    for index in range(6):
        store.add_message(conv_id, "user", f"问题{index}")
        store.add_message(conv_id, "assistant", f"回答{index}")

    history = store.history_before_last_user(conv_id, turns=2)

    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "问题4"}


def test_history_starts_empty_for_first_turn(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]
    store.add_message(conv_id, "user", "只有这一句")

    assert store.history_before_last_user(conv_id) == []


def test_draft_roundtrip(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]

    assert store.get_draft(conv_id) == {}

    store.set_draft(conv_id, {"product_model": "L3250", "error_info": "E05"})

    assert store.get_draft(conv_id)["product_model"] == "L3250"
    assert store.get(conv_id)["ticket_draft"]["error_info"] == "E05"


def test_update_title_and_status(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]

    store.update(conv_id, title="新标题", status="closed")

    conversation = store.get(conv_id)
    assert conversation["title"] == "新标题"
    assert conversation["status"] == "closed"


def test_delete_removes_conversation_and_messages(conn):
    store = ConversationStore(conn)
    conv_id = store.create()["id"]
    store.add_message(conv_id, "user", "提问")

    assert store.delete(conv_id) is True
    assert store.get(conv_id) is None
    assert store.message_count(conv_id) == 0
    assert store.delete(conv_id) is False


def test_get_unknown_conversation_returns_none(conn):
    store = ConversationStore(conn)
    assert store.get("conv-nope") is None
    assert store.get_draft("conv-nope") == {}
