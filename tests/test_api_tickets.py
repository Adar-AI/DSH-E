"""工单接口测试：定稿、列表筛选、详情、状态流转、导出 CSV。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import deps as deps_module
from app.main import create_app
from fakes import GOOD_TICKET_JSON, FakeChatChain, HashEmbedder, StubVisionProvider


@pytest.fixture()
def client(api_env, monkeypatch):
    chain = FakeChatChain(text=GOOD_TICKET_JSON)
    monkeypatch.setattr(deps_module, "get_chat_chain", lambda settings=None: chain)
    monkeypatch.setattr(deps_module, "get_embedder", lambda settings=None: HashEmbedder(dim=64))
    monkeypatch.setattr(deps_module, "get_vision_provider", lambda settings=None: StubVisionProvider())
    # conversations.close 直接绑定了 get_chat_chain，需要单独替换
    from app.api import conversations as conversations_module

    monkeypatch.setattr(conversations_module, "get_chat_chain", lambda settings=None: chain)

    app = create_app(api_env, warmup_enabled=False)
    with TestClient(app) as test_client:
        yield test_client, chain


def seed_conversation(client: TestClient, text: str = "我的打印机型号 L3250 报错E05，已经换了硒鼓") -> str:
    """直接建会话 + 写消息，避免依赖对话链路。"""
    from app import db as dbmod
    from app.config import get_settings
    from app.store import ConversationStore

    settings = get_settings()
    conn = dbmod.connect(settings=settings)
    try:
        store = ConversationStore(conn)
        conv_id = store.create(title="打印机报错")["id"]
        store.add_message(conv_id, "user", text)
        store.add_message(
            conv_id,
            "assistant",
            "请检查硒鼓芯片触点并复位计数器。",
            citations=["常见故障排查表.md · 第 1 页"],
        )
        store.set_draft(
            conv_id,
            {
                "customer_request": text,
                "product_model": "L3250",
                "error_info": "E05",
                "urgency": "较急",
                "citations": ["常见故障排查表.md · 第 1 页"],
            },
        )
        return conv_id
    finally:
        conn.close()


# ------------------------------------------------------------------ 定稿

def test_close_conversation_creates_ticket(client):
    http, chain = client
    conv_id = seed_conversation(http)

    response = http.post(f"/api/conversations/{conv_id}/close")

    assert response.status_code == 200
    ticket = response.json()
    assert ticket["ticket_no"].startswith("T-")
    assert ticket["status"] == "待处理"
    assert ticket["product_model"] == "L3250"
    assert ticket["error_info"] == "E05"
    assert ticket["urgency"] == "较急"
    assert ticket["citations"] == ["常见故障排查表.md · 第 1 页"]
    assert ticket["extraction"]["source"] == "llm"
    assert len(ticket["events"]) == 1
    assert len(chain.calls) == 1, "定稿时应当调用一次模型做结构化抽取"

    detail = http.get(f"/api/conversations/{conv_id}").json()
    assert detail["status"] == "closed"


def test_close_conversation_is_idempotent(client):
    http, _ = client
    conv_id = seed_conversation(http)

    first = http.post(f"/api/conversations/{conv_id}/close").json()
    second = http.post(f"/api/conversations/{conv_id}/close").json()

    assert first["id"] == second["id"]
    assert http.get("/api/tickets").json()["total"] == 1

    forced = http.post(
        f"/api/conversations/{conv_id}/close", params={"force": "true"}
    ).json()
    assert forced["id"] != first["id"]
    assert http.get("/api/tickets").json()["total"] == 2


def test_close_unknown_conversation_returns_404(client):
    http, _ = client

    assert http.post("/api/conversations/conv-nope/close").status_code == 404


def test_close_falls_back_to_draft_when_model_breaks(client):
    http, chain = client
    chain.error = "模型全挂"
    conv_id = seed_conversation(http)

    ticket = http.post(f"/api/conversations/{conv_id}/close").json()

    assert ticket["extraction"]["source"] == "draft"
    assert ticket["product_model"] == "L3250", "模型挂了也要用规则草稿生成工单"
    assert "模型调用失败" in ticket["extraction"]["error"]


# ------------------------------------------------------------------ 列表与详情

def test_list_tickets_with_filters_and_stats(client):
    http, chain = client
    first = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    # 第二张工单换成另一台设备，便于验证关键字筛选
    chain.text = GOOD_TICKET_JSON.replace("L3250", "M1136").replace("E05", "U4")
    second = http.post(
        f"/api/conversations/{seed_conversation(http, '网络连不上，型号 M1136')}/close"
    ).json()

    payload = http.get("/api/tickets").json()

    assert payload["total"] == 2
    assert payload["status_options"] == ["待处理", "处理中", "已解决", "已关闭"]
    assert payload["stats"]["待处理"] == 2
    assert payload["stats"]["全部"] == 2
    assert [item["id"] for item in payload["items"]] == [second["id"], first["id"]]
    assert second["product_model"] == "M1136", "定稿应以模型的结构化结果为准"

    assert http.get("/api/tickets", params={"q": "L3250"}).json()["total"] == 1
    assert http.get("/api/tickets", params={"q": "M1136"}).json()["total"] == 1
    assert http.get("/api/tickets", params={"q": "查不到"}).json()["total"] == 0
    assert http.get("/api/tickets", params={"conv_id": first["conv_id"]}).json()["total"] == 1
    assert len(http.get("/api/tickets", params={"limit": 1}).json()["items"]) == 1


def test_get_ticket_detail_includes_events(client):
    http, _ = client
    ticket = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    detail = http.get(f"/api/tickets/{ticket['id']}").json()

    assert detail["ticket_no"] == ticket["ticket_no"]
    assert [event["to_status"] for event in detail["events"]] == ["待处理"]

    assert http.get("/api/tickets/99999").status_code == 404


# ------------------------------------------------------------------ 状态流转

def test_patch_status_transitions_and_history(client):
    http, _ = client
    ticket = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    working = http.patch(
        f"/api/tickets/{ticket['id']}",
        json={"to_status": "处理中", "note": "客服已接单", "operator": "小王"},
    )
    assert working.status_code == 200
    assert working.json()["status"] == "处理中"

    resolved = http.patch(f"/api/tickets/{ticket['id']}", json={"to_status": "已解决"})
    assert resolved.json()["status"] == "已解决"

    events = resolved.json()["events"]
    assert [(event["from_status"], event["to_status"]) for event in events] == [
        (None, "待处理"),
        ("待处理", "处理中"),
        ("处理中", "已解决"),
    ]
    assert events[1]["note"] == "客服已接单"


def test_patch_status_rejects_illegal_transition(client):
    http, _ = client
    ticket = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    response = http.patch(f"/api/tickets/{ticket['id']}", json={"to_status": "已解决"})

    assert response.status_code == 400
    assert "不允许从「待处理」直接变更为「已解决」" in response.json()["detail"]
    assert http.get(f"/api/tickets/{ticket['id']}").json()["status"] == "待处理"


def test_patch_status_unknown_ticket_returns_404(client):
    http, _ = client

    response = http.patch("/api/tickets/99999", json={"to_status": "处理中"})

    assert response.status_code == 404


def test_patch_status_requires_target(client):
    http, _ = client
    ticket = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    assert http.patch(f"/api/tickets/{ticket['id']}", json={}).status_code == 422


# ------------------------------------------------------------------ 导出

def test_export_csv(client):
    http, _ = client
    ticket = http.post(f"/api/conversations/{seed_conversation(http)}/close").json()

    response = http.get("/api/tickets/export.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    body = response.content.decode("utf-8")
    assert body.startswith("\ufeff")
    lines = body.lstrip("\ufeff").strip().splitlines()
    assert lines[0].startswith("工单号,状态,紧急程度,客户诉求")
    assert ticket["ticket_no"] in lines[1]
    assert len(lines) == 2


def test_export_csv_with_filter(client):
    http, _ = client
    http.post(f"/api/conversations/{seed_conversation(http)}/close")

    response = http.get("/api/tickets/export.csv", params={"q": "不存在的关键字"})

    assert len(response.content.decode("utf-8").lstrip("\ufeff").strip().splitlines()) == 1


def test_deleting_conversation_keeps_ticket(client):
    http, _ = client
    conv_id = seed_conversation(http)
    ticket = http.post(f"/api/conversations/{conv_id}/close").json()

    http.delete(f"/api/conversations/{conv_id}")

    assert http.get("/api/conversations").json()["items"] == []
    assert http.get(f"/api/tickets/{ticket['id']}").status_code == 200
