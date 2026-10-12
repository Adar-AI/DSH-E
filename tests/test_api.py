"""HTTP 接口测试：SSE 事件序列、会话接口、图片上传、健康检查（全程离线打桩）。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import db as dbmod
from app import deps as deps_module
from app.main import create_app
from app.store import ConversationStore
from fakes import FakeChatChain, HashEmbedder, StubVisionProvider, load_corpus, write_png

E05_TEXT = (
    "屏幕出现 E05 表示硒鼓未被识别，请重新安装硒鼓并确认芯片触点清洁；"
    "若仍报错说明硒鼓芯片损坏，需要更换新的硒鼓。"
)
WARRANTY_TEXT = "本机型整机保修期为两年，自购买之日起计算，人为损坏不在保修范围内。"


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成 [(事件名, 数据), ...]。"""
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((name, json.loads(data)))
    return events


def event_names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


def event_data(events: list[tuple[str, dict]], name: str) -> dict:
    for event_name, data in events:
        if event_name == name:
            return data
    raise AssertionError(f"没有找到事件 {name}，实际事件：{event_names(events)}")


@pytest.fixture()
def api_client(api_env, monkeypatch):
    """注入假模型依赖的测试客户端。"""
    from app.api import health as health_module

    chat_chain = FakeChatChain(text="请先检查硒鼓芯片触点 [1]，若仍报错请联系售后。")
    embedder = HashEmbedder(dim=64)
    vision = StubVisionProvider()

    # build_deps 走 app.deps 里的名字
    monkeypatch.setattr(deps_module, "get_chat_chain", lambda settings=None: chat_chain)
    monkeypatch.setattr(deps_module, "get_embedder", lambda settings=None: embedder)
    monkeypatch.setattr(deps_module, "get_vision_provider", lambda settings=None: vision)
    # health.py 用的是导入时绑定的名字，需要单独替换
    monkeypatch.setattr(health_module, "get_chat_chain", lambda settings=None: chat_chain)
    monkeypatch.setattr(health_module, "get_vision_provider", lambda settings=None: vision)

    app = create_app(api_env, warmup_enabled=False)
    with TestClient(app) as client:
        yield client, chat_chain, vision


@pytest.fixture()
def seeded(api_env):
    """把两份资料塞进接口用的数据库。"""
    conn = dbmod.connect(settings=api_env)
    try:
        dbmod.init_db(conn, settings=api_env)
        load_corpus(
            conn,
            [
                ("打印机产品手册.md", E05_TEXT, "报错处理", 12),
                ("售后服务规则.md", WARRANTY_TEXT, "保修政策", None),
            ],
            HashEmbedder(dim=64),
        )
    finally:
        conn.close()
    return api_env


# ------------------------------------------------------------------ 健康检查

def test_health_reports_channels_without_leaking_keys(api_client):
    client, _, _ = api_client

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["db"]["schema_version"] >= 1
    assert payload["knowledge"]["documents"] == 0
    assert payload["retrieval"]["mode"] == "keyword_only"
    assert payload["chat"]["channels"][0]["name"] == "fake"
    assert payload["vision"]["model"] == "stub-vision-model"

    body = response.text
    assert "sk-" not in body, "健康检查响应里不得出现密钥明文"
    if payload["vision"]["api_key"]:
        assert payload["vision"]["api_key"].startswith("*")


# ------------------------------------------------------------------ 会话接口

def test_create_list_and_delete_conversation(api_client):
    client, _, _ = api_client

    created = client.post("/api/conversations", params={"title": "打印机问题"}).json()
    assert created["title"] == "打印机问题"

    listed = client.get("/api/conversations").json()["items"]
    assert [item["id"] for item in listed] == [created["id"]]

    detail = client.get(f"/api/conversations/{created['id']}").json()
    assert detail["message_count"] == 0

    assert client.delete(f"/api/conversations/{created['id']}").json()["deleted"] is True
    assert client.get(f"/api/conversations/{created['id']}").status_code == 404


def test_unknown_conversation_returns_404(api_client):
    client, _, _ = api_client

    assert client.get("/api/conversations/conv-nope").status_code == 404
    assert client.get("/api/conversations/conv-nope/messages").status_code == 404


# ------------------------------------------------------------------ 对话 SSE

def test_chat_hit_path_streams_events_and_persists(api_client, seeded):
    client, chain, _ = api_client

    response = client.post(
        "/api/chat", data={"text": "我的打印机报错E05，硒鼓怎么更换"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = event_names(events)

    assert names[0] == "meta"
    assert "retrieval" in names and "token" in names and "done" in names
    assert names.index("meta") < names.index("retrieval") < names.index("done")

    meta = event_data(events, "meta")
    assert meta["conversation_id"].startswith("conv-")
    assert meta["title"] == "我的打印机报错E05，硒鼓怎么更换"[:24]

    retrieval = event_data(events, "retrieval")
    assert retrieval["hit"] is True
    assert retrieval["chunks"][0]["citation"].startswith("打印机产品手册.md")

    done = event_data(events, "done")
    assert done["answer"] == chain.text
    assert done["citations"] == ["打印机产品手册.md · 第 12 页 · 报错处理"]
    assert done["ticket_draft"]["error_info"] == "E05"
    assert done["trace"]["provider"] == "fake"
    assert done["trace"]["hit"] is True

    streamed = "".join(
        data["text"] for name, data in events if name == "token"
    )
    assert streamed == chain.text, "token 事件拼起来应等于最终回答"

    # 落库校验
    conv_id = meta["conversation_id"]
    messages = client.get(f"/api/conversations/{conv_id}/messages").json()["items"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["citations"] == done["citations"]
    assert messages[1]["trace"]["hit"] is True
    conversation = client.get(f"/api/conversations/{conv_id}").json()
    assert conversation["ticket_draft"]["error_info"] == "E05"


def test_chat_fallback_path_for_unrelated_question(api_client, seeded):
    client, chain, _ = api_client

    response = client.post("/api/chat", data={"text": "预订国际航班机票价格"})
    events = parse_sse(response.text)

    retrieval = event_data(events, "retrieval")
    assert retrieval["hit"] is False

    done = event_data(events, "done")
    assert done["trace"]["fallback_used"] is True
    assert done["citations"] == []
    assert "内部资料中未收录" in done["answer"]
    # 兜底话术逐字来自模板，不经过模型
    assert chain.calls == []


def test_chat_second_turn_reuses_conversation(api_client, seeded):
    client, _, _ = api_client

    first = parse_sse(client.post("/api/chat", data={"text": "报错E05怎么办"}).text)
    conv_id = event_data(first, "meta")["conversation_id"]

    second = parse_sse(
        client.post("/api/chat", data={"text": "保修期多久", "conversation_id": conv_id}).text
    )
    assert event_data(second, "meta")["conversation_id"] == conv_id

    conversations = client.get("/api/conversations").json()["items"]
    assert len(conversations) == 1
    messages = client.get(f"/api/conversations/{conv_id}/messages").json()["items"]
    assert len(messages) == 4
    assert messages[2]["text"] == "保修期多久"


def test_chat_rejects_empty_input(api_client):
    client, _, _ = api_client

    response = client.post("/api/chat", data={"text": "   "})

    assert response.status_code == 400
    assert "请提供文字问题或上传图片" in response.json()["detail"]


def test_chat_with_image_saves_file_and_runs_vision(api_client, seeded, api_env, work_dir):
    client, _, vision = api_client
    image_path = write_png(work_dir, "screenshot.png")
    payload = image_path.read_bytes()

    response = client.post(
        "/api/chat",
        data={"text": "打印不出来了，屏幕上有个提示"},
        files={"image": ("screenshot.png", payload, "image/png")},
    )

    events = parse_sse(response.text)
    vision_event = event_data(events, "vision")
    assert vision_event["ok"] is True
    assert vision_event["detail"]["error_codes"] == ["E05"]
    assert len(vision.calls) == 1

    done = event_data(events, "done")
    assert done["trace"]["vision_ok"] is True
    assert done["ticket_draft"]["error_info"] == "E05"

    # 图片应落盘到 upload_dir，并在消息里记录路径
    saved = list(api_env.upload_path.glob("img-*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == payload
    conv_id = event_data(events, "meta")["conversation_id"]
    messages = client.get(f"/api/conversations/{conv_id}/messages").json()["items"]
    assert messages[0]["image_path"] == str(saved[0])


def test_chat_rejects_unsupported_image_type(api_client):
    client, _, _ = api_client

    response = client.post(
        "/api/chat",
        data={"text": "看下这个"},
        files={"image": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "不支持的图片类型" in response.json()["detail"]


def test_chat_reports_error_event_when_graph_breaks(api_client, seeded, monkeypatch):
    client, _, _ = api_client
    from app.api import chat as chat_api

    def boom(*args, **kwargs):
        raise RuntimeError("模拟图构建失败")

    monkeypatch.setattr(chat_api, "build_graph", boom)

    response = client.post("/api/chat", data={"text": "报错E05怎么办"})
    events = parse_sse(response.text)

    assert "error" in event_names(events)
    assert "模拟图构建失败" in event_data(events, "error")["message"]


def test_chat_falls_back_when_model_down(api_client, seeded):
    client, chain, _ = api_client
    chain.error = "所有对话通道都不可用"

    response = client.post("/api/chat", data={"text": "预订国际航班机票价格"})
    events = parse_sse(response.text)

    done = event_data(events, "done")
    assert "内部资料中未收录" in done["answer"]


def test_health_probe_option_calls_channels(api_client):
    client, _, _ = api_client

    payload = client.get("/api/health", params={"probe": "true"}).json()

    channels = payload["chat"]["channels"]
    assert channels and channels[0]["reachable"] is True
    assert channels[0]["probe_detail"] == "连通"


def test_static_index_is_served(api_client):
    client, _, _ = api_client

    response = client.get("/")

    assert response.status_code == 200
    assert "多模态企业智能客服" in response.text
