"""模型通道测试：payload 组装、重试、流式、降级链、配置组装（全部离线打桩）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings, is_local_url
from app.llm.providers import (
    ChatChain,
    OpenAICompatProvider,
    ProviderError,
    _extract_text,
    apply_no_think,
    build_chat_chain,
    build_vision_provider,
    build_vision_providers,
    is_thinking_model,
)
from fakes import FakeClient, FakeCompletions, FakeModels, client_factory_for

BASE_URL = "http://127.0.0.1:11434/v1"


def make_provider(**overrides):
    completions = overrides.pop("completions", None) or FakeCompletions(["回复内容"])
    models = overrides.pop("models", None)
    client = FakeClient(completions, models)
    factory = client_factory_for(client)
    provider = OpenAICompatProvider(
        name=overrides.pop("name", "fake"),
        base_url=overrides.pop("base_url", BASE_URL),
        model=overrides.pop("model", "test-model"),
        client_factory=factory,
        **overrides,
    )
    return provider, completions, factory


# ------------------------------------------------------------------ 基础

def test_available_requires_key_when_flagged():
    provider, _, _ = make_provider(requires_key=True, api_key="")
    ok, reason = provider.available()
    assert ok is False
    assert "未配置 API Key" in reason

    provider.api_key = "sk-x"  # type: ignore[misc]
    assert provider.available()[0] is True


def test_chat_builds_payload_and_reuses_client():
    provider, completions, factory = make_provider(api_key="k", timeout=12.5)

    text = provider.chat([{"role": "user", "content": "你好"}], max_tokens=8)

    assert text == "回复内容"
    payload = completions.calls[0]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 8
    assert payload["messages"] == [{"role": "user", "content": "你好"}]
    assert "stream" not in payload

    provider.chat([{"role": "user", "content": "再来一次"}])
    assert len(factory.calls) == 1, "客户端应当被缓存复用"
    assert factory.calls[0] == (BASE_URL, "k", 12.5)


def test_chat_retries_then_succeeds():
    provider, completions, _ = make_provider(completions=FakeCompletions(["成功"], fail_times=1))

    assert provider.chat([{"role": "user", "content": "hi"}]) == "成功"
    assert len(completions.calls) == 2


def test_chat_raises_provider_error_after_retries():
    provider, completions, _ = make_provider(
        name="ollama", completions=FakeCompletions(error=RuntimeError("连接被拒绝"))
    )

    with pytest.raises(ProviderError) as error:
        provider.chat([{"role": "user", "content": "hi"}])

    assert "ollama" in str(error.value)
    assert "连接被拒绝" in str(error.value)
    assert len(completions.calls) == 2  # 原始请求 + 1 次重试


def test_chat_raises_on_empty_choices():
    provider, _, _ = make_provider(completions=FakeCompletions(error=ProviderError("x")))

    with pytest.raises(ProviderError):
        provider.chat([{"role": "user", "content": "hi"}])


# ------------------------------------------------------------------ 解析

def test_extract_text_handles_content_parts():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[{"type": "text", "text": "甲"}, SimpleNamespace(text="乙")]
                )
            )
        ]
    )
    assert _extract_text(response) == "甲乙"


def test_extract_text_errors():
    with pytest.raises(ProviderError):
        _extract_text(SimpleNamespace(choices=[]))
    with pytest.raises(ProviderError):
        _extract_text(
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        )


# ------------------------------------------------------------------ 流式

def test_chat_stream_yields_pieces():
    provider, _, _ = make_provider(
        completions=FakeCompletions(stream_pieces=["硒鼓", "需要", "更换"])
    )

    assert list(provider.chat_stream([{"role": "user", "content": "hi"}])) == [
        "硒鼓",
        "需要",
        "更换",
    ]


def test_chat_stream_wraps_errors():
    provider, _, _ = make_provider(
        completions=FakeCompletions(stream_error=RuntimeError("连接中断"))
    )

    with pytest.raises(ProviderError) as error:
        list(provider.chat_stream([{"role": "user", "content": "hi"}]))
    assert "连接中断" in str(error.value)


# ------------------------------------------------------------------ 诊断

def test_list_models_and_failure():
    provider, _, _ = make_provider(models=FakeModels(["qwen3:4b", "bge-m3"]))
    assert provider.list_models() == ["qwen3:4b", "bge-m3"]

    failing, _, _ = make_provider(models=FakeModels(error=RuntimeError("401")))
    with pytest.raises(ProviderError):
        failing.list_models()


def test_ping_reports_connectivity():
    provider, _, _ = make_provider()
    assert provider.ping() == (True, "连通")

    broken, _, _ = make_provider(completions=FakeCompletions(error=RuntimeError("拒绝连接")))
    ok, detail = broken.ping()
    assert ok is False
    assert "拒绝连接" in detail

    no_key, _, _ = make_provider(requires_key=True, api_key="")
    assert no_key.ping()[0] is False


# ------------------------------------------------------------------ 降级链

def test_chain_falls_back_to_second_provider():
    primary, _, _ = make_provider(
        name="ollama", completions=FakeCompletions(error=RuntimeError("Ollama 未启动"))
    )
    secondary, secondary_completions, _ = make_provider(
        name="deepseek", completions=FakeCompletions(["来自备用通道"])
    )

    chain = ChatChain([primary, secondary])
    outcome = chain.chat([{"role": "user", "content": "hi"}])

    assert outcome.text == "来自备用通道"
    assert outcome.provider == "deepseek"
    assert outcome.fallback_used is True
    assert len(secondary_completions.calls) == 1


def test_chain_prefers_primary_when_healthy():
    primary, _, _ = make_provider(name="ollama", completions=FakeCompletions(["本地模型回答"]))
    secondary, secondary_completions, _ = make_provider(name="deepseek")

    outcome = ChatChain([primary, secondary]).chat([{"role": "user", "content": "hi"}])

    assert outcome.provider == "ollama"
    assert outcome.fallback_used is False
    assert secondary_completions.calls == []


def test_chain_raises_when_every_channel_unavailable():
    primary, _, _ = make_provider(name="ollama", completions=FakeCompletions(error=RuntimeError("挂了")))
    secondary, _, _ = make_provider(name="deepseek", requires_key=True, api_key="")

    with pytest.raises(ProviderError) as error:
        ChatChain([primary, secondary]).chat([{"role": "user", "content": "hi"}])

    message = str(error.value)
    assert "ollama" in message and "deepseek" in message


def test_chain_describe_reports_channels():
    primary, _, _ = make_provider(name="ollama")
    secondary, _, _ = make_provider(name="deepseek", requires_key=True, api_key="")

    described = ChatChain([primary, secondary]).describe()

    assert [item["name"] for item in described] == ["ollama", "deepseek"]
    assert described[0]["available"] is True
    assert described[1]["available"] is False


def test_chain_stream_falls_back_before_first_chunk():
    primary, _, _ = make_provider(
        name="ollama", completions=FakeCompletions(stream_error=RuntimeError("管道断开"))
    )
    secondary, _, _ = make_provider(
        name="deepseek", completions=FakeCompletions(stream_pieces=["备用", "流"])
    )

    provider_name, iterator = ChatChain([primary, secondary]).stream(
        [{"role": "user", "content": "hi"}]
    )

    assert provider_name == "deepseek"
    assert list(iterator) == ["备用", "流"]


# ------------------------------------------------------------------ 组装

def test_build_chat_chain_includes_fallback_when_configured():
    settings = Settings(
        _env_file=None,
        fallback_llm_provider="deepseek",
        deepseek_api_key="sk-abcd-1234",
        ollama_model="qwen3:4b",
    )

    chain = build_chat_chain(settings)

    assert [provider.name for provider in chain.providers] == ["ollama", "deepseek"]
    assert chain.primary.model == "qwen3:4b"


def test_build_chat_chain_without_fallback():
    settings = Settings(_env_file=None, fallback_llm_provider="none")

    chain = build_chat_chain(settings)

    assert [provider.name for provider in chain.providers] == ["ollama"]


def test_build_vision_provider_uses_settings():
    settings = Settings(
        _env_file=None,
        qwen_base_url="https://example.com/v1",
        qwen_api_key="sk-abcd-9999",
        qwen_vision_model="qwen3.5-omni-plus",
        qwen_vision_timeout=42.0,
    )

    provider = build_vision_provider(settings)

    assert provider.base_url == "https://example.com/v1"
    assert provider.model == "qwen3.5-omni-plus"
    assert provider.timeout == 42.0
    assert provider.available()[0] is True


# ------------------------------------------------------------------ 本地/云端


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:11434/v1", True),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", False),
        ("", False),
        (None, False),
    ],
)
def test_is_local_url(url, expected):
    assert is_local_url(url) is expected


def test_vision_uses_generic_settings_first():
    settings = Settings(
        _env_file=None,
        vision_base_url="http://127.0.0.1:11434/v1",
        vision_model="qwen2.5vl:3b",
        vision_api_key="",
        qwen_base_url="https://example.com/v1",
        qwen_api_key="sk-abcd-9999",
    )

    providers = build_vision_providers(settings)

    primary = providers[0]
    assert primary.base_url == "http://127.0.0.1:11434/v1"
    assert primary.model == "qwen2.5vl:3b"
    assert primary.name == "vision-local"
    assert primary.is_local is True
    # 本地通道不需要 Key
    assert primary.requires_key is False
    assert primary.available()[0] is True
    # 配置了云端 Key 时自动挂上回退通道
    assert [item.name for item in providers] == ["vision-local", "vision-cloud-fallback"]
    assert providers[1].model == "qwen3.5-omni-plus"


def test_vision_falls_back_to_legacy_qwen_settings():
    settings = Settings(
        _env_file=None,
        vision_base_url="",
        vision_model="",
        qwen_base_url="https://example.com/v1",
        qwen_vision_model="qwen3.5-omni-plus",
        qwen_api_key="sk-abcd-9999",
    )

    providers = build_vision_providers(settings)

    assert len(providers) == 1
    assert providers[0].name == "vision-cloud"
    assert providers[0].base_url == "https://example.com/v1"
    assert providers[0].requires_key is True


def test_vision_cloud_fallback_can_be_disabled():
    settings = Settings(
        _env_file=None,
        vision_base_url="http://127.0.0.1:11434/v1",
        vision_model="qwen2.5vl:3b",
        vision_auto_cloud_fallback=False,
        qwen_api_key="sk-abcd-9999",
    )

    providers = build_vision_providers(settings)

    assert [item.name for item in providers] == ["vision-local"]


def test_local_vision_without_cloud_key_has_single_channel():
    settings = Settings(
        _env_file=None,
        vision_base_url="http://127.0.0.1:11434/v1",
        vision_model="qwen2.5vl:3b",
        qwen_api_key="",
    )

    providers = build_vision_providers(settings)

    assert [item.name for item in providers] == ["vision-local"]


# ------------------------------------------------------------------ 思考模型


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3:4b", True),
        ("qwen3-vl:2b", True),
        ("deepseek-r1:8b", True),
        ("qwq:32b", True),
        ("qwen2.5:latest", False),
        ("qwen2.5vl:3b", False),
        ("", False),
    ],
)
def test_is_thinking_model(model, expected):
    assert is_thinking_model(model) is expected


def test_apply_no_think_appends_to_system_message():
    messages = [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": "报错E05怎么办"},
    ]

    result = apply_no_think(messages, "qwen3:4b")

    assert result[0]["content"].endswith("/no_think")
    assert result[1]["content"] == "报错E05怎么办"
    assert messages[0]["content"] == "你是客服助手。", "不应修改原消息"


def test_apply_no_think_without_system_uses_last_user_message():
    messages = [{"role": "user", "content": "你好"}]

    result = apply_no_think(messages, "deepseek-r1:8b")

    assert result[0]["content"].endswith("/no_think")


def test_apply_no_think_is_idempotent():
    messages = [
        {"role": "system", "content": "系统提示 /no_think"},
        {"role": "user", "content": "hi"},
    ]

    result = apply_no_think(messages, "qwen3:4b")

    assert result[0]["content"].count("/no_think") == 1


def test_apply_no_think_handles_multimodal_parts():
    messages = [
        {"role": "system", "content": "图片分析助手"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        },
    ]

    result = apply_no_think(messages, "qwen3-vl:2b")

    # 追加到 system（字符串），多模态分段不受影响
    assert result[0]["content"].endswith("/no_think")
    assert len(result[1]["content"]) == 2


def test_apply_no_think_multimodal_without_system():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]

    result = apply_no_think(messages, "qwen3-vl:2b")

    assert len(result[0]["content"]) == 3
    assert result[0]["content"][-1]["text"] == "/no_think"


def test_apply_no_think_skips_non_thinking_models_and_disabled_flag():
    messages = [{"role": "system", "content": "x"}]

    assert apply_no_think(messages, "qwen2.5:latest")[0]["content"] == "x"
    assert apply_no_think(messages, "qwen3:4b", enabled=False)[0]["content"] == "x"


def test_payload_disables_thinking_automatically():
    thinking, thinking_calls, _ = make_provider(
        model="qwen3:4b", disable_thinking=True
    )
    thinking.chat([{"role": "system", "content": "资料如下"}])
    assert thinking_calls.calls[0]["messages"][0]["content"].endswith("/no_think")

    normal, normal_calls, _ = make_provider(model="qwen2.5:latest")
    normal.chat([{"role": "system", "content": "资料如下"}])
    assert "/no_think" not in normal_calls.calls[0]["messages"][0]["content"]


def test_payload_can_opt_out_of_thinking_adaptation():
    provider, calls, _ = make_provider(model="qwen3:4b", disable_thinking=False)

    provider.chat([{"role": "system", "content": "资料如下"}])

    assert "/no_think" not in calls.calls[0]["messages"][0]["content"]
