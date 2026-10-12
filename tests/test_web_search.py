"""联网兜底检索测试：触发条件、查询构造、实体过滤、失败隔离（全部离线打桩）。"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.rag.fusion import SOURCE_WEB
from app.rag.web_search import (
    MAX_QUERY_CHARS,
    DashScopeMcpSearchClient,
    WebResult,
    WebSearchError,
    build_query,
    filter_by_entity,
    is_enabled,
    search,
    should_search,
)

PAGES = [
    {
        "title": "云打 L3250 报错 E05 处理方法",
        "url": "https://example.com/a",
        "hostname": "example.com",
        "snippet": "L3250 出现 E05 时，请检查硒鼓芯片触点并复位计数器。",
    },
    {
        "title": "某品牌喷墨打印机 E05 = 墨盒未识别",
        "url": "https://other.com/b",
        "hostname": "other.com",
        "snippet": "E05 表示墨盒无法识别，请更换墨盒。",
    },
    {
        "title": "打印机日常保养",
        "url": "https://third.com/c",
        "hostname": "third.com",
        "snippet": "每月清洁进纸轮可以延长寿命。",
    },
]


class FakeClient:
    """同步假客户端。"""

    tool_name = "search_pro"

    def __init__(self, pages=None, error: str | None = None) -> None:
        self.pages = pages if pages is not None else PAGES
        self.error = error
        self.calls: list[str] = []

    def search(self, query: str):
        self.calls.append(query)
        if self.error:
            raise WebSearchError(self.error)
        return self.pages


class FakeAsyncClient(FakeClient):
    """异步假客户端（与真实 MCP 客户端形态一致）。"""

    async def search(self, query: str):  # type: ignore[override]
        await asyncio.sleep(0)
        return super().search(query)


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        mcp_enabled=True,
        mcp_base_url="https://example.com/mcp",
        mcp_api_key="sk-abcd-1234",
        mcp_require_entity_match=True,
        mcp_max_results=5,
        mcp_trigger_top1=0.35,
    )


# ------------------------------------------------------------------ 启用与触发

def test_is_enabled_requires_url_and_key(settings):
    assert is_enabled(settings) is True
    assert is_enabled(settings.model_copy(update={"mcp_enabled": False})) is False
    assert is_enabled(settings.model_copy(update={"mcp_base_url": ""})) is False
    assert is_enabled(settings.model_copy(update={"mcp_api_key": ""})) is False


def test_should_search_only_when_internal_evidence_is_weak(settings):
    assert should_search(0.10, settings=settings) is True
    assert should_search(0.34, settings=settings) is True
    assert should_search(0.35, settings=settings) is False
    assert should_search(0.90, settings=settings) is False, "内部已命中就不该联网"


def test_should_search_false_when_disabled(settings):
    disabled = settings.model_copy(update={"mcp_enabled": False})
    assert should_search(0.0, settings=disabled) is False


# ------------------------------------------------------------------ 查询构造

def test_build_query_puts_entities_first_and_dedupes():
    query = build_query("保修期是多少？", ["L3250", "E05", "l3250"])

    assert query.startswith("L3250")
    assert "E05" in query
    assert query.count("L3250") == 1, "问题里没提到时才追加实体"
    assert query.endswith("保修期是多少？")


def test_build_query_skips_entities_already_in_question():
    query = build_query("L3250 保修期", ["L3250"])
    assert query == "L3250 保修期"


def test_build_query_truncates_to_mcp_limit():
    query = build_query("问" * 900, ["L3250"])
    assert len(query) <= MAX_QUERY_CHARS


# ------------------------------------------------------------------ 实体过滤

def test_filter_by_entity_drops_results_without_entity():
    results = [WebResult(**{k: page[k] for k in ("title", "url", "hostname", "snippet")}) for page in PAGES]

    kept, dropped = filter_by_entity(results, ["L3250"], require=True)

    assert [item.url for item in kept] == ["https://example.com/a"]
    assert dropped == 2
    assert kept[0].rank == 1


def test_filter_by_entity_keeps_all_when_entity_missing_from_question():
    results = [WebResult(title=page["title"], url=page["url"]) for page in PAGES]

    kept, dropped = filter_by_entity(results, [], require=True)

    assert len(kept) == 3
    assert dropped == 0


def test_filter_by_entity_can_be_disabled():
    results = [WebResult(title=page["title"], url=page["url"]) for page in PAGES]

    kept, dropped = filter_by_entity(results, ["L3250"], require=False)

    assert len(kept) == 3
    assert dropped == 0


# ------------------------------------------------------------------ 主入口

def test_search_maps_results_and_filters(settings):
    client = FakeClient()

    outcome = search("L3250 报错 E05", entities=["L3250"], settings=settings, client=client)

    assert outcome.ok is True
    assert outcome.tool == "search_pro"
    assert outcome.count == 1
    assert outcome.dropped == 2
    assert outcome.results[0].hostname == "example.com"
    assert outcome.elapsed_ms >= 0
    assert client.calls == ["L3250 报错 E05"]


def test_search_skips_network_without_discriminating_entity(settings):
    """没有型号/品牌可校验时连网络都不发（防同名错误码污染，见模块 docstring 实测）。"""
    client = FakeClient()

    outcome = search("打印报错 E05 怎么处理", entities=["E05"], settings=settings, client=client)

    assert outcome.ok is False
    assert "无法验证" in outcome.error
    assert client.calls == [], "不应发起网络请求"


def test_error_code_alone_is_not_a_valid_discriminator(settings):
    """错误码在各品牌含义不同，不能当作校验依据。"""
    client = FakeClient()

    outcome = search(
        "E05",
        entities=["E05", "ER12"],
        settings=settings,
        client=client,
    )

    assert outcome.ok is False
    assert "已忽略错误码" in outcome.error
    assert client.calls == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [("L3250", True), ("M1136", True), ("云打", True), ("E05", False), ("ER12", False), ("", False)],
)
def test_is_discriminating_entity(name, expected):
    from app.rag.web_search import is_discriminating_entity

    assert is_discriminating_entity(name) is expected


def test_search_allows_any_entity_when_check_disabled(settings):
    """关闭实体校验时允许不传实体（联网结果仅作展示，不作答案依据）。"""
    relaxed = settings.model_copy(update={"mcp_require_entity_match": False})
    client = FakeClient()

    outcome = search("打印机 E05", settings=relaxed, client=client)

    assert outcome.ok is True
    assert outcome.count == 3
    assert client.calls == ["打印机 E05"]


def test_search_supports_async_client(settings):
    client = FakeAsyncClient()

    outcome = search("L3250 报错 E05", entities=["L3250"], settings=settings, client=client)

    assert outcome.ok is True
    assert outcome.count == 1
    assert client.calls == ["L3250 报错 E05"]


def test_search_isolates_client_errors(settings):
    client = FakeClient(error="连接超时")

    outcome = search("L3250 报错", entities=["L3250"], settings=settings, client=client)

    assert outcome.ok is False
    assert "连接超时" in outcome.error
    assert outcome.count == 0


def test_search_isolates_unexpected_exceptions(settings):
    class Boom:
        tool_name = "search_pro"

        def search(self, query):
            raise ValueError("返回体不是 JSON")

    outcome = search("L3250", entities=["L3250"], settings=settings, client=Boom())

    assert outcome.ok is False
    assert "ValueError" in outcome.error


def test_search_reports_disabled_or_unconfigured(settings):
    disabled = settings.model_copy(update={"mcp_enabled": False})
    outcome = search("x", entities=["L3250"], settings=disabled, client=FakeClient())
    assert outcome.ok is False and "已关闭" in outcome.error

    empty = settings.model_copy(update={"mcp_api_key": ""})
    outcome = search("x", entities=["L3250"], settings=empty, client=FakeClient())
    assert outcome.ok is False and "未配置" in outcome.error


def test_search_rejects_empty_query(settings):
    outcome = search("   ", settings=settings, client=FakeClient())
    assert outcome.ok is False
    assert "为空" in outcome.error


def test_search_handles_bad_payload_shape(settings):
    outcome = search("x", entities=["L3250"], settings=settings, client=FakeClient(pages="不是列表"))

    assert outcome.ok is False
    assert "格式异常" in outcome.error


def test_search_limits_result_count(settings):
    many = [
        {"title": f"L3250 结果{i}", "url": f"https://x/{i}", "hostname": "x", "snippet": "L3250"}
        for i in range(20)
    ]
    client = FakeClient(pages=many)
    limited = settings.model_copy(update={"mcp_max_results": 3})

    outcome = search("L3250", entities=["L3250"], settings=limited, client=client)

    assert outcome.ok is True
    assert outcome.count == 3


def test_outcome_as_dict_is_trace_friendly(settings):
    outcome = search("L3250 报错", entities=["L3250"], settings=settings, client=FakeClient())

    payload = outcome.as_dict()

    assert set(payload) == {
        "ok", "query", "tool", "count", "dropped", "elapsed_ms", "error", "results",
    }
    assert payload["count"] == 1
    assert payload["results"][0]["url"] == "https://example.com/a"


# ------------------------------------------------------------------ 候选转换

def test_web_result_converts_to_candidate():
    result = WebResult(title="标题", url="https://x/1", hostname="x.com", snippet="摘要")

    candidate = result.as_candidate()

    assert candidate.source == SOURCE_WEB
    assert candidate.key == "web:https://x/1"
    assert candidate.url == "https://x/1"
    assert "标题" in candidate.citation
    assert candidate.is_web is True


# ------------------------------------------------------------------ 客户端解析

def test_parse_pages_tolerates_alternative_keys():
    from app.rag.web_search import _parse_pages
    from types import SimpleNamespace

    content = [
        SimpleNamespace(type="text", text='{"results": [{"title": "T", "link": "https://a", "site": "a.com", "content": "摘要"}]}')
    ]

    pages = _parse_pages(content)

    assert pages == [{"title": "T", "url": "https://a", "hostname": "a.com", "snippet": "摘要"}]


def test_parse_pages_returns_empty_for_non_json():
    from app.rag.web_search import _parse_pages
    from types import SimpleNamespace

    assert _parse_pages([SimpleNamespace(type="text", text="not json")]) == []
    assert _parse_pages([]) == []


def test_client_requires_configuration():
    client = DashScopeMcpSearchClient(base_url="", api_key="", timeout=5, max_results=2)
    assert client.max_results == 2
    assert client.tool_name == ""
