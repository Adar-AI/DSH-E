"""联网兜底检索：通过 MCP 调用百炼 EnhancedSearch（`search_pro`）。

定位（需求 F19，用户已确认）：

* **仅当内部资料证据不足时触发**（内部重排 top1 < `MCP_TRIGGER_TOP1`）；内部已命中就不联网；
* 联网结果必须**包含可判别实体（型号 / 品牌）**，否则整批丢弃；
* 结果单独标 `[网N]` 并注明"仅供参考"，**不入工单**。

为什么只认型号/品牌、不认错误码（实测依据）：用「L3250 E05」查询真实 MCP，
返回的 5 条**全部含 `E05`、没有一条含 `L3250` 或我们的品牌**，而且它们是 Canon 喷墨机的
"E05=墨盒未识别"，与本公司激光机手册的"E05=硒鼓未识别"**正好相反**。
按错误码过滤等于不过滤（100% 通过），放进去就会让客户去换墨盒。
因此：**只按型号/品牌校验；若手里没有型号，就宁可不要联网结果**。

工程要点：

* MCP 客户端是异步的（Streamable HTTP），而图节点是同步的 → 在工作线程里 `asyncio.run()`；
  若调用方本身已在事件循环里，则丢到独立线程执行，避免 "asyncio.run() cannot be called from
  a running event loop"；
* 每次调用自建一次会话：无状态、无共享事件循环问题（实测握手开销可忽略，整次约 0.8~1.1s）；
* 超时、连接失败、工具报错、返回体不是 JSON —— 全部转成 `WebSearchOutcome(ok=False, error=...)`，
  绝不让主链路崩（失败隔离）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 500  # MCP search_pro 的参数上限（1~500）


class WebSearchError(RuntimeError):
    """联网检索失败。"""


@dataclass(frozen=True)
class WebResult:
    """一条联网搜索结果。"""

    title: str = ""
    url: str = ""
    hostname: str = ""
    snippet: str = ""
    rank: int = 0

    @property
    def citation(self) -> str:
        parts = [part for part in (self.title, self.hostname) if part]
        return " · ".join(parts) or self.url

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "hostname": self.hostname,
            "snippet": self.snippet,
            "rank": self.rank,
        }

    def as_candidate(self, channel: str = "web"):
        """转成融合用的候选（延迟导入避免循环依赖）。"""
        from app.rag.fusion import Candidate

        return Candidate.from_web(self.as_dict(), channel=channel)


@dataclass
class WebSearchOutcome:
    """一次联网检索的结果（含失败原因与被过滤条数，便于 trace）。"""

    ok: bool
    query: str = ""
    tool: str = ""
    results: tuple[WebResult, ...] = ()
    error: str = ""
    elapsed_ms: int = 0
    dropped: int = 0

    @property
    def count(self) -> int:
        return len(self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "query": self.query,
            "tool": self.tool,
            "count": self.count,
            "dropped": self.dropped,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "results": [item.as_dict() for item in self.results],
        }


# ------------------------------------------------------------------ 客户端

@runtime_checkable
class WebSearchClient(Protocol):
    """联网检索客户端协议（可同步可异步，便于打桩）。"""

    def search(self, query: str) -> Any:
        ...


class DashScopeMcpSearchClient:
    """百炼 EnhancedSearch MCP 客户端（Streamable HTTP）。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 45.0,
        tool_name: str = "",
        max_results: int = 5,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.tool_name = tool_name
        self.max_results = max_results

    async def search(self, query: str) -> list[dict[str, Any]]:
        """返回规范化后的结果列表（title/url/hostname/snippet）。"""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover
            raise WebSearchError(f"缺少 mcp 依赖：{exc}") from exc

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with streamablehttp_client(
                self.base_url, headers=headers, timeout=self.timeout
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool = await self._pick_tool(session)
                    result = await session.call_tool(tool, {"query": query})
        except WebSearchError:
            raise
        except Exception as exc:  # noqa: BLE001 - 网络/协议异常统一转义
            raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc

        if getattr(result, "isError", False):
            raise WebSearchError("MCP 工具返回错误")

        pages = _parse_pages(getattr(result, "content", []) or [])
        return pages[: self.max_results]

    async def _pick_tool(self, session) -> str:
        if self.tool_name:
            return self.tool_name
        tools = await session.list_tools()
        names = [tool.name for tool in tools.tools]
        if not names:
            raise WebSearchError("MCP 服务没有提供任何工具")
        for name in names:
            if "search" in name.lower():
                return name
        return names[0]


def _parse_pages(content: Sequence[Any]) -> list[dict[str, Any]]:
    """从 MCP 返回的内容里解析出 pages 列表。"""
    for item in content:
        if getattr(item, "type", "") != "text":
            continue
        text = getattr(item, "text", "") or ""
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue

        raw_pages: Any = None
        if isinstance(payload, Mapping):
            for key in ("pages", "results", "items", "data"):
                if isinstance(payload.get(key), list):
                    raw_pages = payload[key]
                    break
        elif isinstance(payload, list):
            raw_pages = payload

        if not raw_pages:
            continue

        pages: list[dict[str, Any]] = []
        for entry in raw_pages:
            if not isinstance(entry, Mapping):
                continue
            pages.append(
                {
                    "title": str(entry.get("title") or ""),
                    "url": str(entry.get("url") or entry.get("link") or ""),
                    "hostname": str(entry.get("hostname") or entry.get("site") or ""),
                    "snippet": str(entry.get("snippet") or entry.get("content") or ""),
                }
            )
        return pages
    return []


def get_client(settings: Settings | None = None) -> DashScopeMcpSearchClient:
    settings = settings or get_settings()
    return DashScopeMcpSearchClient(
        base_url=settings.mcp_base_url,
        api_key=settings.mcp_api_key,
        timeout=settings.mcp_timeout,
        tool_name=settings.mcp_tool_name,
        max_results=settings.mcp_max_results,
    )


# ------------------------------------------------------------------ 触发与查询

def is_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.mcp_enabled and settings.mcp_base_url and settings.mcp_api_key)


def should_search(top1: float, *, settings: Settings | None = None) -> bool:
    """内部资料证据是否不足到需要联网。"""
    settings = settings or get_settings()
    if not is_enabled(settings):
        return False
    return float(top1) < float(settings.mcp_trigger_top1)


def build_query(
    question: str,
    entities: Sequence[str] = (),
    *,
    max_chars: int = MAX_QUERY_CHARS,
) -> str:
    """构造联网查询：问题 + 实体（型号/错误码），保证实体在场。"""
    question = (question or "").strip()
    parts: list[str] = []
    for entity in entities:
        name = str(entity or "").strip()
        if name and name.upper() not in question.upper():
            parts.append(name)
    query = " ".join([*parts, question]).strip() or question
    return query[:max_chars]


def filter_by_entity(
    results: Sequence[WebResult],
    entities: Sequence[str],
    *,
    require: bool = True,
) -> tuple[list[WebResult], int]:
    """按**可判别实体（型号/品牌）**过滤联网结果。

    注意：**不要传错误码**——同名错误码在各品牌间含义不同（实测 E05 在 Canon 是墨盒、
    在本地手册是硒鼓），按错误码过滤等于不过滤。没有实体时不启用过滤（由 `search()` 决定
    是否直接跳过联网）。
    """
    names = [str(item).strip().upper() for item in entities if str(item or "").strip()]
    if not names or not require:
        return list(results), 0

    kept: list[WebResult] = []
    dropped = 0
    for index, item in enumerate(results, start=1):
        haystack = f"{item.title} {item.snippet} {item.url}".upper()
        if any(name in haystack for name in names):
            kept.append(
                WebResult(
                    title=item.title,
                    url=item.url,
                    hostname=item.hostname,
                    snippet=item.snippet,
                    rank=len(kept) + 1,
                )
            )
        else:
            dropped += 1
    return kept, dropped


# ------------------------------------------------------------------ 主入口

def _run(maybe_awaitable: Any) -> Any:
    """兼容同步/异步客户端：必要时在独立线程里跑事件循环。"""
    if not inspect.isawaitable(maybe_awaitable):
        return maybe_awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(maybe_awaitable)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, maybe_awaitable).result()


def is_discriminating_entity(name: str) -> bool:
    """是否是可用于校验的实体（型号 / 品牌），而不是错误码。

    错误码形如 `E05`/`ER12`（1~3 个字母 + 1~3 位数字），各品牌含义不同，不能当校验依据。
    """
    from app.tickets.draft import ERROR_CODE_RE

    text = str(name or "").strip().upper()
    if not text:
        return False
    return ERROR_CODE_RE.fullmatch(text) is None


def search(
    query: str,
    *,
    entities: Sequence[str] = (),
    settings: Settings | None = None,
    client: WebSearchClient | None = None,
) -> WebSearchOutcome:
    """执行一次联网检索。任何失败都返回 `ok=False`，不抛异常。

    `entities` 传的是**可判别实体（型号 / 品牌）**；错误码会被自动剔除。
    当 `MCP_REQUIRE_ENTITY_MATCH=true` 且剔除后没有可判别实体时，直接跳过（连网络都不发）
    —— 因为无法校验结果是否在讲同一款产品。
    """
    settings = settings or get_settings()

    if not settings.mcp_enabled:
        return WebSearchOutcome(ok=False, query=query, error="MCP 联网检索已关闭（MCP_ENABLED=false）")
    if not settings.mcp_base_url or not settings.mcp_api_key:
        return WebSearchOutcome(ok=False, query=query, error="未配置 MCP 地址或密钥")
    if not (query or "").strip():
        return WebSearchOutcome(ok=False, query=query, error="查询为空")

    raw_entities = [str(item).strip() for item in entities if str(item or "").strip()]
    discriminators = [item for item in raw_entities if is_discriminating_entity(item)]
    ignored = [item for item in raw_entities if item not in discriminators]

    if settings.mcp_require_entity_match and not discriminators:
        reason = "没有可用于校验的型号/品牌实体，联网结果无法验证，已跳过"
        if ignored:
            reason += f"（已忽略错误码：{'、'.join(ignored)}）"
        return WebSearchOutcome(ok=False, query=query, error=reason)

    active = client or get_client(settings)
    tool_name = getattr(active, "tool_name", "") or "search"

    started = time.perf_counter()
    try:
        pages = _run(active.search(query[:MAX_QUERY_CHARS]))
    except WebSearchError as exc:
        logger.warning("联网检索失败：%s", exc)
        return WebSearchOutcome(
            ok=False, query=query, tool=tool_name, error=str(exc),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - 打桩/网络/协议异常统一兜住
        logger.warning("联网检索异常：%s", exc)
        return WebSearchOutcome(
            ok=False, query=query, tool=tool_name, error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        return WebSearchOutcome(ok=False, query=query, tool=tool_name, error="MCP 返回格式异常")

    results = [
        WebResult(
            title=str(page.get("title") or ""),
            url=str(page.get("url") or ""),
            hostname=str(page.get("hostname") or ""),
            snippet=str(page.get("snippet") or ""),
            rank=index,
        )
        for index, page in enumerate(pages, start=1)
        if isinstance(page, Mapping)
    ]

    kept, dropped = filter_by_entity(
        results, discriminators, require=bool(settings.mcp_require_entity_match)
    )
    # 条数上限由本层兜住（不依赖具体客户端实现）；先按实体过滤再截断，避免把命中的结果截掉
    limit = int(settings.mcp_max_results or 0)
    if limit > 0:
        kept = kept[:limit]

    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        "联网检索完成：%d 条 → 保留 %d 条（实体过滤丢弃 %d，%dms）",
        len(results), len(kept), dropped, elapsed,
    )
    return WebSearchOutcome(
        ok=True,
        query=query,
        tool=tool_name,
        results=tuple(kept),
        dropped=dropped,
        elapsed_ms=elapsed,
    )


__all__ = [
    "DashScopeMcpSearchClient",
    "MAX_QUERY_CHARS",
    "WebResult",
    "WebSearchClient",
    "WebSearchError",
    "WebSearchOutcome",
    "build_query",
    "filter_by_entity",
    "get_client",
    "is_discriminating_entity",
    "is_enabled",
    "search",
    "should_search",
]
