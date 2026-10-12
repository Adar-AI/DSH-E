"""MCP 联网检索通道自检：连接百炼 EnhancedSearch MCP，列出工具并试调一次搜索。

用法：
    & 'C:\\Users\\19181\\PyCharmMiscProject\\.venv\\Scripts\\python.exe' scripts\\probe_mcp.py
    & '...python.exe' scripts\\probe_mcp.py --query "云打 L3250 打印机 E05 报错"
    & '...python.exe' scripts\\probe_mcp.py --list-only

密钥从 .env 的 MCP_API_KEY 读取（已被 .gitignore 忽略），输出一律脱敏。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings, mask_secret  # noqa: E402


def describe_tool(tool) -> None:
    print(f"  · 工具名：{tool.name}")
    description = (tool.description or "").strip().replace("\n", " ")
    print(f"    说明：{description[:160]}")
    schema = tool.inputSchema or {}
    properties = schema.get("properties") or {}
    if properties:
        print("    参数：")
        for name, spec in properties.items():
            kind = spec.get("type", "?")
            required = "必填" if name in (schema.get("required") or []) else "可选"
            desc = str(spec.get("description", ""))[:70].replace("\n", " ")
            print(f"      - {name} ({kind}, {required}) {desc}")


def summarize_result(result) -> None:
    """打印 call_tool 返回，尽量提取出可读内容。"""
    print(f"    返回：isError={getattr(result, 'isError', None)}")
    for index, item in enumerate(getattr(result, "content", []) or []):
        kind = getattr(item, "type", "?")
        if kind == "text":
            text = getattr(item, "text", "") or ""
            print(f"    [{index}] text（{len(text)} 字）：")
            print("        " + text[:1500].replace("\n", "\n        "))
        else:
            print(f"    [{index}] {kind}：{str(item)[:200]}")


async def probe(query: str, list_only: bool) -> int:
    settings = get_settings()
    base_url = settings.mcp_base_url
    api_key = settings.mcp_api_key

    print("=== MCP 配置 ===")
    print(f"base_url = {base_url}")
    print(f"api_key  = {mask_secret(api_key)}")
    print(f"enabled  = {settings.mcp_enabled}｜timeout = {settings.mcp_timeout}s")

    if not base_url:
        print("❌ 未配置 MCP_BASE_URL")
        return 1
    if not api_key:
        print("❌ 未配置 MCP_API_KEY")
        return 1

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:  # pragma: no cover
        print(f"❌ 缺少 mcp 依赖：{exc}")
        return 1

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with streamablehttp_client(
            base_url, headers=headers, timeout=settings.mcp_timeout
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("\n✅ 连接成功（MCP 会话已初始化）")

                tools = await session.list_tools()
                names = [tool.name for tool in tools.tools]
                print(f"\n=== 可用工具（{len(names)} 个）===")
                for tool in tools.tools:
                    describe_tool(tool)

                if list_only or not names:
                    return 0

                # 优先挑搜索类工具
                target = next(
                    (name for name in names if "search" in name.lower()),
                    names[0],
                )
                schema = next(
                    tool.inputSchema for tool in tools.tools if tool.name == target
                ) or {}
                properties = schema.get("properties") or {}
                required = schema.get("required") or []

                arguments: dict[str, object] = {}
                for name in required:
                    if name in {"query", "q", "keyword", "keywords", "search_query", "text"}:
                        arguments[name] = query
                if not arguments:
                    for name in ("query", "q", "keyword", "keywords", "search_query", "text"):
                        if name in properties:
                            arguments[name] = query
                            break
                if not arguments:
                    print(f"\n⚠️ 无法自动构造参数（入参：{list(properties)}），请手动指定")
                    return 1

                print(f"\n=== 试调 {target}（参数 {list(arguments)}）===")
                print(f"查询：{query}")
                result = await session.call_tool(target, arguments)
                summarize_result(result)
                return 0 if not getattr(result, "isError", False) else 1
    except Exception as exc:  # noqa: BLE001 - 探测脚本要给出原因而不是栈
        print(f"\n❌ 连接/调用失败：{type(exc).__name__}: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测 MCP 联网检索通道")
    parser.add_argument("--query", default="云打 L3250 打印机 E05 报错怎么处理")
    parser.add_argument("--list-only", action="store_true", help="只列工具，不试调")
    args = parser.parse_args(argv)
    return asyncio.run(probe(args.query, args.list_only))


if __name__ == "__main__":
    raise SystemExit(main())
