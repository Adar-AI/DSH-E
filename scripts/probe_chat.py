"""对话通道自检脚本：确认本地/备用通道能用，并测量真实延迟。

用法：
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_chat.py
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_chat.py --question "保修期多久" --max-tokens 120
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_chat.py --provider deepseek   # 只测备用通道
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.prompts import build_answer_messages  # noqa: E402
from app.llm.providers import ChatChain, build_chat_chain  # noqa: E402


class DemoChunk:
    """模拟一条检索片段，用于让回答走完整的「带引用」路径。"""

    def __init__(self, citation: str, text: str) -> None:
        self.citation = citation
        self.text = text


DEMO_CONTEXTS = [
    DemoChunk(
        "打印机使用手册.md · 第 12 页 · 耗材更换",
        "更换硒鼓的步骤：先关闭打印机电源，打开前盖，取出旧硒鼓，装入新硒鼓后合上前盖，"
        "最后开机并复位硒鼓计数器。若更换后仍报 E05，请检查硒鼓芯片触点是否清洁。",
    ),
    DemoChunk(
        "售后服务规则.md · 保修政策",
        "硒鼓、墨盒等耗材属于易损件，保修期为三个月，非人为损坏可免费更换。",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对话通道自检")
    parser.add_argument(
        "--question", default="打印机显示 E05，硒鼓刚换过，怎么处理？", help="测试问题"
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--provider", default=None, help="只测指定通道（ollama / deepseek）")
    parser.add_argument("--no-context", action="store_true", help="不带检索片段（测纯对话速度）")
    parser.add_argument("--stream", action="store_true", help="用流式调用并统计首包延迟")
    args = parser.parse_args(argv)

    settings = get_settings()
    chain = build_chat_chain(settings)
    if args.provider:
        chain = ChatChain([p for p in chain.providers if p.name == args.provider])
        if not chain.providers:
            print(f"没有名为 {args.provider} 的通道")
            return 2

    print("通道状态：")
    for provider in chain.providers:
        ok, detail = provider.available()
        print(f"  {'✅' if ok else '❌'} {provider.name:9} 模型={provider.model:18} {detail}")

    contexts = [] if args.no_context else DEMO_CONTEXTS
    messages = build_answer_messages(args.question, contexts)

    if args.stream:
        started = time.perf_counter()
        provider_name, iterator = chain.stream(messages, max_tokens=args.max_tokens)
        pieces: list[str] = []
        first_at: float | None = None
        for piece in iterator:
            if first_at is None:
                first_at = time.perf_counter() - started
            pieces.append(piece)
        total = time.perf_counter() - started
        print(f"\n通道={provider_name} 首包={first_at:.1f}s 总计={total:.1f}s 增量={len(pieces)}")
        print("".join(pieces))
        return 0

    started = time.perf_counter()
    outcome = chain.chat(messages, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - started

    print(
        f"\n通道={outcome.provider} 模型={outcome.model} 用时={elapsed:.1f}s "
        f"降级={outcome.fallback_used} 片段数={len(contexts)}"
    )
    print("-" * 60)
    print(outcome.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
