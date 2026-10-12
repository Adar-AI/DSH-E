"""离线假模型：让后端逻辑不依赖任何真实模型也能完整跑通。

用途（对应"先把后端逻辑跑通，再验证模型层"的开发节奏）：
* 没装 Ollama、没有 API Key、断网时，依然能启动服务、走完整条链路并看到效果；
* 前端联调时响应稳定、毫秒级返回，不受模型波动影响；
* 单测与 CI 里不需要任何外部依赖。

行为刻意做成**确定性且带标记**：回答里会写明「离线模式·未调用大模型」，
避免把它误当成真实模型的输出；但引用标注、检索片段、工单字段这些都按真实规则生成，
所以链路行为与真实模型一致。启用方式：`LLM_MODE=mock`。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator, Sequence

from app.config import Settings, get_settings
from app.llm.providers import ChatOutcome

MOCK_BADGE = "【离线模式·未调用大模型】"
MOCK_TICKET_FIELDS = (
    "customer_request",
    "product_model",
    "fault_phenomenon",
    "error_info",
    "tried_actions",
    "urgency",
    "suggestion",
)

_DRAFT_LINE_RE = re.compile(r"^-\s*([a-z_]+)\s*:\s*(.*)$")


def _context_section(prompt: str) -> str:
    marker = "【资料片段】"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("请依据以上资料片段", 1)[0].strip()


def parse_contexts(prompt: str) -> list[tuple[str, str]]:
    """从回答提示词里解析出 (来源, 片段正文) 列表。"""
    contexts: list[tuple[str, str]] = []
    for block in _context_section(prompt).split("\n\n"):
        block = block.strip()
        if not block.startswith("["):
            continue
        first_line, _, rest = block.partition("\n")
        citation = first_line.split("来源：", 1)[-1].strip()
        contexts.append((citation, rest.strip()))
    return contexts


def build_mock_answer(prompt: str) -> str:
    """按检索片段生成确定性的"参考答案"。"""
    contexts = parse_contexts(prompt)
    if not contexts:
        return (
            f"{MOCK_BADGE}\n当前没有检索到可用的资料片段，无法给出结论。\n"
            "请补充产品型号与完整报错文字，或转人工客服。"
        )

    citation, text = contexts[0]
    snippet = " ".join(text.split())[:160]
    lines = [
        f"{MOCK_BADGE}",
        f"依据内部资料 [1]（{citation}）：{snippet}",
        "",
        "处理建议：",
        "1. 按上述资料里的步骤逐项排查；",
        "2. 若仍未解决，请补充产品型号与屏幕上的完整报错文字，或转人工客服跟进。",
        "",
        "[1]",
    ]
    return "\n".join(lines)


def build_mock_ticket_json(prompt: str) -> str:
    """把提示词里「已知的规则抽取结果」原样转成 JSON，保证离线也能定稿工单。"""
    fields: dict[str, str] = {}
    for raw_line in prompt.splitlines():
        match = _DRAFT_LINE_RE.match(raw_line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()

    payload = {key: fields.get(key, "") for key in MOCK_TICKET_FIELDS}
    if not payload["urgency"]:
        payload["urgency"] = "普通"
    return json.dumps(payload, ensure_ascii=False)


def _is_ticket_prompt(prompt: str) -> bool:
    return '"customer_request"' in prompt and "只输出" in prompt


def build_mock_reply(messages: Sequence[dict]) -> str:
    """根据提示词类型分流：工单抽取返回 JSON，回答返回带引用的文本。"""
    prompt = ""
    for item in reversed(list(messages)):
        if item.get("role") == "user":
            content = item.get("content")
            prompt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            break

    if _is_ticket_prompt(prompt):
        return build_mock_ticket_json(prompt)
    return build_mock_answer(prompt)


class MockProvider:
    """假通道：接口与 OpenAICompatProvider 一致。"""

    def __init__(self, *, latency: float = 0.0) -> None:
        self.name = "mock"
        self.model = "mock-deterministic"
        self.base_url = "mock://local"
        self.latency = latency
        self.calls: list[list[dict]] = []

    def available(self) -> tuple[bool, str]:
        return True, "离线模式（未调用任何真实模型）"

    def ping(self) -> tuple[bool, str]:
        return True, "离线模式就绪"

    def chat(self, messages: Sequence[dict], **overrides: Any) -> str:
        self.calls.append([dict(item) for item in messages])
        if self.latency:
            time.sleep(self.latency)
        return build_mock_reply(messages)

    def chat_stream(self, messages: Sequence[dict], **overrides: Any) -> Iterator[str]:
        text = self.chat(messages)
        for index in range(0, len(text), 12):
            yield text[index : index + 12]

    def list_models(self) -> list[str]:
        return [self.model]


class MockChatChain:
    """假对话链：接口与 ChatChain 一致（chat / stream / describe / providers）。"""

    def __init__(self, *, latency: float = 0.0) -> None:
        self.provider = MockProvider(latency=latency)

    @property
    def providers(self) -> list[MockProvider]:
        return [self.provider]

    @property
    def primary(self) -> MockProvider:
        return self.provider

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.provider.name,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "available": True,
                "reason": "离线模式（未调用任何真实模型）",
            }
        ]

    def chat(self, messages: Sequence[dict], **overrides: Any) -> ChatOutcome:
        started = time.perf_counter()
        text = self.provider.chat(messages, **overrides)
        return ChatOutcome(
            text=text,
            provider=self.provider.name,
            model=self.provider.model,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def stream(self, messages: Sequence[dict], **overrides: Any) -> tuple[str, Iterator[str]]:
        return self.provider.name, self.provider.chat_stream(messages, **overrides)


class MockVisionProvider:
    """假视觉通道：返回固定的结构化图片描述。"""

    def __init__(self, reply: str | None = None) -> None:
        self.name = "mock-vision"
        self.model = "mock-vision-deterministic"
        self.base_url = "mock://vision"
        self.reply = reply or json.dumps(
            {
                "kind": "错误截图",
                "product_model": "",
                "error_codes": [],
                "screen_location": "",
                "visible_damage": "",
                "other_text": "",
                "summary": "离线模式：未真正识别图片内容，请在 LLM_MODE=auto 下查看真实识别结果",
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )
        self.calls: list[list[dict]] = []

    def available(self) -> tuple[bool, str]:
        return True, "离线模式（未调用任何真实模型）"

    def ping(self) -> tuple[bool, str]:
        return True, "离线模式就绪"

    def chat(self, messages: Sequence[dict], **overrides: Any) -> str:
        self.calls.append([dict(item) for item in messages])
        return self.reply

    def list_models(self) -> list[str]:
        return [self.model]


def build_mock_chat_chain(settings: Settings | None = None) -> MockChatChain:
    return MockChatChain()


def build_mock_vision_provider(settings: Settings | None = None) -> MockVisionProvider:
    return MockVisionProvider()


def is_mock_mode(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return str(settings.llm_mode or "").strip().lower() == "mock"


__all__ = [
    "MOCK_BADGE",
    "MockChatChain",
    "MockProvider",
    "MockVisionProvider",
    "build_mock_answer",
    "build_mock_chat_chain",
    "build_mock_reply",
    "build_mock_ticket_json",
    "build_mock_vision_provider",
    "is_mock_mode",
    "parse_contexts",
]
