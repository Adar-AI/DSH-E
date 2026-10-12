"""对话模型统一封装：全部走 OpenAI 兼容协议。

三个通道只差 base_url / api_key / model，因此共用同一份实现：

| 通道 | base_url | 说明 |
|---|---|---|
| 本地 Ollama | `http://127.0.0.1:11434/v1` | 默认通道，离线可用；api_key 填占位值 |
| DeepSeek | `https://api.deepseek.com/v1` | 备用通道，主通道不可用时自动切换 |
| 多模态网关 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 图片理解（见 vision.py） |

统一成 OpenAI 协议的好处：一套重试/降级/流式逻辑复用到所有通道，
测试里注入假客户端即可完全离线验证。
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

from app.config import Settings, get_settings, is_local_url

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024

# 思考型模型的特征（会先吐一大段推理，正文可能为空）
THINKING_MODEL_HINTS = (
    "qwen3",
    "deepseek-r1",
    "deepseek-reasoner",
    "qwq",
    "magistral",
    "-think",
    "reasoning",
)

ClientFactory = Callable[[str, str, float], Any]


def is_thinking_model(model: str | None) -> bool:
    """判断是否是思考型模型。"""
    name = (model or "").lower()
    return any(hint in name for hint in THINKING_MODEL_HINTS)


def apply_no_think(
    messages: Sequence[dict], model: str, *, enabled: bool = True
) -> list[dict]:
    """给思考型模型追加 `/no_think` 软开关（qwen3 系支持）。

    实测：本地 `qwen3:4b` 在"依据资料作答"的任务上会把 token 预算全花在推理上，
    正文直接为空；加上 `/no_think` 才会正常作答。非思考模型原样返回，
    已经带过标记的不重复追加（多模态消息会把标记挂到文本分段上）。
    """
    items = [dict(message) for message in messages]
    if not enabled or not is_thinking_model(model):
        return items

    marker = "/no_think"

    def has_marker(value: Any) -> bool:
        if isinstance(value, str):
            return marker in value
        if isinstance(value, list):
            return any(
                isinstance(part, dict) and marker in str(part.get("text", ""))
                for part in value
            )
        return False

    if any(has_marker(item.get("content")) for item in items):
        return items

    def attach(item: dict) -> None:
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = f"{content}\n{marker}"
        elif isinstance(content, list):
            item["content"] = [*content, {"type": "text", "text": marker}]
        else:
            item["content"] = marker

    for item in items:
        if item.get("role") == "system":
            attach(item)
            return items

    for item in reversed(items):
        if item.get("role") == "user":
            attach(item)
            return items
    return items


class ProviderError(RuntimeError):
    """模型调用失败：网络异常、鉴权失败、超时、返回格式异常等。"""


@dataclass(frozen=True)
class ChatOutcome:
    """一次对话调用的结果（含使用的通道，便于页面提示当前模式）。"""

    text: str
    provider: str
    model: str
    elapsed_ms: int = 0
    fallback_used: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
            "fallback_used": self.fallback_used,
        }


def _default_client_factory(base_url: str, api_key: str, timeout: float) -> Any:
    from openai import OpenAI  # 延迟导入，测试无需安装也可跑（本机已装）

    return OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)


def _extract_text(response: Any) -> str:
    """从 OpenAI 兼容响应里取出文本（兼容 content 为分段数组的网关）。"""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ProviderError("模型返回中没有 choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        content = "".join(parts)

    if content is None:
        raise ProviderError("模型返回内容为空")
    return str(content)


class OpenAICompatProvider:
    """单个 OpenAI 兼容通道。"""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        requires_key: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 1,
        disable_thinking: bool = True,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.requires_key = requires_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None

    # ------------------------------------------------------------------ 基础

    @property
    def is_local(self) -> bool:
        """是否指向本机（本地 Ollama）。"""
        return is_local_url(self.base_url)

    def available(self) -> tuple[bool, str]:
        """配置层面是否可用（不发网络请求）。"""
        if self.requires_key and not self.api_key:
            return False, f"{self.name} 未配置 API Key"
        if not self.base_url:
            return False, f"{self.name} 未配置 base_url"
        return True, "已配置"

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.base_url, self.api_key, self.timeout)
        return self._client

    def _payload(self, messages: Sequence[dict], **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": apply_no_think(
                messages, self.model, enabled=self.disable_thinking
            ),
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        if overrides.get("timeout") is not None:
            payload["timeout"] = overrides["timeout"]
        return payload

    # ------------------------------------------------------------------ 调用

    def chat(self, messages: Sequence[dict], **overrides: Any) -> str:
        """单轮补全，返回文本。失败重试 `max_retries` 次后抛 ProviderError。"""
        payload = self._payload(messages, **overrides)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().chat.completions.create(**payload)
                return _extract_text(response)
            except Exception as exc:  # noqa: BLE001 - 各类 SDK 异常统一降级
                last_error = exc
                if attempt < self.max_retries:
                    logger.warning("[%s] 调用失败，准备重试：%s", self.name, exc)

        raise ProviderError(f"{self.name} 调用失败：{last_error}") from last_error

    def chat_stream(self, messages: Sequence[dict], **overrides: Any) -> Iterator[str]:
        """流式补全，逐段产出文本增量。"""
        payload = self._payload(messages, **overrides)
        payload["stream"] = True

        try:
            stream = self._get_client().chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.name} 流式调用失败：{exc}") from exc

        try:
            for chunk in stream:
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield str(piece)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.name} 流式中断：{exc}") from exc

    # ------------------------------------------------------------------ 诊断

    def list_models(self) -> list[str]:
        """列出网关可用模型（部分网关不支持，失败时抛 ProviderError）。"""
        try:
            response = self._get_client().models.list()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.name} 无法列出模型：{exc}") from exc

        data = getattr(response, "data", None) or []
        names: list[str] = []
        for item in data:
            identifier = getattr(item, "id", None) or getattr(item, "name", None)
            if identifier:
                names.append(str(identifier))
        return names

    def ping(self, **overrides: Any) -> tuple[bool, str]:
        """真实连通性探测：发一个极小的请求。"""
        ok, reason = self.available()
        if not ok:
            return False, reason
        try:
            self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
                **overrides,
            )
        except ProviderError as exc:
            return False, str(exc)
        return True, "连通"


class ChatChain:
    """按顺序尝试多个通道，第一个成功者胜出（自动降级）。"""

    def __init__(self, providers: Sequence[OpenAICompatProvider]) -> None:
        self.providers = [provider for provider in providers if provider is not None]

    @property
    def primary(self) -> OpenAICompatProvider | None:
        return self.providers[0] if self.providers else None

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": provider.name,
                "model": provider.model,
                "base_url": provider.base_url,
                "available": provider.available()[0],
                "reason": provider.available()[1],
            }
            for provider in self.providers
        ]

    def chat(self, messages: Sequence[dict], **overrides: Any) -> ChatOutcome:
        errors: list[str] = []
        for index, provider in enumerate(self.providers):
            ok, reason = provider.available()
            if not ok:
                errors.append(f"{provider.name}：{reason}")
                continue

            started = time.perf_counter()
            try:
                text = provider.chat(messages, **overrides)
            except ProviderError as exc:
                errors.append(str(exc))
                logger.warning("对话通道 %s 失败，尝试下一个：%s", provider.name, exc)
                continue

            elapsed = int((time.perf_counter() - started) * 1000)
            if index > 0:
                logger.info("已降级到通道 %s", provider.name)
            return ChatOutcome(
                text=text,
                provider=provider.name,
                model=provider.model,
                elapsed_ms=elapsed,
                fallback_used=index > 0,
            )

        raise ProviderError("所有对话通道都不可用：" + "；".join(errors))

    def stream(self, messages: Sequence[dict], **overrides: Any) -> tuple[str, Iterator[str]]:
        """流式对话。降级只发生在「首个 chunk 之前」失败的情况。"""
        errors: list[str] = []
        for _, provider in enumerate(self.providers):
            ok, reason = provider.available()
            if not ok:
                errors.append(f"{provider.name}：{reason}")
                continue

            iterator = provider.chat_stream(messages, **overrides)
            try:
                first = next(iterator)
            except StopIteration:
                return provider.name, iter(())
            except ProviderError as exc:
                errors.append(str(exc))
                logger.warning("流式通道 %s 失败，尝试下一个：%s", provider.name, exc)
                continue
            return provider.name, itertools.chain([first], iterator)

        raise ProviderError("所有对话通道都不可用：" + "；".join(errors))


def build_chat_chain(settings: Settings | None = None) -> ChatChain:
    """按配置组装通道链：Ollama（默认）→ DeepSeek（备用）。"""
    settings = settings or get_settings()

    providers = [
        OpenAICompatProvider(
            name="ollama",
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            api_key="ollama",
            timeout=settings.ollama_timeout,
            requires_key=False,
            disable_thinking=settings.disable_thinking,
        )
    ]

    if (settings.fallback_llm_provider or "").strip().lower() == "deepseek":
        providers.append(
            OpenAICompatProvider(
                name="deepseek",
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                timeout=60.0,
                requires_key=True,
                disable_thinking=settings.disable_thinking,
            )
        )

    return ChatChain(providers)


def build_vision_providers(settings: Settings | None = None) -> list[OpenAICompatProvider]:
    """组装视觉通道链：本地 Ollama 视觉模型（可选）→ 云端网关（可选回退）。

    规则：
    1. `VISION_BASE_URL / VISION_MODEL / VISION_API_KEY` 优先，留空回落到旧名 `QWEN_*`；
    2. 主通道指向本机时不要求 API Key（Ollama 不需要）；
    3. 主通道是本地、且配置了云端 Key 时，按 `VISION_AUTO_CLOUD_FALLBACK` 自动把云端
       作为第二通道 —— 本地模型没拉下来或加载失败时，图片理解仍可用（并在结果里说明来源）。
    """
    settings = settings or get_settings()

    base_url = settings.effective_vision_base_url
    model = settings.effective_vision_model
    local = is_local_url(base_url)

    providers = [
        OpenAICompatProvider(
            name="vision-local" if local else "vision-cloud",
            base_url=base_url,
            model=model,
            api_key=settings.effective_vision_api_key or ("ollama" if local else ""),
            timeout=settings.effective_vision_timeout,
            requires_key=not local,
            disable_thinking=settings.disable_thinking,
        )
    ]

    if local and settings.vision_auto_cloud_fallback and settings.qwen_api_key.strip():
        providers.append(
            OpenAICompatProvider(
                name="vision-cloud-fallback",
                base_url=settings.qwen_base_url,
                model=settings.qwen_vision_model,
                api_key=settings.qwen_api_key,
                timeout=settings.qwen_vision_timeout,
                requires_key=True,
                disable_thinking=settings.disable_thinking,
            )
        )

    return providers


def build_vision_provider(settings: Settings | None = None) -> OpenAICompatProvider:
    """兼容旧调用：返回视觉主通道。"""
    return build_vision_providers(settings)[0]
