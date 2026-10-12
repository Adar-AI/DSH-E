"""测试替身与素材构造：假嵌入器、合成 PDF、写入临时文件。

放在独立模块里，测试文件通过 `from fakes import ...` 引用（pytest 会把
tests/ 目录加入 sys.path）。
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from app.db import to_vec_blob
from app.rag.embedder import EmbeddingUnavailable
from app.rag.tokenize import segment
from app.utils import now_iso


class HashEmbedder:
    """按 token 哈希成计数向量：共享词越多余弦越高（模拟词面相似，结果确定）。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"
        self.calls: list[str] = []

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in segment(text or "").split():
                index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            out.append([value / norm for value in vector])
            self.calls.append(text)
        return out


class ScriptedEmbedder:
    """按文本精确返回预设向量，用于构造「只有向量通道命中」的场景。"""

    def __init__(
        self,
        mapping: dict[str, list[float]],
        *,
        dim: int = 4,
        default: list[float] | None = None,
    ) -> None:
        self.mapping = mapping
        self.dim = dim
        self.name = "scripted"
        self.default = default if default is not None else [0.0] * dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self.mapping.get(text, self.default)) for text in texts]


class BrokenEmbedder:
    """一用就抛 EmbeddingUnavailable，用于验证降级路径。"""

    name = "broken"
    dim = 4

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingUnavailable("模拟嵌入不可用")


# ------------------------------------------------------------------ 语料装载

def load_corpus(
    conn: sqlite3.Connection,
    entries: Sequence[tuple[str, str, str | None, int | None]],
    embedder=None,
    *,
    status: str = "ready",
) -> list[int]:
    """直接往库里塞语料，返回 chunk_id 列表。

    entries 每项：(文件名, 正文, 章节, 页码)
    """
    chunk_ids: list[int] = []
    for index, (filename, text, section, page) in enumerate(entries):
        conn.execute(
            """
            INSERT INTO documents(filename, sha256, mime, page_count, chunk_count,
                                  source, status, uploaded_at)
            VALUES (?, ?, 'text/markdown', 1, 1, 'test', ?, ?)
            """,
            (filename, f"sha-{index}-{filename}", status, now_iso()),
        )
        doc_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            """
            INSERT INTO chunks(doc_id, ordinal, section, page, text, text_tokens, char_len)
            VALUES (?, 1, ?, ?, ?, ?, ?)
            """,
            (doc_id, section, page, text, segment(text), len(text)),
        )
        chunk_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        if embedder is not None:
            vector = embedder.encode([text])[0]
            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, to_vec_blob(vector)),
            )
        chunk_ids.append(chunk_id)
    conn.commit()
    return chunk_ids


def set_document_status(conn: sqlite3.Connection, filename: str, status: str) -> None:
    conn.execute("UPDATE documents SET status = ? WHERE filename = ?", (status, filename))
    conn.commit()


# ------------------------------------------------------------------ 文件素材

def write_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_pdf(pages: Sequence[str]) -> bytes:
    """生成一个真实可解析的最小 PDF（ASCII 文本，用于验证 PDF 解析与页码）。"""
    count = len(pages)
    font_id = 3 + 2 * count
    objects: dict[int, str] = {}

    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(count))
    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>"

    for index, text in enumerate(pages):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        content = f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects[content_id] = f"<< /Length {len(content)} >>\nstream\n{content}\nendstream"
    objects[font_id] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    total = font_id
    out = "%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n{objects[object_id]}\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {total + 1}\n0000000000 65535 f \n"
    for object_id in sorted(objects):
        out += f"{offsets[object_id]:010d} 00000 n \n"
    out += (
        f"trailer\n<< /Size {total + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    return out.encode("latin-1")


def write_pdf(directory: Path, name: str, pages: Sequence[str]) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf(pages))
    return path


def write_png(
    directory: Path,
    name: str = "screenshot.png",
    size: tuple[int, int] = (8, 8),
    color: tuple[int, int, int] = (220, 60, 60),
) -> Path:
    """生成一张真实的小 PNG，用于验证图片读取与 base64 编码路径。"""
    from PIL import Image

    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ------------------------------------------------------------ 假 OpenAI 客户端

class _FakeMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, *, message=None, delta=None) -> None:
        if message is not None:
            self.message = message
        if delta is not None:
            self.delta = delta


class FakeResponse:
    def __init__(self, content: object) -> None:
        self.choices = [_FakeChoice(message=_FakeMessage(content))]


class _FakeChunk:
    def __init__(self, content: object) -> None:
        self.choices = [_FakeChoice(delta=_FakeMessage(content))]


class _FakeModelEntry:
    def __init__(self, identifier: str) -> None:
        self.id = identifier


class FakeModels:
    def __init__(self, names: Sequence[str] | None = None, error: Exception | None = None) -> None:
        self.names = list(names or [])
        self.error = error

    def list(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=[_FakeModelEntry(name) for name in self.names])


class FakeCompletions:
    """可编排的假补全接口：固定回复、按序回复、抛错重试、流式与流中断。"""

    def __init__(
        self,
        replies: Sequence[object] | None = None,
        *,
        error: Exception | None = None,
        fail_times: int = 0,
        stream_pieces: Sequence[str] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.fail_times = fail_times
        self.stream_pieces = list(stream_pieces or ["流", "式"])
        self.stream_error = stream_error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("模拟网络错误")
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            def generator():
                if self.stream_error is not None:
                    raise self.stream_error
                for piece in self.stream_pieces:
                    yield _FakeChunk(piece)

            return generator()

        reply = self.replies.pop(0) if self.replies else "默认回复"
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    def __init__(self, completions=None, models=None) -> None:
        self.chat = SimpleNamespace(completions=completions or FakeCompletions())
        self.models = models or FakeModels(["fake-model"])


def client_factory_for(client: FakeClient):
    """构造可注入 OpenAICompatProvider 的工厂，并记录收到的连接参数。"""

    def factory(base_url: str, api_key: str, timeout: float) -> FakeClient:
        factory.calls.append((base_url, api_key, timeout))
        return client

    factory.calls = []  # type: ignore[attr-defined]
    return factory


# ------------------------------------------------------- 假对话链 / 假视觉通道

GOOD_VISION_JSON = (
    '{"kind": "错误截图", "product_model": "L3250", "error_codes": ["E05"],'
    ' "screen_location": "屏幕中央", "visible_damage": "", "other_text": "请更换硒鼓",'
    ' "summary": "打印机提示 E05，硒鼓未识别", "confidence": 0.9}'
)

# 工单抽取用的标准 JSON 响应
GOOD_TICKET_JSON = (
    '{"customer_request": "打印机报错 E05，希望尽快解决",'
    ' "product_model": "L3250",'
    ' "fault_phenomenon": "更换硒鼓后仍提示硒鼓未识别",'
    ' "error_info": "E05",'
    ' "tried_actions": "客户已经更换了新硒鼓",'
    ' "urgency": "较急",'
    ' "suggestion": "检查硒鼓芯片触点并复位计数器"}'
)


class FakeChatChain:
    """假对话链：`stream()` 逐字产出，`chat()` 一次返回，可配置为直接失败。"""

    def __init__(
        self,
        text: str = "根据资料 [1]，请先检查硒鼓芯片触点是否清洁。",
        *,
        error: str | None = None,
        stream_error: str | None = None,
        provider: str = "fake",
        model: str = "fake-model",
    ) -> None:
        self.text = text
        self.error = error
        self.stream_error = stream_error
        self.provider = provider
        self.model = model
        # 供 /api/health 之类按 provider 遍历的代码使用
        self.name = provider
        self.base_url = "fake://local"
        self.calls: list[list[dict]] = []

    @property
    def providers(self) -> list[object]:
        return [self]

    def available(self) -> tuple[bool, str]:
        return (self.error is None, "已配置" if self.error is None else str(self.error))

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": self.provider,
                "model": self.model,
                "base_url": "fake://local",
                "available": self.error is None,
                "reason": "已配置" if self.error is None else str(self.error),
            }
        ]

    def ping(self) -> tuple[bool, str]:
        if self.error:
            return False, str(self.error)
        return True, "连通"

    def chat(self, messages, **overrides):
        from app.llm.providers import ChatOutcome, ProviderError

        self.calls.append(list(messages))
        if self.error:
            raise ProviderError(str(self.error))
        return ChatOutcome(text=self.text, provider=self.provider, model=self.model)

    def stream(self, messages, **overrides):
        from app.llm.providers import ProviderError

        self.calls.append(list(messages))
        if self.error:
            raise ProviderError(str(self.error))

        def generator():
            # 模拟「流到一半断掉」：先产出两个字符再报错
            if self.stream_error:
                for piece in list(self.text)[:2]:
                    yield piece
                raise ProviderError(str(self.stream_error))
            for piece in list(self.text):
                yield piece

        return self.provider, generator()


class StubVisionProvider:
    """假视觉通道：直接返回预设 JSON 文本。"""

    def __init__(
        self,
        reply: str = GOOD_VISION_JSON,
        *,
        available: bool = True,
        reason: str = "已配置",
        error: str | None = None,
        name: str = "stub-vision",
        model: str = "stub-vision-model",
        base_url: str = "fake://vision",
    ) -> None:
        self.reply = reply
        self._available = available
        self.reason = reason
        self.error = error
        self.name = name
        self.model = model
        self.base_url = base_url
        self.calls: list[list[dict]] = []

    def available(self) -> tuple[bool, str]:
        return self._available, self.reason

    def chat(self, messages, **overrides):
        from app.llm.providers import ProviderError

        self.calls.append(list(messages))
        if self.error:
            raise ProviderError(str(self.error))
        return self.reply

    def ping(self) -> tuple[bool, str]:
        return self.available()


class BrokenVisionProvider(StubVisionProvider):
    """总是调用失败，用于验证视觉通道回退。"""

    def __init__(self, *, error: str = "本地视觉模型不可用", **kwargs) -> None:
        kwargs.setdefault("name", "vision-local-broken")
        kwargs.setdefault("base_url", "http://127.0.0.1:11434/v1")
        super().__init__(error=error, **kwargs)


class StubReranker:
    """可控假重排器：关键字规则给分，便于对图逻辑做确定性断言。

    规则是**查询感知**的：只有关键字同时出现在查询与文本里才给分
    （真实交叉编码器的相关性本来就取决于两侧，忽略查询的打分器会给出误导性的高分）。

    真实重排质量由 tests/test_reranker.py 用真模型验证；这里只需要"能稳定拉开差距"。
    """

    name = "stub-reranker"

    def __init__(
        self,
        rules: Sequence[tuple[str, float]] | None = None,
        *,
        default: float = 0.02,
        error: str | None = None,
    ) -> None:
        self.rules = list(
            rules
            if rules is not None
            else [("E05", 0.95), ("保修", 0.88), ("网络", 0.90), ("回收", 0.91), ("卡纸", 0.93)]
        )
        self.default = default
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        from app.llm.providers import ProviderError

        self.calls.append((query, list(texts)))
        if self.error:
            raise ProviderError(self.error)

        active = [(keyword, score) for keyword, score in self.rules if keyword in query]
        scores: list[float] = []
        for text in texts:
            value = self.default
            for keyword, score in active:
                if keyword in text:
                    value = max(value, score)
            scores.append(value)
        return scores

    def health(self) -> tuple[bool, str]:
        return True, "stub-reranker 就绪"


def make_stub_reranker(**kwargs) -> StubReranker:
    return StubReranker(**kwargs)
