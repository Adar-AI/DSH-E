"""文本分块：按段落边界装箱，超长段落再按句子切分并保留重叠。

分块策略：
1. 先按空行切成段落；
2. 依次把段落装进当前块，直到接近 `chunk_size` 为止（保持段落完整，可读性好）；
3. 单个段落超过 `chunk_size` 时，按中文/英文句子边界切分；仍超长则按字符硬切，
   并在相邻块之间保留 `overlap` 个字符的重叠，避免答案正好被切断；
4. Markdown 额外跟踪标题层级，把标题路径写进 `section`，便于引用定位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingest.loader import LoadedDocument, PageText

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.\s)")


@dataclass(frozen=True)
class TextChunk:
    """一个待入库的知识片段。"""

    ordinal: int
    text: str
    section: str | None = None
    page: int | None = None


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """把超长文本按句子装箱，必要时按字符硬切，并保留重叠。"""
    chunk_size = max(chunk_size, 1)
    overlap = max(min(overlap, chunk_size - 1), 0)

    sentences = [piece.strip() for piece in _SENTENCE_RE.split(text) if piece and piece.strip()]
    if not sentences:
        sentences = [text]

    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            step = chunk_size - overlap
            for start in range(0, len(sentence), max(step, 1)):
                piece = sentence[start : start + chunk_size]
                if piece:
                    pieces.append(piece)
                if start + chunk_size >= len(sentence):
                    break
        elif len(buffer) + len(sentence) <= chunk_size:
            buffer += sentence
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)

    # 给相邻块补重叠（仅硬切场景，段落装箱本身已是自然边界）
    if overlap:
        overlapped: list[str] = []
        for index, piece in enumerate(pieces):
            if index == 0:
                overlapped.append(piece)
                continue
            tail = pieces[index - 1][-overlap:]
            merged = (tail + piece).strip()
            overlapped.append(merged if len(merged) <= chunk_size + overlap else piece)
        pieces = overlapped

    return [piece for piece in pieces if piece.strip()]


def split_plain_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """按段落装箱切分纯文本。"""
    chunk_size = max(chunk_size, 1)
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text or "") if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            split = _hard_split(paragraph, chunk_size, overlap)
            pieces.extend(split[:-1])
            buffer = split[-1] if split else ""
            continue

        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = paragraph

    if buffer:
        pieces.append(buffer)
    return pieces


def split_markdown(text: str, *, chunk_size: int, overlap: int) -> list[tuple[str | None, str]]:
    """按标题切段，返回 [(section 路径, 正文), ...]。"""
    section_stack: list[tuple[int, str]] = []
    blocks: list[tuple[str | None, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            path = " / ".join(title for _, title in section_stack) or None
            blocks.append((path, body))

    for line in (text or "").splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            continue
        buffer.append(line)
    flush()
    return blocks


def split_document(
    document: LoadedDocument,
    *,
    chunk_size: int = 450,
    overlap: int = 80,
) -> list[TextChunk]:
    """把解析结果切成带定位信息（章节 / 页码）的片段。"""
    chunks: list[TextChunk] = []
    ordinal = 0

    if any(page.page is not None for page in document.pages):
        # PDF：逐页切分，页码天然可用
        for page in document.pages:
            for piece in split_plain_text(page.text, chunk_size=chunk_size, overlap=overlap):
                ordinal += 1
                chunks.append(TextChunk(ordinal=ordinal, text=piece, page=page.page))
        return chunks

    # Markdown / 纯文本：跟踪标题路径
    combined = "\n\n".join(page.text for page in document.pages)
    if document.mime == "text/markdown":
        for section, body in split_markdown(combined, chunk_size=chunk_size, overlap=overlap):
            for piece in split_plain_text(body, chunk_size=chunk_size, overlap=overlap):
                ordinal += 1
                chunks.append(TextChunk(ordinal=ordinal, text=piece, section=section))
        return chunks

    for piece in split_plain_text(combined, chunk_size=chunk_size, overlap=overlap):
        ordinal += 1
        chunks.append(TextChunk(ordinal=ordinal, text=piece))
    return chunks


__all__ = [
    "TextChunk",
    "PageText",
    "split_document",
    "split_markdown",
    "split_plain_text",
]
