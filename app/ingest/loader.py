"""文档解析：PDF / Markdown / 纯文本 → 带页码的文本。

PDF 用 pypdfium2（PDFium）逐页取文本并保留页码；Markdown 与 TXT 直接读文本，
页码记为空（引用时改用章节名定位）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class UnsupportedDocument(ValueError):
    """不支持的文件类型。"""


@dataclass(frozen=True)
class PageText:
    """一页（PDF）或整篇（MD/TXT，page 为 None）的文本。"""

    page: int | None
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    mime: str
    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        return len([page for page in self.pages])

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def _normalize(text: str) -> str:
    """压缩多余空白，保留段落结构。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_mime(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，仅支持 "
            + "、".join(sorted(SUPPORTED_SUFFIXES))
        )
    return MIME_BY_SUFFIX[suffix]


def _load_pdf(path: Path) -> tuple[PageText, ...]:
    import pypdfium2 as pdfium  # 延迟导入：非 PDF 场景不需要

    pdf = pdfium.PdfDocument(str(path))
    pages: list[PageText] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
            finally:
                page.close()
            pages.append(PageText(page=index + 1, text=_normalize(text)))
    finally:
        pdf.close()
    return tuple(pages)


def load_document(path: Path | str) -> LoadedDocument:
    """解析文件。PDF 按页返回，MD/TXT 返回单页（page=None）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    mime = detect_mime(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = _load_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = (PageText(page=None, text=_normalize(text)),)

    return LoadedDocument(filename=path.name, mime=mime, pages=pages)
