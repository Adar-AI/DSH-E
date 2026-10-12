"""文档解析测试：Markdown / TXT / PDF（含页码）与异常路径。"""

from __future__ import annotations

import pytest

from app.ingest.loader import (
    UnsupportedDocument,
    detect_mime,
    load_document,
    sha256_of,
)
from fakes import write_file, write_pdf


def test_load_markdown(work_dir):
    path = write_file(work_dir, "手册.md", "# 标题\n\n正文内容。\n")
    document = load_document(path)

    assert document.filename == "手册.md"
    assert document.mime == "text/markdown"
    assert document.page_count == 1
    assert document.pages[0].page is None
    assert "正文内容" in document.text


def test_load_txt_normalizes_whitespace(work_dir):
    path = write_file(work_dir, "规则.txt", "第一行   有空格\r\n\n\n\n第二段\t制表符")
    document = load_document(path)

    assert document.mime == "text/plain"
    assert "第一行 有空格" in document.text
    assert "\n\n\n" not in document.text
    assert "第二段 制表符" in document.text


def test_load_pdf_keeps_page_numbers(work_dir):
    path = write_pdf(work_dir, "彩页.pdf", ["Error E05 drum unit", "Warranty two years"])

    document = load_document(path)

    assert document.mime == "application/pdf"
    assert document.page_count == 2
    assert [page.page for page in document.pages] == [1, 2]
    assert "E05" in document.pages[0].text
    assert "Warranty" in document.pages[1].text


def test_detect_mime_rejects_unsupported(work_dir):
    path = write_file(work_dir, "说明.docx", "x")

    with pytest.raises(UnsupportedDocument):
        detect_mime(path)


def test_load_document_rejects_unsupported(work_dir):
    path = write_file(work_dir, "图纸.png", "not really an image")

    with pytest.raises(UnsupportedDocument):
        load_document(path)


def test_load_missing_file_raises(work_dir):
    with pytest.raises(FileNotFoundError):
        load_document(work_dir / "不存在.md")


def test_sha256_is_content_based(work_dir):
    first = write_file(work_dir, "a.md", "同样的内容")
    second = write_file(work_dir, "b.md", "同样的内容")
    third = write_file(work_dir, "c.md", "不同的内容")

    assert sha256_of(first) == sha256_of(second)
    assert sha256_of(first) != sha256_of(third)
