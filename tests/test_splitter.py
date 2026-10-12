"""分块测试：段落装箱、超长段落硬切与重叠、Markdown 章节、页码与序号。"""

from __future__ import annotations

from app.ingest.loader import LoadedDocument, PageText
from app.ingest.splitter import split_document, split_markdown, split_plain_text


def test_split_plain_text_packs_paragraphs():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    pieces = split_plain_text(text, chunk_size=12, overlap=0)

    assert len(pieces) >= 2
    assert all(len(piece) <= 12 + 12 for piece in pieces)
    assert "".join(pieces).replace("\n", "").count("段内容") == 3


def test_split_plain_text_keeps_short_text_whole():
    pieces = split_plain_text("只有一段很短的话。", chunk_size=100, overlap=0)
    assert pieces == ["只有一段很短的话。"]


def test_split_plain_text_empty_input():
    assert split_plain_text("", chunk_size=100, overlap=0) == []
    assert split_plain_text("   \n\n  ", chunk_size=100, overlap=0) == []


def test_long_paragraph_is_hard_split_with_overlap():
    paragraph = "这是一个很长的句子。" * 12  # 120 字，无空行
    pieces = split_plain_text(paragraph, chunk_size=40, overlap=8)

    assert len(pieces) >= 3
    for piece in pieces:
        assert len(piece) <= 40 + 8
    # 相邻块之间存在重叠内容
    assert pieces[1][:8] == pieces[0][-8:]


def test_split_markdown_tracks_nested_sections():
    text = (
        "# 使用手册\n\n"
        "## 耗材更换\n\n打开前盖取出旧硒鼓。\n\n"
        "## 报错处理\n\n### E05\n\n屏幕显示 E05 表示硒鼓未识别。\n"
    )

    blocks = split_markdown(text, chunk_size=200, overlap=0)
    sections = [section for section, _ in blocks]

    assert sections == ["使用手册 / 耗材更换", "使用手册 / 报错处理 / E05"]
    assert "硒鼓" in blocks[0][1]
    assert "E05" in blocks[1][1]


def test_split_document_numbers_ordinals_for_markdown():
    document = LoadedDocument(
        filename="手册.md",
        mime="text/markdown",
        pages=(PageText(page=None, text="# 一\n\n内容一。\n\n# 二\n\n内容二。"),),
    )

    chunks = split_document(document, chunk_size=100, overlap=0)

    assert [chunk.ordinal for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert [chunk.section for chunk in chunks] == ["一", "二"]
    assert all(chunk.page is None for chunk in chunks)


def test_split_document_keeps_pdf_pages():
    document = LoadedDocument(
        filename="手册.pdf",
        mime="application/pdf",
        pages=(
            PageText(page=1, text="第一页内容。"),
            PageText(page=2, text="第二页内容。"),
        ),
    )

    chunks = split_document(document, chunk_size=100, overlap=0)

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.ordinal for chunk in chunks] == [1, 2]
    assert all(chunk.section is None for chunk in chunks)


def test_split_document_respects_chunk_size():
    document = LoadedDocument(
        filename="长文.txt",
        mime="text/plain",
        pages=(PageText(page=None, text="句子内容重复。" * 30),),
    )

    chunks = split_document(document, chunk_size=50, overlap=5)

    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 55 for chunk in chunks)
