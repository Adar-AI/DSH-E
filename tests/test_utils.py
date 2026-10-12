"""通用工具测试：JSON 抽取、时间戳、ID。"""

from __future__ import annotations

import re

from app.utils import new_id, now_iso, parse_json_block


def test_parse_json_block_direct_json():
    assert parse_json_block('{"a": 1}') == {"a": 1}


def test_parse_json_block_with_fence():
    text = '```json\n{"summary": "硒鼓未识别", "error_codes": ["E05"]}\n```'
    assert parse_json_block(text) == {"summary": "硒鼓未识别", "error_codes": ["E05"]}


def test_parse_json_block_with_prose_around():
    text = '好的，分析结果如下：\n{"kind": "错误截图"}\n希望有帮助！'
    assert parse_json_block(text) == {"kind": "错误截图"}


def test_parse_json_block_invalid_returns_none():
    assert parse_json_block("完全没有 JSON") is None
    assert parse_json_block("{不是合法 JSON}") is None
    assert parse_json_block("") is None
    assert parse_json_block(None) is None


def test_parse_json_block_ignores_non_object():
    assert parse_json_block("[1, 2, 3]") is None


def test_parse_json_block_tolerates_raw_newlines_inside_strings():
    """模型常把多行文本直接塞进字段值——标准 JSON 不允许，但必须能解析。"""
    text = '{\n  "suggestion": "第一步：检查触点\n第二步：复位计数器",\n  "error_info": "E05"\n}'

    data = parse_json_block(text)

    assert data is not None
    assert data["error_info"] == "E05"
    assert "第一步" in data["suggestion"] and "第二步" in data["suggestion"]


def test_parse_json_block_tolerates_tabs_in_strings():
    data = parse_json_block('{"note": "带\\t制表符"}')
    assert data["note"] == "带\t制表符"


def test_now_iso_format():
    value = now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+00:00)?$", value)


def test_new_id_has_prefix_and_is_unique():
    first = new_id("conv")
    second = new_id("conv")

    assert first.startswith("conv-")
    assert first != second
    assert len(first) == len("conv-") + 12
