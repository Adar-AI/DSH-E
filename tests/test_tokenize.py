"""中文分词模块测试（FTS5 检索的基础）。"""

from __future__ import annotations

from app.rag.tokenize import build_match_query, is_cjk, segment, tokens_of


def test_segment_splits_cjk_into_bigrams():
    assert segment("打印机报错") == "打印 印机 机报 报错"


def test_segment_keeps_single_cjk_char():
    assert segment("硒鼓") == "硒鼓"
    assert segment("换") == "换"


def test_segment_lowercases_ascii_and_keeps_model_codes():
    assert segment("E05") == "e05"
    assert segment("L3250 打印机") == "l3250 打印 印机"


def test_segment_drops_punctuation_and_whitespace():
    assert segment("打开前盖，取出旧硒鼓！") == "打开 开前 前盖 取出 出旧 旧硒 硒鼓"
    assert segment("   ") == ""
    assert segment("") == ""


def test_segment_mixed_cjk_ascii():
    assert segment("报错E05怎么处理") == "报错 e05 怎么 么处 处理"


def test_is_cjk():
    assert is_cjk("打") is True
    assert is_cjk("a") is False
    assert is_cjk("5") is False


def test_tokens_of_dedupes_and_keeps_order():
    assert tokens_of("硒鼓 硒鼓 更换") == ["硒鼓", "更换"]


def test_build_match_query_quotes_every_token():
    assert build_match_query("硒鼓") == '"硒鼓"'
    assert build_match_query("更换硒鼓") == '"更换" OR "换硒" OR "硒鼓"'


def test_build_match_query_escapes_fts_syntax():
    # 用户输入里的 FTS5 语法字符不应被当作语法解析
    query = build_match_query('E05*')
    assert query == '"e05"'


def test_build_match_query_empty_input():
    assert build_match_query("") == ""
    assert build_match_query("！！！") == ""
