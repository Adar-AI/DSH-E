"""工单草稿抽取（规则版）测试。"""

from __future__ import annotations

from app.tickets.draft import (
    completeness,
    detect_urgency,
    extract_error_codes,
    extract_fault_phenomenon,
    extract_from_text,
    extract_product_model,
    extract_tried_actions,
    merge_draft,
)


# ------------------------------------------------------------------ 字段抽取

def test_extract_error_codes_from_text_and_vision():
    assert extract_error_codes("屏幕显示 E05，又出现 E06") == ["E05", "E06"]
    assert extract_error_codes("没有报错") == []
    assert extract_error_codes("", ["E05", "E05", "ER12"]) == ["E05", "ER12"]


def test_extract_error_codes_without_spaces_before_chinese():
    """汉字紧跟错误码是最常见的写法，必须能抽出来（不能用 \\b 做边界）。"""
    assert extract_error_codes("打印机报错E05，怎么处理") == ["E05"]
    assert extract_error_codes("屏幕显示ER12提示") == ["ER12"]
    # 不能把型号误判成错误码
    assert extract_error_codes("型号L3250报错") == []


def test_extract_product_model_without_spaces_before_chinese():
    assert extract_product_model("我用的是L3250打印机") == "L3250"
    assert extract_product_model("机器型号M1136，报错E05") == "M1136"


def test_extract_product_model_prefers_explicit_keyword():
    assert extract_product_model("机型：WF-2830，报错 E05") == "WF-2830"
    assert extract_product_model("我的型号是 M1136") == "M1136"


def test_extract_product_model_from_code_like_token():
    assert extract_product_model("我用的是 L3250 打印机") == "L3250"


def test_extract_product_model_ignores_error_codes():
    # E05 是报错码不是型号，没有型号时应返回空
    assert extract_product_model("报错 E05 怎么办") == ""
    assert extract_product_model("报错 E05 怎么办", vision_model="L3250") == "L3250"


def test_detect_urgency_levels():
    assert detect_urgency("这个问题非常紧急，产线停产了") == "紧急"
    assert detect_urgency("麻烦尽快处理一下") == "较急"
    assert detect_urgency("我要投诉你们") == "较急"
    assert detect_urgency("打印机有点小问题") == "普通"


def test_extract_fault_phenomenon_prefers_fault_sentences():
    text = "打印机报错 E05。今天天气不错。换了硒鼓还是不识别。"
    assert extract_fault_phenomenon(text) == "打印机报错 E05；换了硒鼓还是不识别"


def test_extract_fault_phenomenon_falls_back_to_whole_text():
    assert extract_fault_phenomenon("你好") == "你好"


def test_extract_tried_actions():
    text = "我已经重启过了。也换过硒鼓。请问怎么办？"
    assert "重启" in extract_tried_actions(text)
    assert "换过硒鼓" in extract_tried_actions(text)
    assert extract_tried_actions("请问保修多久") == ""


def test_extract_from_text_full():
    draft = extract_from_text(
        "我的打印机型号 L3250 报错 E05，已经换了硒鼓还是不行，麻烦尽快处理！",
        vision_detail={"error_codes": ["E05"], "product_model": "L3250"},
        answer="请检查硒鼓芯片触点。",
        citations=["产品手册.md · 第 12 页"],
    )

    assert draft["product_model"] == "L3250"
    assert draft["error_info"] == "E05"
    assert draft["urgency"] == "较急"
    assert draft["suggestion"] == "请检查硒鼓芯片触点。"
    assert draft["citations"] == ["产品手册.md · 第 12 页"]
    assert "报错" in draft["fault_phenomenon"]


def test_extract_from_text_empty_input_returns_minimal_draft():
    draft = extract_from_text("")
    assert "customer_request" not in draft
    assert "error_info" not in draft


# ------------------------------------------------------------------ 草稿合并

def test_merge_keeps_first_request_and_appends_new_facts():
    first = {"customer_request": "打印机报错", "fault_phenomenon": "报错 E05"}
    second = {
        "customer_request": "还有别的问题",
        "fault_phenomenon": "换了硒鼓还是不识别",
        "error_info": "E05",
    }

    merged = merge_draft(first, second)

    assert merged["customer_request"] == "打印机报错"  # 原始诉求保留
    assert "报错 E05" in merged["fault_phenomenon"]
    assert "换了硒鼓" in merged["fault_phenomenon"]
    assert merged["error_info"] == "E05"


def test_merge_unions_error_codes_and_keeps_model():
    merged = merge_draft(
        {"error_info": "E05", "product_model": "L3250"},
        {"error_info": "E06、E05", "product_model": "M1136"},
    )

    assert merged["error_info"] == "E05、E06"
    assert merged["product_model"] == "L3250"


def test_merge_takes_higher_urgency_and_latest_suggestion():
    merged = merge_draft(
        {"urgency": "较急", "suggestion": "先检查触点"},
        {"urgency": "紧急", "suggestion": "立即停用并联系售后"},
    )

    assert merged["urgency"] == "紧急"
    assert merged["suggestion"] == "立即停用并联系售后"


def test_merge_keeps_urgency_when_new_is_lower():
    merged = merge_draft({"urgency": "紧急"}, {"urgency": "普通"})
    assert merged["urgency"] == "紧急"


def test_merge_dedupes_citations_and_caps_length():
    existing = {"citations": ["A.md · 第 1 页", "B.md · 第 2 页"]}
    incoming = {"citations": ["B.md · 第 2 页", "C.md · 第 3 页"]}

    merged = merge_draft(existing, incoming)

    assert merged["citations"] == ["A.md · 第 1 页", "B.md · 第 2 页", "C.md · 第 3 页"]


def test_merge_drops_empty_values():
    merged = merge_draft({"customer_request": "报错"}, {"error_info": "", "suggestion": ""})
    assert merged == {"customer_request": "报错"}


def test_completeness_counts_core_fields():
    assert completeness({}) == (0, 6)
    assert completeness({"customer_request": "x", "product_model": "L3250"}) == (2, 6)
    assert completeness({"urgency": "普通", "suggestion": "忽略非核心字段"}) == (1, 6)
