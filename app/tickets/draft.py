"""工单草稿的增量抽取（规则版基线）。

对话过程中每轮都会更新草稿，右侧栏实时可见（需求 F7）。这里先用**确定性规则**
抓取最关键的字段（型号、报错码、诉求、已尝试操作、紧急程度），好处是：
零延迟、零成本、结果可解释；M5 会用大模型做结构化抽取来补全规则抓不到的语义字段，
两者结果按 `merge_draft` 的策略合并。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

CORE_FIELDS = (
    "customer_request",
    "product_model",
    "fault_phenomenon",
    "error_info",
    "tried_actions",
    "urgency",
)

URGENCY_ORDER = {"普通": 0, "较急": 1, "紧急": 2}

# 报错码：1~3 个字母 + 1~3 位数字（E05、ER12、U4）
# 注意：不能用 \b 做边界——Python 正则里汉字也算 \w，「报错E05」这种没有空格的
# 常见写法下 \b 不成立，会漏掉错误码。改用否定环视：左右只要不是字母数字即可。
ERROR_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,3}\d{1,3})(?![A-Za-z0-9])")
# 产品型号：1~4 个字母 + 3~5 位数字（L3250、M1136、WF2830）
MODEL_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,4}\d{3,5}[A-Za-z]?)(?![A-Za-z0-9])")
# 「型号：xxx」这类显式写法
MODEL_KEYWORD_RE = re.compile(r"(?:型号|机型|产品型号|Model)\s*[:：]?\s*([A-Za-z0-9\-]{3,24})", re.IGNORECASE)

FAULT_KEYWORDS = (
    "报错",
    "错误",
    "不能",
    "无法",
    "失败",
    "卡",
    "不识别",
    "异常",
    "坏",
    "漏",
    "异响",
    "慢",
    "死机",
    "重启",
    "连不上",
    "打不出来",
    "提示",
)

TRIED_KEYWORDS = (
    "已经",
    "试过",
    "试了",
    "重启过",
    "换过",
    "换了",
    "重新安装",
    "重装",
    "重置",
    "清理过",
    "检查过",
    "复位",
)

URGENCY_RULES: Sequence[tuple[str, tuple[str, ...]]] = (
    ("紧急", ("非常急", "紧急", "马上要", "立刻", "停工", "停产", "重大损失", "投诉到")),
    ("较急", ("尽快", "着急", "急用", "投诉", "差评", "退款", "退货", "等着用")),
)

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]+")


def _split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(text or "") if piece.strip()]


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_error_codes(text: str, extra: Iterable[str] | None = None) -> list[str]:
    """抓报错码（文本 + 图片识别结果），去重且保持出现顺序。"""
    found: dict[str, None] = {}
    for match in ERROR_CODE_RE.finditer((text or "").upper()):
        found.setdefault(match.group(1), None)
    for item in extra or []:
        for match in ERROR_CODE_RE.finditer(str(item).upper()):
            found.setdefault(match.group(1), None)
    return list(found)


def extract_product_model(text: str, vision_model: str = "") -> str:
    """抓产品型号：先看「型号：」这类显式写法，再看符合型号特征的编码。"""
    keyword = MODEL_KEYWORD_RE.search(text or "")
    if keyword:
        return keyword.group(1).upper()

    errors = set(extract_error_codes(text))
    for match in MODEL_CODE_RE.finditer((text or "").upper()):
        candidate = match.group(1)
        if candidate not in errors:
            return candidate
    return (vision_model or "").strip().upper()


def detect_urgency(text: str) -> str:
    for level, keywords in URGENCY_RULES:
        if any(keyword in (text or "") for keyword in keywords):
            return level
    return "普通"


def extract_fault_phenomenon(text: str, limit: int = 400) -> str:
    """抓故障现象：优先取含故障关键词的句子，没有就退回整段描述。"""
    sentences = _split_sentences(text)
    hits = [item for item in sentences if any(word in item for word in FAULT_KEYWORDS)]
    source = "；".join(hits) if hits else "；".join(sentences)
    return _truncate(source, limit)


def extract_tried_actions(text: str, limit: int = 200) -> str:
    sentences = _split_sentences(text)
    hits = [item for item in sentences if any(word in item for word in TRIED_KEYWORDS)]
    return _truncate("；".join(hits), limit)


def extract_from_text(
    text: str,
    *,
    vision_detail: dict[str, Any] | None = None,
    answer: str = "",
    citations: Iterable[str] | None = None,
) -> dict[str, Any]:
    """从本轮对话内容里抽取草稿字段（只返回非空字段）。"""
    vision_detail = vision_detail or {}
    vision_codes = vision_detail.get("error_codes") or []
    if isinstance(vision_codes, str):
        vision_codes = [vision_codes]

    draft: dict[str, Any] = {}

    request = _truncate(text, 120)
    if request:
        draft["customer_request"] = request

    model = extract_product_model(text, str(vision_detail.get("product_model") or ""))
    if model:
        draft["product_model"] = model

    phenomenon = extract_fault_phenomenon(text)
    if phenomenon:
        draft["fault_phenomenon"] = phenomenon

    codes = extract_error_codes(text, vision_codes)
    if codes:
        draft["error_info"] = "、".join(codes)

    tried = extract_tried_actions(text)
    if tried:
        draft["tried_actions"] = tried

    urgency = detect_urgency(text)
    if urgency != "普通" or text.strip():
        draft["urgency"] = urgency

    if answer.strip():
        draft["suggestion"] = _truncate(answer, 300)

    citation_list = [item for item in (citations or []) if item]
    if citation_list:
        draft["citations"] = citation_list

    return draft


def _merge_text(existing: str, new: str, *, limit: int, keep_first: bool = False) -> str:
    existing = (existing or "").strip()
    new = (new or "").strip()
    if not new:
        return existing
    if not existing:
        return new
    if keep_first or new in existing:
        return existing
    if existing in new:
        return _truncate(new, limit)
    return _truncate(f"{existing}；{new}", limit)


def merge_draft(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """把新抽取的字段合并进已有草稿。

    策略：客户原始诉求保留第一条；故障现象与已尝试操作追加去重；
    报错码取并集；型号优先保留已有；紧急程度取更高一级；建议与引用取最新。
    """
    merged: dict[str, Any] = dict(existing or {})

    merged["customer_request"] = _merge_text(
        merged.get("customer_request", ""), new.get("customer_request", ""),
        limit=200, keep_first=True,
    )
    merged["fault_phenomenon"] = _merge_text(
        merged.get("fault_phenomenon", ""), new.get("fault_phenomenon", ""), limit=600
    )
    merged["tried_actions"] = _merge_text(
        merged.get("tried_actions", ""), new.get("tried_actions", ""), limit=300
    )
    merged["product_model"] = (
        merged.get("product_model") or new.get("product_model") or ""
    )

    codes: dict[str, None] = {}
    for source in (merged.get("error_info", ""), new.get("error_info", "")):
        for code in re.split(r"[、,，\s]+", str(source or "")):
            if code.strip():
                codes.setdefault(code.strip(), None)
    if codes:
        merged["error_info"] = "、".join(codes)

    current_urgency = str(merged.get("urgency") or "")
    incoming_urgency = str(new.get("urgency") or "")
    if current_urgency or incoming_urgency:
        merged["urgency"] = max(
            (current_urgency or "普通", incoming_urgency or "普通"),
            key=lambda item: URGENCY_ORDER.get(item, 0),
        )

    if new.get("suggestion"):
        merged["suggestion"] = _truncate(str(new["suggestion"]), 400)

    citations: dict[str, None] = {}
    for source in (merged.get("citations") or [], new.get("citations") or []):
        for item in source:
            if item:
                citations.setdefault(str(item), None)
    if citations:
        merged["citations"] = list(citations)[:8]

    return {key: value for key, value in merged.items() if value not in (None, "", [], {})}


def completeness(draft: dict[str, Any]) -> tuple[int, int]:
    """返回 (已填核心字段数, 核心字段总数)，用于页面显示草稿完整度。"""
    filled = sum(1 for field in CORE_FIELDS if str((draft or {}).get(field) or "").strip())
    return filled, len(CORE_FIELDS)
