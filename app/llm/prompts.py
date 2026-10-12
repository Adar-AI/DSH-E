"""提示词集中管理。

反幻觉约束（需求第 8 节）在这里落地：回答只能依据检索到的资料片段，
资料没写就明确说没写，并用 [序号] 标注来源。
"""

from __future__ import annotations

from string import Template
from typing import Sequence

# ---------------------------------------------------------------- 图片理解

VISION_SYSTEM = """你是企业售后客服的图片分析助手。
你只描述图片中**确实能看到**的内容，不要猜测、不要编造看不到的信息。
看不清的项目留空字符串，并降低置信度。"""

# 用 Template 而不是 str.format：提示词里带 JSON 花括号，format 会把它们当占位符
VISION_PROMPT_TEMPLATE = Template("""请分析这张客户上传的图片（可能是报错截图、设备屏幕、实物照片或产品标签），
提取售后排查需要的信息，并**只输出 JSON**，不要输出任何解释文字：

{
  "kind": "错误截图 | 设备屏幕 | 实物照片 | 产品标签 | 其他",
  "product_model": "能识别到的产品型号，没有就空字符串",
  "error_codes": ["图片中出现的错误码或提示文字，例如 E05"],
  "screen_location": "报错出现在界面的什么位置，或空字符串",
  "visible_damage": "肉眼可见的损坏/异常（如卡纸、漏墨、外壳破损），没有就空字符串",
  "other_text": "图片中其他与故障相关的文字",
  "summary": "一句话中文概括这张图说明了什么问题",
  "confidence": 0.0
}

客户的文字描述（可能为空，仅作参考，不要被它带偏）：
$context""")


def build_vision_prompt(context: str = "") -> str:
    """渲染图片理解提示词（安全替换，不受 JSON 花括号影响）。"""
    return VISION_PROMPT_TEMPLATE.substitute(context=(context or "（无）").strip())


# ---------------------------------------------------------------- 回答生成

ANSWER_SYSTEM = """你是企业售后客服助手。请严格按以下规则回答：

1. **只能依据【资料片段】中的内容回答**。禁止使用你自己的知识、常识或经验补充任何信息。
2. 【资料片段】里没有提到的内容，必须明确回答「内部资料中未收录该信息」，并建议转人工客服，
   绝对不要猜测或编造数值、保修期限、型号、步骤。
3. 每条结论后面用 [序号] 标注依据来自哪个片段，例如：保修期为两年 [2]。
4. 使用简体中文，条理清晰，优先给出可执行的处理步骤。
5. 如果资料片段之间互相矛盾，指出矛盾点并建议人工确认。
6. 如果客户上传了图片，把图片分析结果作为已知事实一并考虑，但仍需以资料片段为准。"""

FALLBACK_SYSTEM = """你是企业售后客服助手。当前内部资料没有检索到与该问题相关的内容。"""

FALLBACK_TEMPLATE = """抱歉，企业内部资料中未收录与「{question}」相关的内容，我不能凭猜测给你答复。

建议这样做：
1. 补充更具体的信息（产品型号、屏幕上的完整报错文字、故障发生的操作步骤）；
2. 或者直接转人工客服，由人工帮你确认。

{extra}"""


def format_contexts(contexts: Sequence[object]) -> str:
    """把检索片段编号排成提示词里的【资料片段】块。

    contexts 支持两种元素：带 `citation`/`text` 属性的对象（RetrievedChunk），
    或形如 {"citation": ..., "text": ...} 的字典。
    """
    if not contexts:
        return "（无）"

    lines: list[str] = []
    for index, item in enumerate(contexts, start=1):
        if isinstance(item, dict):
            citation = str(item.get("citation") or item.get("filename") or "未知来源")
            text = str(item.get("text") or "")
        else:
            citation = str(getattr(item, "citation", "未知来源"))
            text = str(getattr(item, "text", ""))
        lines.append(f"[{index}] 来源：{citation}\n{text.strip()}")
    return "\n\n".join(lines)


def build_answer_messages(
    question: str,
    contexts: Sequence[object],
    *,
    vision_summary: str = "",
    history: Sequence[dict] | None = None,
) -> list[dict]:
    """组装回答请求（含多轮历史 + 图片理解结果 + 资料片段）。"""
    messages: list[dict] = [{"role": "system", "content": ANSWER_SYSTEM}]

    for item in history or []:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    vision_block = (
        f"\n\n【客户上传图片的分析结果】\n{vision_summary.strip()}" if vision_summary else ""
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"【客户问题】\n{question.strip()}"
                f"{vision_block}\n\n"
                f"【资料片段】\n{format_contexts(contexts)}\n\n"
                "请依据以上资料片段给出售后处理建议，并按规则标注来源。"
            ),
        }
    )
    return messages


def build_fallback_messages(
    question: str,
    *,
    vision_summary: str = "",
    history: Sequence[dict] | None = None,
) -> list[dict]:
    """未命中时的兜底回答请求（仍然过一遍模型，让措辞自然）。"""
    messages: list[dict] = [{"role": "system", "content": FALLBACK_SYSTEM}]

    for item in history or []:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    extra = f"【图片分析结果】{vision_summary.strip()}" if vision_summary else ""
    messages.append(
        {
            "role": "user",
            "content": (
                f"客户问题：{question.strip()}\n"
                "内部资料中没有检索到相关内容。请按下面模板回复（可以润色措辞，但不要添加任何"
                "资料里没有的技术细节）：\n\n"
                + FALLBACK_TEMPLATE.format(question=question.strip(), extra=extra)
            ),
        }
    )
    return messages
