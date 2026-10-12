"""三档置信判定：高置信检索 / 中置信反问 / 低置信拒答。

为什么分两段（见 `docs/架构说明.md` 13.2、13.6 的实测依据）：

* **第一段（检索前粗判）**只解决"问得清不清楚"。实测发现重排分判不出这一点：
  语义不全的问题 top1 分布在 0.015~0.355，还有 0.355 这种"有点像"的情况；
* **第二段（检索后终判）**只解决"资料够不够答"，用重排 top1 判定，分离度极好
  （可回答问题 ≥0.613、无关问题 0.000）。

两段各管一件事，缺一不可：第一段防止"缺主语时瞎猜"，第二段防止"资料不相关时硬答"。

本模块**不调用模型**（话术是确定性模板，理由同兜底话术：这类回复必须逐字可控）。
`LLM_ALIGN_ENABLED=true` 时，边界情形会标记 `needs_llm_align=True`，由图节点决定是否再问一次模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.config import Settings, get_settings
from app.tickets.draft import ERROR_CODE_RE, MODEL_CODE_RE

LEVEL_HIGH = "high"
LEVEL_MEDIUM = "medium"
LEVEL_LOW = "low"

# 答案随产品/型号变化的问法：这类问题缺主语时必须先问清楚，不能猜
ENTITY_DEPENDENT_HINTS = (
    "保修",
    "质保",
    "保几年",
    "价格",
    "多少钱",
    "报价",
    "耗材型号",
    "硒鼓型号",
    "墨盒型号",
    "寿命",
    "更换周期",
    "多久换",
    "兼容",
    "规格",
    "参数",
    "尺寸",
    "重量",
    "电压",
    "功率",
    "耗电量",
)

# 疑问语气：用于判断"这是一个需要实体的问题"还是陈述句
QUESTION_HINTS = ("?", "？", "多少", "多久", "多长", "怎么", "如何", "为什么", "能不能", "是否", "哪个", "哪款", "吗")

# 明确指代本机/本产品的词：出现这些词说明主语在句子内，不算缺主语
SELF_REFERENCE_HINTS = ("这台", "这个机器", "本机", "我买的", "我的打印机", "这台机器")


@dataclass(frozen=True)
class EntityHint:
    """实体池里的一项（来自会话历史 / 图片识别 / 工单草稿）。"""

    name: str
    kind: str = "product_model"  # product_model | error_code
    source: str = "history"  # history | image | draft
    weight: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "source": self.source, "weight": self.weight}


@dataclass
class PreJudgement:
    """第一段判定结果（检索前）。"""

    level: str = LEVEL_HIGH
    entities: list[str] = field(default_factory=list)  # 从当前问题里直接抽到的实体
    context_entities: list[str] = field(default_factory=list)  # 从上下文对齐来的实体
    reason: str = ""
    clarify_question: str = ""
    reject_message: str = ""
    needs_llm_align: bool = False
    rewritten_query: str = ""

    @property
    def should_reject(self) -> bool:
        return self.level == LEVEL_LOW

    @property
    def should_clarify(self) -> bool:
        return self.level == LEVEL_MEDIUM

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "entities": self.entities,
            "context_entities": self.context_entities,
            "reason": self.reason,
            "needs_llm_align": self.needs_llm_align,
            "clarify_question": self.clarify_question,
            "reject_message": self.reject_message,
        }


@dataclass
class FinalJudgement:
    """第二段判定结果（检索 + 重排之后）。"""

    level: str = LEVEL_HIGH
    top1: float = 0.0
    kept: int = 0
    reason: str = ""
    clarify_question: str = ""
    reject_message: str = ""

    @property
    def should_reject(self) -> bool:
        return self.level == LEVEL_LOW

    @property
    def should_clarify(self) -> bool:
        return self.level == LEVEL_MEDIUM

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "top1": round(self.top1, 4),
            "kept": self.kept,
            "reason": self.reason,
        }


# ------------------------------------------------------------------ 实体抽取

def extract_entities(text: str) -> list[tuple[str, str]]:
    """从一段文字里抽出 (实体, 类型)：型号与错误码，去重保序。"""
    upper = (text or "").upper()
    found: dict[str, str] = {}

    errors = {match.group(1) for match in ERROR_CODE_RE.finditer(upper)}
    for code in sorted(errors):
        found.setdefault(code, "error_code")
    for match in MODEL_CODE_RE.finditer(upper):
        candidate = match.group(1)
        if candidate not in errors:
            found.setdefault(candidate, "product_model")
    return list(found.items())


def collect_entity_pool(
    *,
    history_texts: Iterable[str] = (),
    draft: dict | None = None,
    vision_detail: dict | None = None,
) -> list[EntityHint]:
    """汇总上下文里出现过的实体：草稿与图片最可信，历史次之。"""
    pool: dict[str, EntityHint] = {}

    def add(name: str, kind: str, source: str, weight: float) -> None:
        name = str(name or "").strip().upper()
        if not name:
            return
        existing = pool.get(name)
        if existing is None or existing.weight < weight:
            pool[name] = EntityHint(name=name, kind=kind, source=source, weight=weight)

    vision_detail = vision_detail or {}
    add(vision_detail.get("product_model", ""), "product_model", "image", 1.0)
    for code in vision_detail.get("error_codes") or []:
        add(code, "error_code", "image", 1.0)

    draft = draft or {}
    add(draft.get("product_model", ""), "product_model", "draft", 1.0)
    for code in re.split(r"[、,，\s]+", str(draft.get("error_info") or "")):
        add(code, "error_code", "draft", 1.0)

    for text in history_texts:
        for name, kind in extract_entities(str(text or "")):
            add(name, kind, "history", 0.6)

    # 排序：来源可信度优先 → 型号优先于错误码（型号才是澄清时该问的东西）→ 名称
    return sorted(
        pool.values(),
        key=lambda item: (-item.weight, 0 if item.kind == "product_model" else 1, item.name),
    )


def has_inline_entity(question: str) -> bool:
    """问题本身是否带实体（带了就不算缺主语）。"""
    return bool(extract_entities(question))


def is_entity_dependent(question: str) -> bool:
    """问题是否属于"答案随产品/型号变化"的类型。"""
    text = question or ""
    return any(hint in text for hint in ENTITY_DEPENDENT_HINTS)


def looks_like_question(question: str) -> bool:
    text = question or ""
    return any(hint in text for hint in QUESTION_HINTS)


# ------------------------------------------------------------------ 话术模板

def build_clarify_message(question: str, candidates: Sequence[str], *, extra_options: Sequence[str] = ()) -> str:
    """中置信：只反问、不给答案（用户确认的行为）。"""
    options = [item for item in candidates if item]
    if len(options) < 2:
        options = list(dict.fromkeys([*options, *extra_options]))[:3]

    option_text = "、".join(options) if options else "具体型号"
    example = options[0] if options else "L3250"
    return (
        f"为了给你准确的答复，我需要先确认一下你问的是哪一款产品：**{option_text}**。\n\n"
        f"不同型号在这方面的规定不一样（比如整机与耗材的保修期就不同），"
        f"所以我不猜——你回复具体型号（例如「{example} 的{_topic_of(question)}」），我立刻查内部资料回答你。"
    )


def build_reject_message(question: str, *, missing: str = "产品型号") -> str:
    """低置信：拒答（明确说明缺什么、怎么补），与"资料未收录"的兜底话术区分开。"""
    topic = _topic_of(question)
    return (
        f"抱歉，这个问题缺少必要信息，我不能凭猜测回答「{topic}」。\n\n"
        f"请补充以下信息之一：\n"
        f"1. **{missing}**（机身标签或设置菜单里能看到，例如 L3250 / M1136）；\n"
        f"2. 具体现象或屏幕上完整的提示文字（例如「屏幕显示 E05」）。\n\n"
        f"补充后我会立刻查内部资料给你答复；如果着急，也可以直接转人工客服。"
    )


def _topic_of(question: str) -> str:
    """从问题里摘一个短主题词，用于话术里回指客户的问题。"""
    text = re.sub(r"[？?！!。，,、\s]+$", "", (question or "").strip())
    return text[:20] or "这个问题"


# ------------------------------------------------------------------ 第一段判定

def judge_before_recall(
    question: str,
    *,
    context_entities: Sequence[EntityHint] = (),
    extra_options: Sequence[str] = (),
    settings: Settings | None = None,
) -> PreJudgement:
    """检索前粗判：只判断"问得清不清楚"。

    规则（确定性，0ms）：
    1. 问题自带型号/错误码 → high（主语明确）；
    2. 问题需要实体（保修期/价格/耗材型号…）但没有实体：
       - 上下文里只有一个候选实体 → high（对齐到它，并把实体写进检索查询）；
       - 上下文里有多个候选 → medium（反问澄清）；
       - 上下文里没有候选 → low（拒答，请补充型号）；
    3. 其余问题 → high（交给检索与重排判定）。
    """
    settings = settings or get_settings()
    inline = [name for name, _ in extract_entities(question)]
    pool = [item for item in context_entities if item.name]

    if inline:
        return PreJudgement(
            level=LEVEL_HIGH,
            entities=inline,
            context_entities=[],
            reason=f"问题自带实体：{'、'.join(inline)}",
            rewritten_query=question,
        )

    if not is_entity_dependent(question):
        return PreJudgement(
            level=LEVEL_HIGH,
            reason="问题不依赖具体型号，直接检索",
            rewritten_query=question,
        )

    model_candidates = [item.name for item in pool if item.kind == "product_model"]
    if len(model_candidates) == 1:
        model = model_candidates[0]
        return PreJudgement(
            level=LEVEL_HIGH,
            context_entities=[model],
            reason=f"问题缺主语，但对齐到上下文唯一型号 {model}",
            rewritten_query=f"{model} {question}",
        )

    if len(model_candidates) > 1:
        options = list(dict.fromkeys([*model_candidates, *extra_options]))
        return PreJudgement(
            level=LEVEL_MEDIUM,
            context_entities=model_candidates,
            reason=f"问题缺主语且上下文出现多个型号：{'、'.join(model_candidates)}",
            clarify_question=build_clarify_message(question, model_candidates, extra_options=extra_options),
            needs_llm_align=True,
        )

    # 上下文里连型号都没有
    return PreJudgement(
        level=LEVEL_LOW,
        reason="问题缺主语，且会话中没有可对齐的型号",
        reject_message=build_reject_message(question),
        needs_llm_align=True,
    )


# ------------------------------------------------------------------ 第二段判定

def judge_after_recall(
    question: str,
    *,
    top1: float,
    kept: int,
    pre: PreJudgement | None = None,
    extra_options: Sequence[str] = (),
    settings: Settings | None = None,
) -> FinalJudgement:
    """检索后终判：只看"资料够不够答"。"""
    settings = settings or get_settings()
    high = float(settings.confidence_high)
    low = float(settings.confidence_low)

    if kept <= 0 or top1 < low:
        return FinalJudgement(
            level=LEVEL_LOW,
            top1=top1,
            kept=kept,
            reason=f"重排 top1={top1:.3f} 低于低置信阈值 {low}（保留 {kept} 条）",
            reject_message=build_reject_message(question, missing="更具体的现象或型号"),
        )

    if top1 >= high:
        return FinalJudgement(
            level=LEVEL_HIGH,
            top1=top1,
            kept=kept,
            reason=f"重排 top1={top1:.3f} ≥ 高置信阈值 {high}",
        )

    # 中间地带：证据有但不够硬
    candidates = list(pre.context_entities) if pre else []
    return FinalJudgement(
        level=LEVEL_MEDIUM,
        top1=top1,
        kept=kept,
        reason=f"重排 top1={top1:.3f} 落在 {low}~{high} 之间，证据不足",
        clarify_question=build_clarify_message(question, candidates, extra_options=extra_options),
    )


__all__ = [
    "ENTITY_DEPENDENT_HINTS",
    "EntityHint",
    "FinalJudgement",
    "LEVEL_HIGH",
    "LEVEL_LOW",
    "LEVEL_MEDIUM",
    "PreJudgement",
    "build_clarify_message",
    "build_reject_message",
    "collect_entity_pool",
    "extract_entities",
    "has_inline_entity",
    "is_entity_dependent",
    "judge_after_recall",
    "judge_before_recall",
    "looks_like_question",
]
