"""图装配：把节点接成带条件分支的 LangGraph 状态图（v2.1 六节点）。

```
START → prepare ──缺输入──→ reject → END
             ├──有图片──→ vision_understand ─→ intent_align
             └──无图片───────────────────────→ intent_align

intent_align（① 意图确认与改写，第一段判定）
   ├─ low  → reject_low   拒答：说明缺什么、怎么补
   ├─ medium → clarify    反问：列出候选实体，不猜不答
   └─ high → recall_internal（② 内部多路召回）
                 ↓
             fuse_rerank（③ RRF 融合 ④ 重排 ⑤ 动态 TopK；内部不足时在此联网补充）
                 ↓
             post_judge（第二段判定：资料够不够答）
                 ├─ high   → generate_answer（⑥ 生成）
                 ├─ medium → clarify
                 └─ low    → fallback_answer（兜底：资料未收录 + 转人工）
                 ↓
             update_draft → END
```

`checkpointer` 传入 SqliteSaver 后，`thread_id = conversation_id` 即可跨请求、跨重启
保留状态（工单草稿等），满足需求 F1 的「刷新页面不丢上下文」。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.deps import GraphDeps
from app.graph.nodes import (
    build_nodes,
    route_after_intent,
    route_after_judge,
    route_after_prepare,
)
from app.graph.state import GraphState

logger = logging.getLogger(__name__)


def build_graph(deps: GraphDeps, checkpointer: Any | None = None):
    """编译对话图。同一个 `deps` 只服务一次请求（内部持有该请求的数据库连接）。"""
    builder = StateGraph(GraphState)
    for name, node in build_nodes(deps).items():
        builder.add_node(name, node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {
            "reject": "reject",
            "vision_understand": "vision_understand",
            "intent_align": "intent_align",
        },
    )
    builder.add_edge("vision_understand", "intent_align")

    builder.add_conditional_edges(
        "intent_align",
        route_after_intent,
        {
            "reject_low": "reject_low",
            "clarify": "clarify",
            "recall_internal": "recall_internal",
        },
    )

    builder.add_edge("recall_internal", "fuse_rerank")
    builder.add_edge("fuse_rerank", "post_judge")
    builder.add_conditional_edges(
        "post_judge",
        route_after_judge,
        {
            "generate_answer": "generate_answer",
            "clarify": "clarify",
            "fallback_answer": "fallback_answer",
        },
    )

    for terminal in ("generate_answer", "fallback_answer", "clarify", "reject_low"):
        builder.add_edge(terminal, "update_draft")
    builder.add_edge("update_draft", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_graph"]
