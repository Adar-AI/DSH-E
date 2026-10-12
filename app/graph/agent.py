"""图装配：把节点接成带条件分支的 LangGraph 状态图。

```
START → prepare ──缺输入──→ reject → END
             ├──有图片──→ vision_understand ─→ rewrite_query
             └──无图片──────────────────────→ rewrite_query
rewrite_query → hybrid_retrieve → (命中?) ──是──→ generate_answer → update_draft → END
                                          └─否──→ fallback_answer → update_draft → END
```

`checkpointer` 传入 SqliteSaver 后，`thread_id = conversation_id` 即可跨请求、跨重启
保留状态（工单草稿等），满足需求 F1 的「刷新页面不丢上下文」。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.deps import GraphDeps
from app.graph.nodes import build_nodes, route_after_prepare, route_after_retrieve
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
            "rewrite_query": "rewrite_query",
        },
    )
    builder.add_edge("vision_understand", "rewrite_query")
    builder.add_edge("rewrite_query", "hybrid_retrieve")
    builder.add_conditional_edges(
        "hybrid_retrieve",
        route_after_retrieve,
        {"generate_answer": "generate_answer", "fallback_answer": "fallback_answer"},
    )
    builder.add_edge("generate_answer", "update_draft")
    builder.add_edge("fallback_answer", "update_draft")
    builder.add_edge("update_draft", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_graph"]
