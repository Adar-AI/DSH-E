"""业务目标验收测试：对着运行中的服务，逐条验证 5 项业务目标。

用法（需先启动服务）：

    & 'C:\\Users\\19181\\PyCharmMiscProject\\.venv\\Scripts\\python.exe' scripts/acceptance_test.py
    ... scripts/acceptance_test.py --base-url http://127.0.0.1:8000

覆盖（与需求文档的业务目标一一对应）：

    G1 内部资料优先 + 不依赖大模型原生知识（抗幻觉）
    G2 文本 + 图片多模态（报错截图 + 文字描述）
    G3 对话结束生成结构化工单并本地保存
    G4 同一会话多轮上下文记忆
    G5 转人工：结束会话与「转人工提醒」栏的消息对接

退出码 0 表示全部通过。脚本只读取服务，不改代码，可反复执行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "app" / "web" / "img" / "error-e05-sample.png"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str = ""


@dataclass
class Goal:
    title: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.checks)


class Client:
    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, text: str, conversation_id: str | None = None, image: Path | None = None) -> dict:
        """POST /api/chat（SSE），返回 {事件名: 数据}。"""
        data = {"text": text}
        if conversation_id:
            data["conversation_id"] = conversation_id
        files = None
        if image is not None:
            files = {"image": (image.name, image.read_bytes(), "image/png")}
        response = self.http.post("/api/chat", data=data, files=files)
        response.raise_for_status()
        events: dict[str, dict] = {}
        for block in response.text.strip().split("\n\n"):
            name = payload = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
            if name and payload is not None:
                events[name] = payload
        return events

    def get(self, path: str, **params) -> object:
        response = self.http.get(path, params=params or None)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, **params) -> object:
        response = self.http.post(path, params=params or None)
        response.raise_for_status()
        return response.json()

    def post_status(self, path: str, **params) -> int:
        return self.http.post(path, params=params or None).status_code


def wait_ready(client: Client, seconds: int = 60) -> bool:
    for _ in range(seconds * 2):
        try:
            client.get("/api/health")
            return True
        except Exception:  # noqa: BLE001 - 启动等待
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------- G1

def goal1(client: Client) -> Goal:
    goal = Goal("G1 内部资料优先 & 不依赖大模型原生知识")

    hit = client.chat("打印机报错 E05 怎么处理")
    done = hit.get("done", {})
    citations = done.get("citations") or []
    trace = done.get("trace") or {}
    goal.checks.append(
        Check(
            "命中内部资料并给出可核查引用",
            bool(citations) and trace.get("confidence") == "high",
            f"置信={trace.get('confidence')} top1={trace.get('rerank_top1')} 引用={citations}",
        )
    )

    # 引用的文件必须真实存在于内部资料目录（证明答案来自企业文档而不是模型记忆）
    docs_dir = ROOT / "data" / "docs"
    names = {path.name for path in docs_dir.rglob("*") if path.is_file()}
    cited_files = {cite.split(" · ")[0].strip() for cite in citations if cite}
    goal.checks.append(
        Check(
            "引用文件确实存在于本地知识库",
            bool(cited_files) and cited_files <= names,
            f"引用文件={sorted(cited_files)} 库内文件={len(names)} 个",
        )
    )

    # 大模型肯定知道、但企业内部资料里没有的问题 → 必须兜底，不能凭原生知识作答
    off = client.chat("用 Python 读取 CSV 文件应该怎么写")
    off_done = off.get("done", {})
    off_answer = str(off_done.get("answer") or "")
    goal.checks.append(
        Check(
            "资料未收录时兜底（不靠模型原生知识作答）",
            "未收录" in off_answer and not (off_done.get("citations") or []),
            f"回答片段={off_answer[:46].replace(chr(10), ' ')}… 引用={len(off_done.get('citations') or [])} 条",
        )
    )
    return goal


# ---------------------------------------------------------------------- G2

def goal2(client: Client) -> Goal:
    goal = Goal("G2 文本 + 图片多模态")

    if not IMAGE.exists():
        goal.checks.append(Check("测试图片存在", False, f"缺少 {IMAGE}"))
        return goal

    events = client.chat("屏幕上提示这个，帮我看看是什么问题、怎么处理", image=IMAGE)
    vision = events.get("vision", {})
    done = events.get("done", {})
    trace = done.get("trace") or {}
    info = vision.get("image_info") or {}

    goal.checks.append(
        Check(
            "图片被识别（本地视觉模型）",
            vision.get("ok") is True,
            f"模型={vision.get('model')} 本地={vision.get('local')} 耗时信息={info.get('sent_size')}",
        )
    )
    summary = str(vision.get("summary") or "")
    goal.checks.append(
        Check(
            "识别出关键信息（型号 / 错误码）",
            "L3250" in summary.upper() and "E05" in summary.upper(),
            f"识别摘要={summary[:60].replace(chr(10), ' ')}…",
        )
    )
    goal.checks.append(
        Check(
            "图文结合后引用企业内部资料作答",
            bool(done.get("citations")) and trace.get("vision_ok") is True,
            f"引用={done.get('citations')}",
        )
    )
    return goal


# ---------------------------------------------------------------------- G3

def goal3(client: Client) -> Goal:
    goal = Goal("G3 对话结束生成结构化工单并本地保存")

    events = client.chat("我的打印机是 L3250，屏幕显示 E05，硒鼓刚换过还是报错，怎么办")
    conv_id = (events.get("meta") or {}).get("conversation_id")
    ticket = client.post(f"/api/conversations/{conv_id}/close")

    no = str(ticket.get("ticket_no") or "")
    goal.checks.append(
        Check(
            "生成结构化字段的工单",
            bool(no) and ticket.get("product_model") == "L3250" and "E05" in str(ticket.get("error_info") or ""),
            f"工单={no} 型号={ticket.get('product_model')} 报错={ticket.get('error_info')} "
            f"紧急={ticket.get('urgency')} 来源={(ticket.get('extraction') or {}).get('source')}",
        )
    )

    stored = client.get(f"/api/tickets/{ticket['id']}")
    goal.checks.append(
        Check(
            "工单已落本地数据库且可查",
            stored.get("ticket_no") == no and bool(stored.get("events")),
            f"库内工单={stored.get('ticket_no')} 状态={stored.get('status')} 时间线={len(stored.get('events') or [])} 条",
        )
    )

    conversation = client.get(f"/api/conversations/{conv_id}")
    goal.checks.append(
        Check(
            "工单与来源会话关联",
            conversation.get("id") == conv_id,
            f"会话={conv_id} 消息数={conversation.get('message_count')}",
        )
    )

    csv = client.http.get("/api/tickets/export.csv")
    goal.checks.append(
        Check(
            "支持导出 CSV（UTF-8 BOM + 中文表头）",
            csv.status_code == 200 and csv.content.startswith(b"\xef\xbb\xbf") and "工单号" in csv.text,
            f"HTTP {csv.status_code} 字节={len(csv.content)}",
        )
    )
    return goal


# ---------------------------------------------------------------------- G4

def goal4(client: Client) -> Goal:
    goal = Goal("G4 同一会话多轮上下文记忆")

    first = client.chat("我有一台打印机是 L3250，最近老是报错 E05")
    conv_id = (first.get("meta") or {}).get("conversation_id")

    second = client.chat("保修期是多少？", conversation_id=conv_id)
    second_done = second.get("done", {})
    second_intent = second.get("intent", {})
    goal.checks.append(
        Check(
            "缺主语问题靠上下文对齐（问「保修期是多少」不反问、不拒答）",
            (second_done.get("trace") or {}).get("confidence") == "high",
            f"意图档位={second_intent.get('level')} 实体={second_intent.get('entities')} "
            f"查询={(second_intent.get('queries') or [''])[0][:40]}",
        )
    )

    third = client.chat("那人为损坏算不算保修范围？", conversation_id=conv_id)
    third_done = third.get("done", {})
    answer = str(third_done.get("answer") or "")
    goal.checks.append(
        Check(
            "第三轮继续沿用同一会话上下文作答",
            bool(answer) and (third_done.get("trace") or {}).get("confidence") == "high",
            f"回答片段={answer[:50].replace(chr(10), ' ')}…",
        )
    )

    messages = client.get(f"/api/conversations/{conv_id}/messages")
    roles = [item["role"] for item in messages["items"]]
    goal.checks.append(
        Check(
            "多轮消息完整落库",
            roles.count("user") == 3 and roles.count("assistant") >= 2,
            f"角色序列={roles}",
        )
    )
    return goal


# ---------------------------------------------------------------------- G5

def goal5(client: Client) -> Goal:
    goal = Goal("G5 转人工与「转人工提醒」栏的消息对接")

    ask = client.chat("我要转人工客服")
    conv_id = (ask.get("meta") or {}).get("conversation_id")
    handoff = ask.get("handoff") or {}
    goal.checks.append(
        Check(
            "客户要求转人工 → 立即建单并进入提醒栏",
            handoff.get("handoff_state") == "requested" and bool(handoff.get("ticket_no")),
            f"状态={handoff.get('handoff_state')} 工单={handoff.get('ticket_no')}",
        )
    )

    bar = client.get("/api/handoffs")
    in_bar = any(item["id"] == conv_id for item in bar["items"])
    goal.checks.append(
        Check(
            "工单台提醒栏能看到该会话",
            in_bar and bar["waiting"] >= 1,
            f"等待接入={bar['waiting']} 人工中={bar['serving']} 含该会话={in_bar}",
        )
    )

    during = client.chat("在吗？", conversation_id=conv_id)
    during_done = during.get("done", {})
    goal.checks.append(
        Check(
            "人工期间 AI 静默（只登记不抢答）",
            (during_done.get("trace") or {}).get("handoff_used") is True,
            f"trace.handoff_used={((during_done.get('trace') or {}).get('handoff_used'))}",
        )
    )

    before = client.get(f"/api/conversations/{conv_id}/messages")["items"]
    reply = client.post(f"/api/conversations/{conv_id}/agent-messages", text="你好，我是人工客服，正在为你处理")
    after = client.get(
        f"/api/conversations/{conv_id}/messages", after_id=before[-1]["id"]
    )["items"]
    goal.checks.append(
        Check(
            "客服发言送达客户（增量轮询可见）",
            reply.get("handoff_state") == "claimed"
            and any(item["role"] == "agent" for item in after),
            f"状态={reply.get('handoff_state')} 客户增量={[m['role'] for m in after]}",
        )
    )

    last_id = after[-1]["id"] if after else before[-1]["id"]
    finish = client.post(f"/api/conversations/{conv_id}/handoff/finish", sync_fields=False)
    increments = client.get(
        f"/api/conversations/{conv_id}/messages", after_id=last_id
    )["items"]
    notice_ok = any(
        item["role"] == "system" and "AI" in item["text"] and "接管" in item["text"]
        for item in increments
    )
    goal.checks.append(
        Check(
            "结束会话 → 客户收到「AI 接管」提示",
            notice_ok and finish.get("handoff_state") == "done",
            f"提示={[m['text'][:30] for m in increments if m['role'] == 'system']}",
        )
    )

    bar_after = client.get("/api/handoffs")
    gone = all(item["id"] != conv_id for item in bar_after["items"])
    ticket = finish.get("ticket") or {}
    goal.checks.append(
        Check(
            "结束会话与提醒栏对接：该会话移出提醒栏、工单同步已解决",
            gone and ticket.get("status") == "已解决",
            f"提醒栏剩={len(bar_after['items'])} 条 含该会话={not gone} "
            f"工单={ticket.get('ticket_no')} 状态={ticket.get('status')}",
        )
    )

    code = client.post_status(f"/api/conversations/{conv_id}/handoff/finish")
    goal.checks.append(
        Check(
            "重复结束会话被拒绝（幂等保护）",
            code == 404,
            f"第二次结束 HTTP {code}",
        )
    )
    return goal


# ---------------------------------------------------------------------- 主流程

GOALS = (goal1, goal2, goal3, goal4, goal5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="业务目标验收测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--json", default="", help="把结果写入 JSON 文件")
    args = parser.parse_args(argv)

    client = Client(args.base_url)
    if not wait_ready(client):
        print(f"❌ 服务不可达：{args.base_url}（先启动服务再跑验收）")
        return 2

    health = client.get("/api/health")
    docs = health["knowledge"]
    print("=" * 78)
    print("业务目标验收测试")
    print(f"服务：{args.base_url}")
    print(
        f"知识库：{docs['documents']} 份文档 / {docs['chunks']} 分块 / {docs['vectors']} 向量"
        f"｜检索模式={health['retrieval']['mode']}"
    )
    channels = ", ".join(
        f"{item['name']}={item['model']}({'可用' if item['available'] else '不可用'})"
        for item in health["chat"]["channels"]
    )
    print(f"对话通道：{channels}")
    print(f"视觉通道：{health['vision']['model']}（{health['vision']['kind']}）")
    print("=" * 78)

    goals: list[Goal] = []
    for builder in GOALS:
        started = time.perf_counter()
        goal = builder(client)
        goals.append(goal)
        elapsed = time.perf_counter() - started
        print(f"\n【{'PASS' if goal.ok else 'FAIL'}】{goal.title}（{elapsed:.1f}s）")
        for check in goal.checks:
            print(f"   {'✅' if check.ok else '❌'} {check.name}")
            if check.evidence:
                print(f"        {check.evidence}")

    passed = sum(1 for goal in goals if goal.ok)
    checks_total = sum(len(goal.checks) for goal in goals)
    checks_ok = sum(1 for goal in goals for check in goal.checks if check.ok)
    print("\n" + "=" * 78)
    print(f"结果：业务目标 {passed}/{len(goals)} 通过｜检查项 {checks_ok}/{checks_total} 通过")
    print("=" * 78)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "knowledge": docs,
                    "goals": [
                        {
                            "title": goal.title,
                            "ok": goal.ok,
                            "checks": [
                                {"name": c.name, "ok": c.ok, "evidence": c.evidence}
                                for c in goal.checks
                            ],
                        }
                        for goal in goals
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入：{args.json}")

    return 0 if passed == len(goals) else 1


if __name__ == "__main__":
    sys.exit(main())
