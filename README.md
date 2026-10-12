# DSH-E · 多模态企业智能客服系统

远程仓库：<https://github.com/Adar-AI/DSH-E>

面向中小企业售后场景、**单机本地运行**的 Web 应用：客户用文字 + 故障截图/实物照片提问，系统**只依据企业内部资料**（PDF/MD 知识库）给出售后方案，并在会话结束时自动生成结构化工单归档，供人工客服跟进。

- 多模态输入：文字 + 图片（报错截图 / 实物照片）
- 私域知识优先：本地 BGE-M3 向量 + SQLite FTS5 混合检索，检索不到就明确说"资料未收录"，不编造
- 多轮记忆：LangGraph + SQLite checkpointer，刷新页面不丢上下文
- 工单闭环：对话中增量抽取，会话结束定稿落库，工单台可流转状态并导出

## 文档

| 文档 | 内容 |
|---|---|
| [需求规格说明书](docs/需求规格说明书.md) | 角色场景、11 项功能需求与验收标准、非功能需求、界面规范、里程碑 |
| [架构说明](docs/架构说明.md) | 总体架构、模块职责、数据模型、检索设计、LangGraph 图、接口契约、测试策略 |

## 技术栈

FastAPI + uvicorn · LangGraph（含 SQLite checkpointer） · SQLite（FTS5 + sqlite-vec） · 本地 BGE-M3（1024 维） · Ollama `qwen3:4b`（默认对话，DeepSeek API 备用） · OpenAI 兼容多模态网关（图片理解） · 原生 HTML/CSS/JS 前端（淡蓝色）

## 快速开始

```powershell
# 1. 安装依赖（复用本机 Python 环境）
& 'E:\PyCharmMiscProject\python.exe' -m pip install -r requirements.txt

# 2. 准备配置：复制模板后填入密钥（.env 已被 .gitignore 忽略）
Copy-Item .env.example .env

# 3. 准备本地对话模型（本机已有 qwen2.5:latest，可直接用；否则拉一个指令模型）
ollama list
ollama pull qwen3:4b    # 可选：更小更快；不拉也能用 qwen2.5:latest
ollama serve

# 4. 启动服务（脚本随里程碑补充，最终为 scripts\run.ps1）
& 'E:\PyCharmMiscProject\python.exe' scripts\run.py
# 浏览器打开 http://127.0.0.1:8000
```

## 目录结构

```
app/        后端：API、LangGraph 图、检索、入库、工单
app/web/    前端：对话页 index.html、工单台 admin.html
data/       本地数据：app.db、uploads/、docs/（samples 演示资料入库，user 私有资料不入库）
docs/       需求规格与架构说明
scripts/    探测脚本、初始化脚本、启动与测试脚本
tests/      Python 单测（pytest）
test/       Node 文档与仓库完整性校验
```

## 开发进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 需求与架构文档、仓库骨架、依赖清单、仓库校验测试 | ✅ 完成 |
| M1 | 数据层：schema、FTS5、sqlite-vec 装载与迁移 | ✅ 完成 |
| M2 | 入库管线 + 混合检索（BGE-M3 + RRF + 阈值） | ✅ 完成 |
| M3 | LLM 封装 + 多模态视觉 + 探测脚本 | ✅ 完成 |
| M4 | LangGraph 对话图 + 多轮记忆 + SSE 接口 | ✅ 完成 |
| M5 | 工单抽取、落库、状态流转、导出 | ⏳ 待开始 |
| M6 | 两个前端页面 + 端到端联调 | ⏳ 待开始 |
| M7 | 模拟资料入库 + 真实文档通道 + 启动说明 | ⏳ 待开始 |

## 开发与测试

```powershell
npm test                                                  # 文档与仓库完整性校验（零依赖）
& 'E:\PyCharmMiscProject\python.exe' -m pytest tests -q    # Python 单元测试
.\scripts\test.ps1                                        # 两套顺序跑（随 M1 提供）
```

## 约定

本仓库遵循 [AGENTS.md](AGENTS.md)：

- 每次改动完成，都要创建一个对应的 Git commit，便于后续追踪和回滚；
- 每次改动完成，都要编写或更新相关测试，并在交付前确保所有测试和验证全部通过。
