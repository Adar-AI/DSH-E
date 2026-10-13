"""开发启动入口（PyCharm / 命令行通用）。

等价于在项目根执行 `python -m app.main`，但**不依赖工作目录与 PYTHONPATH**：
先把项目根放进 sys.path，再调用 `app.main.main()`。

为什么需要这个文件：PyCharm 的「Module name」运行方式在不同版本里序列化字段不一致
（本机 PyCharm 2026.1 实测报 "Python module name must be set"），而脚本方式稳定可靠。
另外 `uvicorn.run("app.main:app")` 需要能 import 到 `app` 包，脚本方式下由本文件保证。

用法：

    python run_dev.py                 # 真实模型（读 .env）
    LLM_MODE=mock python run_dev.py   # 离线模式：不调用任何模型
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import main  # noqa: E402  （必须在 sys.path 调整之后导入）

if __name__ == "__main__":
    main()
