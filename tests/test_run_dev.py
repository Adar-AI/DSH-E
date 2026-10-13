"""开发启动入口测试：保证脚本方式启动时能正确导入应用。

背景：PyCharm 的「Module name」运行方式在不同版本里字段序列化不一致
（本机 2026.1 实测报 "Python module name must be set"），因此改用脚本入口 `run_dev.py`。
这个文件必须与「项目根执行 python -m app.main」等价，且不依赖工作目录与 PYTHONPATH。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "run_dev.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_dev_under_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_exists_at_project_root():
    assert LAUNCHER.exists(), "缺少 PyCharm/命令行启动入口 run_dev.py"


def test_launcher_adds_project_root_to_sys_path(monkeypatch):
    """故意把项目根从 sys.path 里拿掉，启动入口必须自己加回来。"""
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != str(ROOT)])

    module = _load_launcher()

    assert str(ROOT) in sys.path
    assert callable(module.main)


def test_launcher_exposes_app_main():
    module = _load_launcher()
    from app.main import main as app_main

    assert module.main is app_main, "启动入口应当直接调用 app.main.main"


def test_launcher_documents_modes():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "LLM_MODE=mock" in text, "启动入口应写明离线模式用法"
    assert "python -m app.main" in text, "启动入口应说明与模块方式的等价关系"
