"""真实网关探测脚本（不进入自动化测试）。

做三件事：
1. 打印当前配置（密钥一律脱敏）；
2. 探测对话通道（Ollama / 备用通道）是否连通；
3. 探测多模态通道：尝试列出模型、找出视觉模型候选名，并真的分析一张测试图，
   确认「图片 → 结构化描述」这条链路可用。

用法：
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_vision.py
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_vision.py --image 截图.png
    & 'E:\\PyCharmMiscProject\\python.exe' scripts\\probe_vision.py --list-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.providers import ProviderError, build_chat_chain, build_vision_provider  # noqa: E402
from app.llm.vision import analyze_image  # noqa: E402

VISION_HINTS = ("omni", "vl", "vision", "multimodal")


def make_test_image(path: Path) -> Path:
    """生成一张带报错码文字的测试图，用于验证多模态链路。"""
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (560, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([16, 16, 544, 244], outline="black", width=3)

    font = None
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        try:
            font = ImageFont.truetype(candidate, 30)
            break
        except OSError:
            continue

    draw.text((40, 50), "ERROR E05", fill="black", font=font)
    draw.text((40, 105), "Drum unit not detected", fill="black", font=font)
    draw.text((40, 160), "Model: L3250", fill="black", font=font)
    image.save(path)
    return path


def probe_chat(settings) -> None:
    print("=== 对话通道 ===")
    chain = build_chat_chain(settings)
    for provider in chain.providers:
        ok, detail = provider.ping()
        flag = "✅" if ok else "❌"
        print(f"{flag} {provider.name:9} 模型={provider.model:16} {detail}")


def probe_vision(settings, image_path: Path | None, list_only: bool, model_override: str | None) -> int:
    provider = build_vision_provider(settings)
    if model_override:
        provider.model = model_override

    print("\n=== 多模态通道 ===")
    print(f"base_url = {provider.base_url}")
    print(f"model    = {provider.model}")
    print(f"api_key  = {provider.api_key[:4]}****{provider.api_key[-4:]} (已脱敏)")

    candidates: list[str] = []
    try:
        names = provider.list_models()
    except ProviderError as exc:
        print(f"⚠️  无法列出模型（网关可能不支持该接口）：{exc}")
    else:
        print(f"✅ 网关返回 {len(names)} 个模型")
        candidates = sorted(
            name for name in names if any(hint in name.lower() for hint in VISION_HINTS)
        )
        if candidates:
            print("   视觉模型候选：")
            for name in candidates:
                print(f"     - {name}")
        else:
            print("   未发现名字里带 omni/vl/vision 的模型，前 20 个模型如下：")
            for name in names[:20]:
                print(f"     - {name}")

    if list_only:
        return 0

    print("\n=== 图片理解实测 ===")
    target = image_path or make_test_image(ROOT / "data" / "test_tmp" / "probe" / "error_e05.png")
    print(f"图片：{target}（{target.stat().st_size} 字节）")

    result = analyze_image(target, provider=provider, context="打印不出来东西，屏幕上有个报错")
    print(f"结果：ok={result.ok}")
    if result.ok:
        print("结构化字段：")
        print(json.dumps(result.detail, ensure_ascii=False, indent=2))
        print("给提示词用的描述：")
        print(result.summary)
    else:
        print(f"失败原因：{result.error}")

    if result.ok:
        return 0
    print("\n提示：如果失败原因是模型不存在，用 --model 换成上面列出的视觉模型候选再试。")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测对话与多模态通道")
    parser.add_argument("--image", default=None, help="指定要分析的图片（默认自动生成测试图）")
    parser.add_argument("--model", default=None, help="覆盖多模态模型名")
    parser.add_argument("--list-models", action="store_true", help="只列出网关模型")
    parser.add_argument("--skip-chat", action="store_true", help="跳过对话通道探测")
    args = parser.parse_args(argv)

    settings = get_settings()
    print("=== 当前配置（密钥脱敏）===")
    print(json.dumps(settings.describe(), ensure_ascii=False, indent=2))

    if not args.skip_chat:
        print()
        probe_chat(settings)

    image_path = Path(args.image) if args.image else None
    return probe_vision(settings, image_path, args.list_models, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
