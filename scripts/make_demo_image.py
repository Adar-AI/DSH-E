"""生成演示用的"故障截图"素材（前端页面与演示脚本都会用到）。

用法：
    & 'C:\\Users\\19181\\PyCharmMiscProject\\.venv\\Scripts\\python.exe' scripts\\make_demo_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = ROOT / "app" / "web" / "img" / "error-e05-sample.png"


def pick_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return None


def main() -> int:
    from PIL import Image, ImageDraw

    width, height = 720, 420
    image = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(image)

    # 模拟打印机屏幕边框
    draw.rounded_rectangle([40, 30, width - 40, height - 30], radius=18, fill="#111827")

    title_font = pick_font(26)
    body_font = pick_font(22)
    code_font = pick_font(34)

    draw.text((80, 60), "云打 L3250", fill="#9ca3af", font=body_font)
    draw.line([(80, 100), (width - 80, 100)], fill="#374151", width=2)

    # 报错图标
    draw.ellipse([80, 140, 150, 210], fill="#b91c1c")
    draw.text((104, 155), "!", fill="#ffffff", font=code_font)

    draw.text((175, 145), "ERROR E05", fill="#f87171", font=code_font)
    draw.text((175, 190), "Drum unit not detected", fill="#e5e7eb", font=body_font)

    draw.text((80, 250), "硒鼓未识别，请重新安装硒鼓", fill="#e5e7eb", font=body_font)
    draw.text((80, 290), "或更换新的硒鼓单元", fill="#e5e7eb", font=body_font)

    draw.rounded_rectangle([80, 340, 250, 382], radius=8, fill="#2563eb")
    draw.text((120, 350), "确定", fill="#ffffff", font=body_font)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    image.save(TARGET)
    print(f"已生成：{TARGET}（{TARGET.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
