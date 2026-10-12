"""图片理解测试：编码、请求组装、JSON 解析、各类失败降级。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import vision as vision_module
from app.llm.providers import OpenAICompatProvider
from app.llm.vision import (
    analyze_image,
    build_vision_messages,
    describe_detail,
    detect_image_mime,
    encode_image_data_url,
    normalize_error_codes,
    prepare_image,
)
from fakes import (
    BrokenVisionProvider,
    FakeClient,
    FakeCompletions,
    StubVisionProvider,
    client_factory_for,
    write_png,
)

GOOD_JSON = (
    "```json\n"
    '{"kind": "错误截图", "product_model": "L3250", "error_codes": ["E05"],'
    ' "screen_location": "屏幕中央", "visible_damage": "",'
    ' "other_text": "请更换硒鼓", "summary": "打印机提示 E05，硒鼓未识别", "confidence": 0.86}\n'
    "```"
)


def make_vision_provider(replies=None, *, api_key="sk-abcd-1234", error=None, **overrides):
    completions = FakeCompletions(replies or [GOOD_JSON], error=error)
    factory = client_factory_for(FakeClient(completions))
    provider = OpenAICompatProvider(
        name="vision",
        base_url="https://example.com/v1",
        model="qwen3.5-omni-plus",
        api_key=api_key,
        requires_key=True,
        client_factory=factory,
        **overrides,
    )
    return provider, completions


# ------------------------------------------------------------------ 编码

def test_detect_image_mime(work_dir):
    assert detect_image_mime("a.png") == "image/png"
    assert detect_image_mime("a.JPEG") == "image/jpeg"

    with pytest.raises(ValueError):
        detect_image_mime("a.txt")


def test_encode_image_data_url(work_dir):
    path = write_png(work_dir, "shot.png")

    data_url = encode_image_data_url(path)

    assert data_url.startswith("data:image/png;base64,")
    assert len(data_url) > len("data:image/png;base64,")


def test_build_vision_messages_structure():
    messages = build_vision_messages("屏幕显示 E05", "data:image/png;base64,AAA")

    assert messages[0]["role"] == "system"
    assert "不要猜测" in messages[0]["content"]

    user = messages[1]
    assert user["role"] == "user"
    parts = user["content"]
    assert parts[0]["type"] == "text"
    assert "屏幕显示 E05" in parts[0]["text"]
    # JSON 花括号必须原样保留（不能被当模板占位符吃掉）
    assert '"error_codes"' in parts[0]["text"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAA"


# ------------------------------------------------------------------ 成功路径

def test_analyze_image_parses_structured_json(work_dir):
    path = write_png(work_dir, "error.png")
    provider, completions = make_vision_provider()

    result = analyze_image(path, provider=provider, context="打印不出来东西")

    assert result.ok is True
    assert result.detail["error_codes"] == ["E05"]
    assert "E05" in result.summary
    assert "产品型号：L3250" in result.summary
    assert "识别置信度：0.86" in result.summary
    assert result.model == "qwen3.5-omni-plus"

    sent = completions.calls[0]["messages"]
    assert sent[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_analyze_image_falls_back_to_raw_text(work_dir):
    path = write_png(work_dir, "blur.png")
    provider, _ = make_vision_provider(replies=["图片太模糊，只能看出是一台打印机。"])

    result = analyze_image(path, provider=provider)

    assert result.ok is True
    assert result.detail == {}
    assert "模糊" in result.summary


# ------------------------------------------------------------------ 失败路径

def test_analyze_image_missing_file(work_dir):
    provider, _ = make_vision_provider()

    result = analyze_image(work_dir / "没有这张图.png", provider=provider)

    assert result.ok is False
    assert "不存在" in result.error


def test_analyze_image_rejects_unsupported_suffix(work_dir):
    path = work_dir / "图.txt"
    path.write_text("x", encoding="utf-8")
    provider, _ = make_vision_provider()

    result = analyze_image(path, provider=provider)

    assert result.ok is False
    assert "不支持的图片类型" in result.error


def test_analyze_image_rejects_oversized_file(work_dir, monkeypatch):
    path = write_png(work_dir, "big.png")
    monkeypatch.setattr(vision_module, "MAX_IMAGE_BYTES", 10)
    provider, _ = make_vision_provider()

    result = analyze_image(path, provider=provider)

    assert result.ok is False
    assert "图片过大" in result.error


def test_analyze_image_reports_unavailable_provider(work_dir):
    path = write_png(work_dir, "shot.png")
    provider, _ = make_vision_provider(api_key="")

    result = analyze_image(path, provider=provider)

    assert result.ok is False
    assert "不可用" in result.error
    assert "API Key" in result.error


def test_analyze_image_reports_call_failure(work_dir):
    path = write_png(work_dir, "shot.png")
    provider, _ = make_vision_provider(error=RuntimeError("网关 429 限流"))

    result = analyze_image(path, provider=provider)

    assert result.ok is False
    assert "429" in result.error


def test_analyze_image_reports_empty_answer(work_dir):
    path = write_png(work_dir, "shot.png")
    provider, _ = make_vision_provider(replies=["   "])

    result = analyze_image(path, provider=provider)

    assert result.ok is False
    assert "没有返回可用的图片描述" in result.error


def test_analyze_image_without_provider_uses_settings(work_dir):
    path = write_png(work_dir, "shot.png")
    settings = Settings(_env_file=None, qwen_api_key="")

    result = analyze_image(path, settings=settings)

    assert result.ok is False
    assert "API Key" in result.error


# ------------------------------------------------------------------ 描述拼装

def test_describe_detail_joins_known_fields_and_skips_empty():
    summary = describe_detail(
        {
            "kind": "实物照片",
            "product_model": "",
            "error_codes": ["E05", "E06"],
            "visible_damage": "外壳破损",
            "summary": "硒鼓仓有破损",
            "confidence": 0.5,
            "unknown_field": "忽略",
        }
    )

    assert "图片类型：实物照片" in summary
    assert "产品型号" not in summary
    assert "错误码或提示：E05、E06" in summary
    assert "可见异常：外壳破损" in summary
    assert "unknown_field" not in summary


def test_describe_detail_empty():
    assert describe_detail({}) == ""


# -------------------------------------------------------------- 图片预处理

def test_prepare_image_scales_large_image_to_max_side(work_dir):
    from PIL import Image

    path = work_dir / "huge.png"
    Image.new("RGB", (3000, 1500), "white").save(path)

    data_url, info = prepare_image(path, max_side=1280)

    assert info["scaled"] is True
    assert info["original_size"] == [3000, 1500]
    assert max(info["sent_size"]) == 1280
    assert info["sent_size"] == [1280, 640]
    assert info["sent_bytes"] < info["original_bytes"]
    assert data_url.startswith("data:image/jpeg;base64,")


def test_prepare_image_keeps_small_image_untouched(work_dir):
    from PIL import Image

    path = work_dir / "small.png"
    Image.new("RGB", (640, 480), "white").save(path)

    data_url, info = prepare_image(path, max_side=1280)

    assert info["scaled"] is False
    assert info["sent_size"] == [640, 480]
    # 不需要缩放时原样发送原始字节（不重编码）
    assert info["sent_bytes"] == info["original_bytes"]
    assert data_url.startswith("data:image/png;base64,")


def test_prepare_image_keeps_transparency_as_png(work_dir):
    from PIL import Image

    path = work_dir / "logo.png"
    Image.new("RGBA", (2000, 2000), (255, 0, 0, 128)).save(path)

    data_url, info = prepare_image(path, max_side=1000)

    assert info["scaled"] is True
    assert data_url.startswith("data:image/png;base64,")


def test_prepare_image_without_limit_does_not_scale(work_dir):
    from PIL import Image

    path = work_dir / "big.png"
    Image.new("RGB", (2400, 1200), "white").save(path)

    _, info = prepare_image(path, max_side=0)

    assert info["scaled"] is False
    assert info["sent_size"] == [2400, 1200]


# -------------------------------------------------------------- 通道回退链

def test_analyze_image_falls_back_to_second_provider(work_dir):
    path = write_png(work_dir, "shot.png")
    broken = BrokenVisionProvider(error="本地模型未拉取")
    fallback = StubVisionProvider(name="vision-cloud-fallback", model="qwen3.5-omni-plus")

    result = analyze_image(path, providers=[broken, fallback])

    assert result.ok is True
    assert result.provider == "vision-cloud-fallback"
    assert result.model == "qwen3.5-omni-plus"
    assert len(broken.calls) == 1


def test_analyze_image_reports_all_channel_errors(work_dir):
    path = write_png(work_dir, "shot.png")
    broken = BrokenVisionProvider(error="连接被拒绝")
    unavailable = StubVisionProvider(name="vision-cloud", available=False, reason="未配置 API Key")

    result = analyze_image(path, providers=[broken, unavailable])

    assert result.ok is False
    assert "连接被拒绝" in result.error
    assert "未配置 API Key" in result.error


def test_analyze_image_marks_local_provider(work_dir):
    path = write_png(work_dir, "shot.png")
    local = StubVisionProvider(name="vision-local", base_url="http://127.0.0.1:11434/v1")

    result = analyze_image(path, providers=[local])

    assert result.ok is True
    assert result.local is True
    assert result.image_info["scaled"] is False


def test_analyze_image_marks_cloud_provider(work_dir):
    path = write_png(work_dir, "shot.png")
    cloud = StubVisionProvider(name="vision-cloud", base_url="https://example.com/v1")

    result = analyze_image(path, providers=[cloud])

    assert result.local is False


def test_analyze_image_uses_settings_chain(work_dir):
    path = write_png(work_dir, "shot.png")
    settings = Settings(
        _env_file=None,
        vision_base_url="http://127.0.0.1:11434/v1",
        vision_model="qwen2.5vl:3b",
        qwen_api_key="",
    )
    local = StubVisionProvider(name="vision-local")

    result = analyze_image(path, providers=[local], settings=settings)

    assert result.ok is True
    assert result.image_info["max_side"] == settings.vision_max_side


# -------------------------------------------------------------- 错误码规范化

def test_normalize_error_codes_extracts_code_from_verbose_text():
    assert normalize_error_codes(["ERROR E05"]) == ["E05"]
    assert normalize_error_codes("报错码：E05、E06") == ["E05", "E06"]
    assert normalize_error_codes(["E05", "e05", "E05"]) == ["E05"]


def test_normalize_error_codes_keeps_text_without_code():
    assert normalize_error_codes(["硒鼓未识别"]) == ["硒鼓未识别"]
    assert normalize_error_codes(None) == []
    assert normalize_error_codes([]) == []


def test_normalize_error_codes_ignores_product_model():
    # L3250 是型号不是错误码，不应出现在错误码里
    assert normalize_error_codes(["型号 L3250"]) == ["型号 L3250"]


def test_analyze_image_normalizes_codes_from_model(work_dir):
    path = write_png(work_dir, "verbose.png")
    verbose_reply = (
        '{"kind": "错误截图", "product_model": "L3250", "error_codes": ["ERROR E05"],'
        ' "summary": "提示 E05", "confidence": 0.8}'
    )
    provider, _ = make_vision_provider(replies=[verbose_reply])

    result = analyze_image(path, provider=provider)

    assert result.detail["error_codes"] == ["E05"]
    assert "错误码或提示：E05" in result.summary
