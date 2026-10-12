"""配置模块测试：环境变量覆盖、路径解析、密钥脱敏。"""

from __future__ import annotations

from pathlib import Path

from app.config import BASE_DIR, Settings, mask_secret


def test_mask_secret_hides_all_but_tail():
    assert mask_secret("sk-abcd-PA28") == "********PA28"
    assert mask_secret("") == "(未配置)"
    assert mask_secret(None) == "(未配置)"
    assert mask_secret("   ") == "(未配置)"


def test_mask_secret_short_value_fully_masked():
    assert mask_secret("abc") == "***"


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "sk-test-1234")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("RETRIEVE_TOP_K", "7")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")

    settings = Settings(_env_file=None)

    assert settings.qwen_api_key == "sk-test-1234"
    assert settings.ollama_model == "qwen3:4b"
    assert settings.retrieve_top_k == 7
    assert settings.embedding_dim == 1024


def test_relative_paths_resolve_against_project_root():
    settings = Settings(_env_file=None, db_path=Path("data/app.db"))
    assert settings.db_file == BASE_DIR / "data" / "app.db"


def test_absolute_paths_are_kept():
    absolute = Path("D:/tmp/custom.db")
    settings = Settings(_env_file=None, db_path=absolute)
    assert settings.db_file == absolute


def test_derived_paths():
    settings = Settings(
        _env_file=None,
        docs_dir=Path("data/docs"),
        upload_dir=Path("data/uploads"),
    )
    assert settings.samples_path == BASE_DIR / "data" / "docs" / "samples"
    assert settings.user_docs_path == BASE_DIR / "data" / "docs" / "user"
    assert settings.upload_path == BASE_DIR / "data" / "uploads"


def test_ensure_dirs_creates_every_runtime_directory(tmp_path=None):
    # 不用 pytest 的 tmp_path（沙箱下 mkdtemp 目录无权限），改用 settings 固定的相对目录
    settings = Settings(
        _env_file=None,
        db_path=Path("data/test_tmp/dirs/app.db"),
        upload_dir=Path("data/test_tmp/dirs/uploads"),
        docs_dir=Path("data/test_tmp/dirs/docs"),
    )
    settings.ensure_dirs()
    try:
        assert settings.db_file.parent.is_dir()
        assert settings.upload_path.is_dir()
        assert settings.samples_path.is_dir()
        assert settings.user_docs_path.is_dir()
    finally:
        import shutil

        shutil.rmtree(BASE_DIR / "data" / "test_tmp" / "dirs", ignore_errors=True)


def test_describe_masks_keys(monkeypatch):
    # 注意：这里的假密钥必须短于仓库密钥扫描阈值（sk- 后 >= 16 位会被判为疑似真实密钥）
    monkeypatch.setenv("QWEN_API_KEY", "sk-fake-PA28")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-9876")
    settings = Settings(_env_file=None)

    described = settings.describe()

    assert described["vision"]["api_key"] == "********PA28"
    assert described["chat_fallback"]["api_key"] == "********9876"
    # 明文不得出现在摘要里
    assert "sk-fake-PA28" not in str(described)
    assert "sk-fake-9876" not in str(described)


def test_env_example_keys_all_exist_in_settings():
    """防漂移：`.env.example` 里的每个配置项都必须有对应的 Settings 字段。

    这条守卫是有来历的：项目里出现过两次「.env.example 写了、Settings 忘了定义」，
    结果配置静默失效（RERANK_CLIFF_RATIO 被忽略、MCP_TRIGGER_TOP1 直接报错）。
    """
    env_example = BASE_DIR / ".env.example"
    keys: list[str] = []
    for raw in env_example.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name and name.replace("_", "").isalnum():
            keys.append(name.lower())

    assert keys, ".env.example 里没有解析到任何配置项"

    fields = set(Settings.model_fields)
    missing = sorted({key for key in keys if key not in fields})
    assert missing == [], f"这些配置项在 Settings 里没有对应字段：{missing}"
