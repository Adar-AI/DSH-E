"""配置读取与密钥脱敏。

设计要点：
1. 所有配置项都可以被环境变量覆盖（pydantic-settings 默认大小写不敏感），
   本地开发把值写在 `.env`（已被 .gitignore 忽略），仓库里只有 `.env.example`。
2. 相对路径统一按「项目根目录」解析，保证从任意工作目录启动都指向同一份数据。
3. 密钥绝不原样出现在日志或接口里，一律经过 `mask_secret()`。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：app/config.py -> app/ -> 项目根
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def mask_secret(value: str | None, keep: int = 4) -> str:
    """把密钥脱敏成 `********abcd` 形式，未配置时返回明确标记。"""
    if not value:
        return "(未配置)"
    value = value.strip()
    if not value:
        return "(未配置)"
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * 8 + value[-keep:]


class Settings(BaseSettings):
    """应用配置。字段名 -> 环境变量名：`qwen_api_key` -> `QWEN_API_KEY`。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- 多模态（图片理解）：OpenAI 兼容网关 ---
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key: str = ""
    qwen_vision_model: str = "qwen3.5-omni-plus"
    qwen_vision_timeout: float = 60.0

    # --- 本地对话模型（默认通道）---
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen3:4b"
    ollama_timeout: float = 120.0

    # --- 备用对话通道 ---
    fallback_llm_provider: str = "deepseek"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # --- 本地嵌入模型 ---
    embedding_model_path: str = (
        "D:/ai_models/modelscope_cachessss/models/models/BAAI--bge-m3/snapshots/master"
    )
    embedding_device: str = "cpu"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8

    # --- 检索参数 ---
    retrieve_top_k: int = 4
    retrieve_candidates: int = 12
    retrieve_min_score: float = 0.35
    chunk_size: int = 450
    chunk_overlap: int = 80

    # --- 存储 ---
    db_path: Path = Path("data/app.db")
    upload_dir: Path = Path("data/uploads")
    docs_dir: Path = Path("data/docs")

    # --- 服务 ---
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # ------------------------------------------------------------------ 派生路径

    @staticmethod
    def _resolve(path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (BASE_DIR / path)

    @property
    def db_file(self) -> Path:
        return self._resolve(self.db_path)

    @property
    def upload_path(self) -> Path:
        return self._resolve(self.upload_dir)

    @property
    def docs_path(self) -> Path:
        return self._resolve(self.docs_dir)

    @property
    def samples_path(self) -> Path:
        return self.docs_path / "samples"

    @property
    def user_docs_path(self) -> Path:
        return self.docs_path / "user"

    def ensure_dirs(self) -> None:
        """创建运行期需要的目录（幂等）。"""
        for path in (
            self.db_file.parent,
            self.upload_path,
            self.samples_path,
            self.user_docs_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 诊断信息

    def describe(self) -> dict[str, object]:
        """可直接输出到日志或 /api/health 的配置摘要（密钥已脱敏）。"""
        return {
            "db_file": str(self.db_file),
            "embedding": {
                "model_path": self.embedding_model_path,
                "device": self.embedding_device,
                "dim": self.embedding_dim,
            },
            "chat_primary": {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            },
            "chat_fallback": {
                "provider": self.fallback_llm_provider or "(禁用)",
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model,
                "api_key": mask_secret(self.deepseek_api_key),
            },
            "vision": {
                "base_url": self.qwen_base_url,
                "model": self.qwen_vision_model,
                "api_key": mask_secret(self.qwen_api_key),
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内缓存的配置实例。测试里需要独立配置时直接构造 `Settings(...)`。"""
    return Settings()
