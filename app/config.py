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

    # --- 图片理解（视觉）通道 ---
    # 支持两种后端：**本地 Ollama**（离线，推荐）与**云端 OpenAI 兼容网关**。
    # VISION_* 是通用名；留空时回落到下面的 QWEN_*（向后兼容旧配置）。
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    vision_timeout: float = 0.0
    # 本地视觉模型不可用时，是否自动回退到云端网关（需已配置 QWEN_API_KEY）
    vision_auto_cloud_fallback: bool = True
    # 编码前的等比缩放（长边像素，0 表示不缩放）：本地小模型对超大图很慢
    vision_max_side: int = 1280

    # --- 旧名（云端 Qwen 多模态网关），保留以兼容已有 .env ---
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key: str = ""
    qwen_vision_model: str = "qwen3.5-omni-plus"
    qwen_vision_timeout: float = 60.0

    # --- 模型总开关 ---
    # auto：用真实模型（本地 Ollama → 备用通道 → 多模态网关）
    # mock：全部换成离线确定性假模型，用于"先跑通后端逻辑"、前端联调与演示
    llm_mode: str = "auto"

    # --- 思考型模型适配 ---
    # qwen3 / deepseek-r1 等推理模型会把 token 预算花在思考上（实测正文直接为空），
    # 打开后自动追加 `/no_think` 软开关；非思考模型不受影响。
    disable_thinking: bool = True

    # --- 联网检索：MCP 服务 ---
    # 定位：内部知识库高置信时**不触发**；内部不足时才联网补充，结果单独标注来源。
    mcp_enabled: bool = True
    mcp_base_url: str = ""
    mcp_api_key: str = ""
    mcp_timeout: float = 45.0
    mcp_max_results: int = 5
    mcp_tool_name: str = ""  # 留空则自动挑选名字里带 search 的工具
    # 内部重排 top1 低于该值才联网（内部优先，不足才联网）
    mcp_trigger_top1: float = 0.35
    # 问题内容词在候选里的覆盖率低于该值也联网。
    # 为什么需要第二个信号：实测重排模型只要看到型号就给高分（0.74~0.84），
    # 光看重排分则"资料里根本没这个主题"也判成够用，联网永远不会触发。
    mcp_trigger_coverage: float = 0.34
    # 联网结果必须包含问题实体，否则丢弃（防"同名错误码含义不同"污染答案）
    mcp_require_entity_match: bool = True

    # --- 重排（Rerank）---
    # 本地 CrossEncoder 权重路径；为空则退化为「不重排」（用融合分）
    rerank_model_path: str = ""
    rerank_device: str = "cpu"
    rerank_batch_size: int = 8
    rerank_max_length: int = 512
    # 是否在服务启动时后台预热重排模型（2.1GB，加载约 6.5s，占内存；默认懒加载）
    warmup_reranker: bool = False
    # 动态 TopK：绝对下限 + 相对断崖 + 条数上限
    rerank_min_score: float = 0.10
    rerank_relative_floor: float = 0.35
    rerank_cliff_ratio: float = 0.5
    rerank_top_k_max: int = 6
    # 进入重排的候选数量（每个查询变体各取这么多，重排后由动态 TopK 收敛）
    rerank_candidates: int = 24

    # --- 三档置信判定（高置信检索 / 中置信反问 / 低置信拒答）---
    confidence_high: float = 0.50
    confidence_low: float = 0.10
    clarify_max_rounds: int = 2
    # 边界区间是否再问一次 LLM 做实体对齐确认（默认关：省一次模型调用）
    llm_align_enabled: bool = False

    # --- 本地对话模型（默认通道）---
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen2.5:latest"
    ollama_timeout: float = 120.0

    # --- 对话行为 ---
    chat_max_tokens: int = 700
    history_turns: int = 6
    # 兜底回答是否交给模型润色。默认 false：兜底话术必须逐字可控，
    # 否则模型可能把「内部资料未收录」改写成模糊说法，甚至编造无关建议。
    fallback_use_model: bool = False

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
    warmup_on_start: bool = True

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

    # -------------------------------------------------------- 视觉通道派生配置

    @property
    def effective_vision_base_url(self) -> str:
        """视觉通道地址：VISION_BASE_URL 优先，否则回落到 QWEN_BASE_URL。"""
        return (self.vision_base_url or self.qwen_base_url or "").strip()

    @property
    def effective_vision_model(self) -> str:
        return (self.vision_model or self.qwen_vision_model or "").strip()

    @property
    def effective_vision_api_key(self) -> str:
        return (self.vision_api_key or self.qwen_api_key or "").strip()

    @property
    def effective_vision_timeout(self) -> float:
        return float(self.vision_timeout or self.qwen_vision_timeout or 60.0)

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
                "base_url": self.effective_vision_base_url,
                "model": self.effective_vision_model,
                "api_key": mask_secret(self.effective_vision_api_key),
                "local": is_local_url(self.effective_vision_base_url),
                "max_side": self.vision_max_side,
                "auto_cloud_fallback": self.vision_auto_cloud_fallback,
            },
            "rerank": {
                "model_path": self.rerank_model_path or "(未配置，退化为融合分)",
                "device": self.rerank_device,
                "min_score": self.rerank_min_score,
                "top_k_max": self.rerank_top_k_max,
            },
            "mcp_search": {
                "enabled": self.mcp_enabled,
                "base_url": self.mcp_base_url or "(未配置)",
                "api_key": mask_secret(self.mcp_api_key),
                "timeout": self.mcp_timeout,
            },
        }


LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1", "host.docker.internal"}


def is_local_url(url: str | None) -> bool:
    """判断地址是否指向本机（用于区分"本地 Ollama"与"云端网关"）。"""
    if not url:
        return False
    from urllib.parse import urlparse

    try:
        host = urlparse(url if "://" in url else f"http://{url}").hostname
    except ValueError:
        return False
    return (host or "").lower() in LOCAL_HOSTS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内缓存的配置实例。测试里需要独立配置时直接构造 `Settings(...)`。"""
    return Settings()
