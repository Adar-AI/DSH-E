"""FastAPI 应用装配与启动入口。

启动流程：建目录 → 建表（幂等）→ 后台线程预热模型 → 就绪。
`create_app()` 是工厂函数，便于测试注入不同配置。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db as dbmod
from app.api import chat as chat_api
from app.api import conversations as conversations_api
from app.api import health as health_api
from app.config import Settings, get_settings
from app.deps import start_warmup

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app(
    settings: Settings | None = None, *, warmup_enabled: bool | None = None
) -> FastAPI:
    """构造应用。测试里传 `warmup_enabled=False` 可跳过模型预热。"""
    settings = settings or get_settings()
    should_warm = settings.warmup_on_start if warmup_enabled is None else warmup_enabled

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.ensure_dirs()
        conn = dbmod.connect(settings=settings)
        try:
            dbmod.init_db(conn, settings=settings)
        finally:
            conn.close()
        logger.info("数据库就绪：%s", settings.db_file)
        if should_warm:
            start_warmup(settings)
        yield

    app = FastAPI(
        title="多模态企业智能客服系统",
        description="企业内部资料驱动的售后客服：多模态理解 + 知识库检索 + 工单归档",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health_api.router)
    app.include_router(chat_api.router)
    app.include_router(conversations_api.router)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

    return app


app = create_app()


def main() -> None:
    """`python -m app.main` 启动开发服务。"""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=str(settings.log_level).lower(),
    )


if __name__ == "__main__":
    main()
