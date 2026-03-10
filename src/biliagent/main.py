"""BiliAgent — FastAPI 应用入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from biliagent.api.routes import router
from biliagent.storage.database import init_db

# 日志配置（输出英文）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("biliagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表"""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")
    logger.info("BiliAgent is ready.")
    yield
    logger.info("BiliAgent shutting down.")


app = FastAPI(
    title="BiliAgent",
    description="Multi-agent Bilibili video summarization bot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "biliagent"}
