import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.engine import CognitiveMemoryEngine
from app.api import routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing engine and background daemon."""
    logger.info("Initializing CognitiveMemoryEngine...")
    engine = CognitiveMemoryEngine()
    await engine.initialize()
    routes.engine_instance = engine
    
    logger.info("Starting background ConsolidationDaemon...")
    engine.start_daemon()

    yield

    logger.info("Stopping background ConsolidationDaemon...")
    await engine.stop_daemon()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Cognitive-Memory-Engine API Gateway",
    description="Local-first Cognitive Memory Engine REST API powered by Ollama and Qwen2.5",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Cognitive-Memory-Engine API Gateway",
        "model": settings.OLLAMA_MODEL,
        "docs_url": "/docs",
    }
