import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routes.health_routes import router as health_router
from app.routes.match_routes import router as match_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai-cv-job-matching")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI CV-Job Matching Service khởi động (%s).", settings.ALGORITHM_VERSION)
    yield
    logger.info("AI CV-Job Matching Service đang tắt.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Dịch vụ NLP phân tích và chấm điểm mức độ phù hợp CV - Job cho KiemJob",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(match_router)
