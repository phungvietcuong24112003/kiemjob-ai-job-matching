from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": settings.PROJECT_NAME,
        "version": settings.ALGORITHM_VERSION,
        "environment": settings.ENVIRONMENT,
    }
