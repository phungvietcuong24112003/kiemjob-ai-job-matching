from fastapi import APIRouter, Depends
from app.core.config import settings
from app.core.security import verify_internal_secret
from app.schemas.match_schema import MatchRequest, MatchResponse
from app.services.scoring_engine import ScoringEngine

router = APIRouter(prefix=settings.API_V1_STR, tags=["CV-Job Matching"])

_scoring_engine = ScoringEngine()


@router.post("/match", response_model=MatchResponse, dependencies=[Depends(verify_internal_secret)])
def match_cv_and_job(request: MatchRequest) -> MatchResponse:
    """
    Phân tích và chấm điểm mức độ phù hợp giữa 1 CV và 1 Job Description
    bằng thuật toán NLP truyền thống (Taxonomy + RapidFuzz + TF-IDF Cosine + Evidence Verification).
    """
    return _scoring_engine.evaluate(request)
