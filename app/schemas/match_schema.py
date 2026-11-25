from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ParsedCvProfile(BaseModel):
    skills: List[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    education_level: Optional[str] = None
    job_titles: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


class CvInput(BaseModel):
    cv_id: Optional[int] = None
    raw_text: str = Field(..., min_length=1, description="Văn bản CV trích xuất từ PDF hoặc DB")
    parsed_profile: Optional[ParsedCvProfile] = None


class JobInput(BaseModel):
    job_id: Optional[int] = None
    title: str = Field(..., min_length=1, description="Tiêu đề vị trí tuyển dụng")
    category: Optional[str] = None
    specializations: List[str] = Field(default_factory=list)
    required_skills_from_db: List[str] = Field(default_factory=list)
    education_requirement: Optional[str] = None
    working_mode: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = ""
    requirements: Optional[str] = ""
    benefits: Optional[str] = ""


class MatchRequest(BaseModel):
    cv: CvInput
    job: JobInput


EvidenceStatus = Literal["evidenced", "unclear_evidence", "not_found", "insufficient_data"]
EvidenceType = Optional[Literal["direct_quote", "fuzzy_match"]]
SkillCategory = Literal["core_skills", "tools_and_tech", "supplementary_skills"]


class SkillMatchItem(BaseModel):
    name: str
    category: SkillCategory
    is_required: bool = True
    evidence_status: EvidenceStatus
    evidence_type: EvidenceType = None
    evidence: Optional[str] = None
    verified: bool = True
    comment: str = ""


class EvidenceItem(BaseModel):
    skill: str
    quote: str
    evidence_type: Literal["direct_quote", "fuzzy_match"]
    verified: bool = True


class ExperienceEvaluation(BaseModel):
    candidate_years: Optional[float] = None
    required_min_years: Optional[float] = None
    tfidf_similarity: float = 0.0
    project_tech_ratio: float = 0.0
    assessment: str = ""


class EducationEvaluation(BaseModel):
    required_education: str = "Không yêu cầu"
    candidate_education: Optional[str] = None
    meets_requirement: bool = True
    assessment: str = ""


class ConfidenceDetail(BaseModel):
    level: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    reason: str = ""


class StructuredStrengthItem(BaseModel):
    title: str
    description: str
    evidence: Optional[str] = None


class MissingRequirementItem(BaseModel):
    title: str
    description: str
    jd_requirement: Optional[str] = None
    status: Literal["not_found", "unclear_evidence", "insufficient_data"] = "not_found"


class RecommendationDetail(BaseModel):
    level: Literal["CONSIDER_INTERVIEW", "NEED_MORE_REVIEW", "INSUFFICIENT_DATA"] = "NEED_MORE_REVIEW"
    label: str = ""
    description: str = ""


class MatchResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    skill_score: Optional[int] = Field(default=None, ge=0, le=100)
    experience_score: int = Field(..., ge=0, le=100)
    job_relevance_score: int = Field(..., ge=0, le=100)
    technology_score: Optional[int] = Field(default=None, ge=0, le=100)
    education_score: Optional[int] = Field(default=None, ge=0, le=100)
    supplementary_skill_score: Optional[int] = Field(default=None, ge=0, le=100)

    data_sufficiency: Literal["sufficient", "insufficient"] = "sufficient"
    confidence: Literal["high", "medium", "low"] = "high"

    matched_skills: List[SkillMatchItem] = Field(default_factory=list)
    missing_skills: List[SkillMatchItem] = Field(default_factory=list)
    required_skills: List[SkillMatchItem] = Field(default_factory=list)
    preferred_skills: List[SkillMatchItem] = Field(default_factory=list)

    experience_evaluation: ExperienceEvaluation
    education_evaluation: EducationEvaluation

    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    structured_strengths: List[StructuredStrengthItem] = Field(default_factory=list)
    missing_requirements: List[MissingRequirementItem] = Field(default_factory=list)
    recommendation: Optional[RecommendationDetail] = None
    confidence_detail: Optional[ConfidenceDetail] = None
    evidence_from_cv: List[EvidenceItem] = Field(default_factory=list)

    analysis_summary: str = ""
    algorithm_version: str = "v1.1-nlp-explain"
