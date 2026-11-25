from app.schemas.match_schema import CvInput, JobInput, MatchRequest
from app.services.evidence_verifier import EvidenceVerifier
from app.services.scoring_engine import ScoringEngine
from app.services.skill_extractor import SkillExtractor


def test_dynamic_weight_normalization_when_edu_and_supplementary_null():
    engine = ScoringEngine()
    req = MatchRequest(
        cv=CvInput(
            raw_text=(
                "Nguyễn Văn A - Backend Developer\n"
                "Kinh nghiệm làm việc\n"
                "1 năm kinh nghiệm phát triển RESTful API với Java Spring Boot và PostgreSQL tại Công ty ABC.\n"
                "Dự án tiêu biểu\n"
                "Xây dựng hệ thống tuyển dụng KiemJob dùng Java, Spring Boot, PostgreSQL, REST API.\n"
            )
        ),
        job=JobInput(
            title="Backend Developer",
            required_skills_from_db=["Java", "Spring Boot", "PostgreSQL"],
            education_requirement="Không yêu cầu",
            description="Phát triển hệ thống Backend hiệu năng cao.",
            requirements="Thành thạo Java, Spring Boot, PostgreSQL và REST API. Kinh nghiệm 1-2 năm. Biết Docker là lợi thế.",
        ),
    )
    res = engine.evaluate(req)
    assert res.education_score is None
    assert res.supplementary_skill_score is None
    assert res.skill_score == 100
    assert res.technology_score == 80  # PostgreSQL (required) = 1.0, Docker (preferred) = 0.0 -> 80
    assert res.overall_score >= 85
    assert res.data_sufficiency == "sufficient"


def test_evidence_verifier_never_fabricates_quotes():
    extractor = SkillExtractor()
    verifier = EvidenceVerifier(extractor)
    cv_text = "Ứng viên làm việc với Python và FastAPI trong 2 năm."
    jd_skills = extractor.extract_jd_skills(
        requirements="Yêu cầu Python, FastAPI, Docker.",
        required_skills_from_db=["Python", "Docker"],
    )
    items, evidences = verifier.verify_jd_skills_against_cv(jd_skills, cv_text)
    by_name = {i.name: i for i in items}

    assert by_name["python"].evidence_status == "evidenced"
    assert by_name["python"].evidence_type == "direct_quote"
    assert by_name["python"].evidence in cv_text

    assert by_name["docker"].evidence_status == "not_found"
    assert by_name["docker"].evidence is None
    assert len(evidences) == 2  # python + fastapi


def test_determinism_same_input_same_score():
    engine = ScoringEngine()
    req = MatchRequest(
        cv=CvInput(raw_text="Backend Developer với 2 năm kinh nghiệm Java, Spring Boot, MySQL, Git và Docker."),
        job=JobInput(
            title="Backend Developer",
            required_skills_from_db=["Java", "Spring Boot", "MySQL"],
            requirements="Kinh nghiệm từ 1 năm với Java, Spring Boot, MySQL, Docker.",
        ),
    )
    scores = [engine.evaluate(req).overall_score for _ in range(5)]
    assert len(set(scores)) == 1
