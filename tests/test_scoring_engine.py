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


def test_explicit_required_skills_prioritized_and_returned():
    engine = ScoringEngine()
    cv_text = (
        "Lập trình viên Backend với 2 năm kinh nghiệm.\n"
        "Kinh nghiệm làm việc:\n"
        "- Phát triển REST API bằng Java và Spring Boot tại công ty ABC."
    )
    req = MatchRequest(
        cv=CvInput(cv_id=10, raw_text=cv_text),
        job=JobInput(
            job_id=20,
            title="Java Backend Developer",
            explicit_required_skills=["Java", "Spring Boot"],
            requirements="Thành thạo Microservices, GraphQL, gRPC, Scala.",
        ),
    )
    res = engine.evaluate(req)
    explicit_names = {item.name for item in res.explicit_required_skills}
    assert explicit_names == {"java", "spring boot"}
    assert all(item.source == "db" for item in res.explicit_required_skills)
    # 2/2 explicit skills evidenced (70% * 1.0) even when 4 extra JD skills are missing
    assert res.skill_score is not None and res.skill_score >= 70


def test_vietnamese_freetext_and_custom_category_and_preferred_section():
    extractor = SkillExtractor()
    assert extractor.get_category("Sử dụng thành thạo phần mềm MISA") == "tools_and_tech"
    assert extractor.get_category("Kỹ năng giao tiếp tốt") == "supplementary_skills"

    jd_skills = extractor.extract_jd_skills(
        requirements="Yêu cầu:\n- Thành thạo Java.\nƯu tiên:\n- Biết Docker.\n- Biết Kubernetes.",
        explicit_required_skills=["Lập báo cáo tài chính", "Kiểm soát công nợ khách hàng"],
    )
    by_name = {s.name: s for s in jd_skills}
    assert by_name["báo cáo tài chính"].source == "db"
    assert by_name["kiểm soát công nợ khách hàng"].source == "db"
    assert by_name["java"].is_required is True
    assert by_name["docker"].is_required is False
    assert by_name["kubernetes"].is_required is False

    engine = ScoringEngine()
    res = engine.evaluate(
        MatchRequest(
            cv=CvInput(
                cv_id=11,
                raw_text="Kinh nghiệm làm việc:\n- Phụ trách kiểm soát công nợ khách hàng và lập báo cáo tài chính quý.",
            ),
            job=JobInput(
                job_id=21,
                title="Kế toán tổng hợp",
                explicit_required_skills=["Kiểm soát công nợ khách hàng"],
            ),
        )
    )
    assert len(res.explicit_required_skills) == 1
    assert res.explicit_required_skills[0].evidence_status in ("evidenced", "unclear_evidence")
