from fastapi.testclient import TestClient
from app.main import app
from app.services.skill_extractor import SkillExtractor


extractor = SkillExtractor()


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert data["service"] == "ai-cv-job-matching"


def test_skill_normalization_synonyms_and_fuzzy():
    assert extractor.normalize_skill("JS") == "javascript"
    assert extractor.normalize_skill("ReactJS") == "react"
    assert extractor.normalize_skill("Postgre SQL") == "postgresql"
    assert extractor.normalize_skill("Springboot") == "spring boot"
    assert extractor.normalize_skill("K8s") == "kubernetes"
    # Fuzzy match (typo nhẹ với từ >= 5 ký tự)
    assert extractor.normalize_skill("Postgresqls") == "postgresql"


def test_mutual_exclusivity_no_duplicate_skill_across_categories():
    # PostgreSQL chỉ thuộc tools_and_tech, Java chỉ thuộc core_skills, English chỉ thuộc supplementary_skills
    assert extractor.get_category("postgresql") == "tools_and_tech"
    assert extractor.get_category("java") == "core_skills"
    assert extractor.get_category("english") == "supplementary_skills"

    text = "Sử dụng Java, Spring Boot, PostgreSQL, Postgres, Docker và Tiếng Anh giao tiếp."
    skills = extractor.extract_skills_from_text(text)
    names = [s.name for s in skills]
    assert len(names) == len(set(names))
    assert "postgresql" in names
    assert names.count("postgresql") == 1


def test_short_ambiguous_skills_require_context():
    # Câu tiếng Anh bình thường có "go" và "c" không phải kỹ năng lập trình
    plain_text = "I want to go to the office and take plan C for marketing."
    skills_plain = [s.name for s in extractor.extract_skills_from_text(plain_text)]
    assert "go" not in skills_plain
    assert "c" not in skills_plain

    # Câu có ngữ cảnh kỹ thuật rõ ràng
    tech_text = "Phát triển microservices backend bằng ngôn ngữ Go (Golang) và lập trình C/C++."
    skills_tech = [s.name for s in extractor.extract_skills_from_text(tech_text)]
    assert "go" in skills_tech
    assert "c" in skills_tech
    assert "c++" in skills_tech


def test_extract_jd_skills_required_vs_preferred():
    desc = "Tham gia phát triển hệ thống backend quy mô lớn."
    reqs = (
        "Thành thạo Java, Spring Boot, PostgreSQL và có kinh nghiệm RESTful API. "
        "Biết Docker hoặc Kubernetes là một lợi thế."
    )
    jd_skills = extractor.extract_jd_skills(
        description=desc,
        requirements=reqs,
        required_skills_from_db=["Java", "Spring Boot"],
    )
    by_name = {s.name: s for s in jd_skills}

    assert by_name["java"].is_required is True
    assert by_name["spring boot"].is_required is True
    assert by_name["postgresql"].is_required is True
    assert by_name["rest api"].is_required is True
    assert by_name["docker"].is_required is False
    assert by_name["kubernetes"].is_required is False

    # Kiểm tra phân nhóm không trùng lặp
    assert by_name["java"].category == "core_skills"
    assert by_name["postgresql"].category == "tools_and_tech"
    assert by_name["docker"].category == "tools_and_tech"
