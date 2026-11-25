from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _match(payload: dict):
    res = client.post("/api/v1/match", json=payload)
    assert res.status_code == 200
    return res.json()


def test_case_1_high_match_has_structured_strengths_and_consider_interview():
    data = _match({
        "cv": {
            "raw_text": (
                "Nguyễn Văn An - Backend Java Developer\n"
                "Mục tiêu nghề nghiệp\n"
                "Phát triển hệ thống Backend hiệu năng cao với Java Spring Boot.\n"
                "Kinh nghiệm làm việc\n"
                "2 năm kinh nghiệm làm Backend Developer tại Công ty ABC. "
                "Trực tiếp phát triển các REST API bằng Java, Spring Boot và tối ưu truy vấn trên PostgreSQL.\n"
                "Dự án tiêu biểu\n"
                "Hệ thống tuyển dụng: xây dựng microservices bằng Java, Spring Boot, PostgreSQL, Git và Docker.\n"
                "Học vấn\n"
                "Đại học Bách Khoa Hà Nội - Công nghệ thông tin.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "education_requirement": "Đại học trở lên",
            "requirements": "Yêu cầu Java, Spring Boot, PostgreSQL, REST API. Kinh nghiệm từ 1-2 năm. Biết Docker là lợi thế.",
        },
    })
    assert data["overall_score"] >= 80
    assert data["confidence_detail"]["level"] == "HIGH"
    assert data["recommendation"]["level"] == "CONSIDER_INTERVIEW"
    assert len(data["structured_strengths"]) >= 3
    assert any("java" in s["title"].lower() and s["evidence"] for s in data["structured_strengths"])
    assert len(data["missing_requirements"]) == 0


def test_case_2_medium_match_separates_strengths_and_missing_without_assuming_inability():
    data = _match({
        "cv": {
            "raw_text": (
                "Lê Thị Bình - Java Developer\n"
                "Kinh nghiệm làm việc\n"
                "1.5 năm kinh nghiệm phát triển ứng dụng web bằng Java và Spring Boot tại Công ty XYZ.\n"
                "Dự án tiêu biểu\n"
                "Website quản lý đơn hàng sử dụng Java, Spring Boot và MySQL.\n"
                "Học vấn\n"
                "Đại học Công nghệ - Công nghệ thông tin.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL", "Docker"],
            "requirements": "Có kinh nghiệm sử dụng Java, Spring Boot, PostgreSQL và Docker. Kinh nghiệm từ 1 năm.",
        },
    })
    assert 45 <= data["overall_score"] < 85
    assert len(data["structured_strengths"]) >= 2
    missing_titles = [m["title"].lower() for m in data["missing_requirements"]]
    assert any("postgresql" in t for t in missing_titles)
    assert any("docker" in t for t in missing_titles)
    for m in data["missing_requirements"]:
        # Phải diễn đạt "Chưa tìm thấy minh chứng...", không quy chụp "không biết"
        assert "chưa tìm thấy minh chứng" in m["description"].lower() or "chưa" in m["description"].lower()


def test_case_3_sparse_cv_low_confidence_even_if_keywords_match():
    data = _match({
        "cv": {
            "raw_text": "Kỹ năng: Java, Spring Boot, PostgreSQL, REST API."
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot"],
            "requirements": "Yêu cầu Java, Spring Boot, PostgreSQL, kinh nghiệm 2 năm.",
        },
    })
    assert data["confidence_detail"]["level"] == "LOW"
    assert data["recommendation"]["level"] == "INSUFFICIENT_DATA"


def test_case_4_unsuitable_cv_explains_missing_reasons():
    data = _match({
        "cv": {
            "raw_text": (
                "Phạm Văn Cường - Chuyên viên Kế toán tổng hợp\n"
                "Kinh nghiệm làm việc\n"
                "3 năm làm kế toán thuế, lập báo cáo tài chính trên phần mềm MISA và Excel tại Công ty Thương mại.\n"
                "Học vấn\n"
                "Đại học Kinh tế Quốc dân - chuyên ngành Kế toán.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "requirements": "Thành thạo Java, Spring Boot, PostgreSQL, REST API. Kinh nghiệm 2 năm.",
        },
    })
    assert data["overall_score"] < 40
    assert data["recommendation"]["level"] == "NEED_MORE_REVIEW"
    assert len(data["missing_requirements"]) >= 3


def test_case_5_synonyms_in_cv_and_jd():
    data = _match({
        "cv": {
            "raw_text": (
                "Trần Văn Dũng - Frontend Developer\n"
                "Kinh nghiệm làm việc\n"
                "2 năm kinh nghiệm phát triển giao diện web với ReactJS, JS (ES6) và sử dụng Postgre SQL.\n"
            )
        },
        "job": {
            "title": "Frontend Developer",
            "required_skills_from_db": ["React", "JavaScript", "PostgreSQL"],
            "requirements": "Thành thạo React, JavaScript, PostgreSQL. Kinh nghiệm 1 năm.",
        },
    })
    matched_names = [s["name"] for s in data["matched_skills"]]
    assert "react" in matched_names
    assert "javascript" in matched_names
    assert "postgresql" in matched_names
    assert data["overall_score"] >= 80


def test_case_6_unclear_keyword_only_without_usage_evidence():
    data = _match({
        "cv": {
            "raw_text": (
                "Hồ sơ ứng viên: Nguyễn Minh E\n"
                "Giới thiệu chung: Tốt nghiệp Đại học Công nghệ thông tin năm 2025.\n"
                "Danh sách công nghệ đã biết: Java, Spring Boot, Docker, PostgreSQL.\n"
                "Hoạt động: Tham gia câu lạc bộ thể thao của trường.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "requirements": "Yêu cầu Java, Spring Boot, PostgreSQL. Kinh nghiệm 1 năm.",
        },
    })
    # Có nhắc đến nhưng không có ngữ cảnh sử dụng/dự án -> phải là unclear_evidence, KHÔNG được coi là evidenced đầy đủ
    by_name = {s["name"]: s for s in data["matched_skills"]}
    assert by_name["java"]["evidence_status"] == "unclear_evidence"
    assert by_name["spring boot"]["evidence_status"] == "unclear_evidence"
    assert by_name["postgresql"]["evidence_status"] == "unclear_evidence"
    assert data["confidence_detail"]["level"] in ("LOW", "MEDIUM")
    assert any(m["status"] == "unclear_evidence" for m in data["missing_requirements"])
