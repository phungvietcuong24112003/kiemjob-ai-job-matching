from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _post_match(payload: dict):
    res = client.post("/api/v1/match", json=payload)
    assert res.status_code == 200
    return res.json()


def test_01_cv_very_suitable_for_job():
    data = _post_match({
        "cv": {
            "cv_id": 1,
            "raw_text": (
                "Trần Minh B - Backend Developer\n"
                "Kinh nghiệm làm việc\n"
                "2 năm kinh nghiệm phát triển hệ thống Backend bằng Java, Spring Boot, REST API và PostgreSQL.\n"
                "Dự án tiêu biểu\n"
                "Triển khai hệ thống microservices với Docker, Git, PostgreSQL và Spring Boot.\n"
                "Học vấn\n"
                "Đại học Bách Khoa - Công nghệ thông tin\n"
            ),
        },
        "job": {
            "job_id": 10,
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "education_requirement": "Đại học trở lên",
            "description": "Xây dựng hệ thống Backend và REST API.",
            "requirements": "Yêu cầu Java, Spring Boot, PostgreSQL, REST API, kinh nghiệm từ 1-2 năm. Biết Docker là lợi thế.",
        },
    })
    assert data["overall_score"] >= 85
    assert data["skill_score"] == 100
    assert data["technology_score"] == 100
    assert len(data["evidence_from_cv"]) >= 4


def test_02_cv_partially_suitable():
    data = _post_match({
        "cv": {
            "raw_text": (
                "Backend Developer\n"
                "Kinh nghiệm làm việc\n"
                "1 năm kinh nghiệm phát triển REST API bằng Java và MySQL.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "requirements": "Thành thạo Java, Spring Boot, PostgreSQL, REST API. Kinh nghiệm 1-2 năm.",
        },
    })
    assert 40 <= data["overall_score"] < 80
    missing_names = [s["name"] for s in data["missing_skills"]]
    assert "spring boot" in missing_names
    assert "postgresql" in missing_names


def test_03_cv_not_suitable_different_domain():
    data = _post_match({
        "cv": {
            "raw_text": (
                "Nhân viên Kế toán tổng hợp\n"
                "Kinh nghiệm làm việc\n"
                "3 năm làm kế toán thuế, lập báo cáo tài chính trên phần mềm MISA và Excel.\n"
                "Học vấn\n"
                "Đại học Kinh tế - chuyên ngành Kế toán.\n"
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "requirements": "Thành thạo Java, Spring Boot, PostgreSQL, Docker. Kinh nghiệm 2 năm.",
        },
    })
    assert data["overall_score"] < 35
    assert data["skill_score"] == 0
    assert data["technology_score"] == 0


def test_04_cv_insufficient_information():
    data = _post_match({
        "cv": {"raw_text": "CV trống"},
        "job": {"title": "Backend Developer", "requirements": "Java, Spring Boot"},
    })
    assert data["data_sufficiency"] == "insufficient"
    assert data["overall_score"] == 0
    assert data["confidence"] == "low"


def test_05_cv_with_synonyms_and_variants():
    data = _post_match({
        "cv": {
            "raw_text": (
                "Frontend Engineer\n"
                "Kinh nghiệm làm việc\n"
                "1.5 năm xây dựng giao diện web bằng ReactJS, JS (ES6), Tailwind CSS và kết nối Postgre SQL.\n"
            )
        },
        "job": {
            "title": "Frontend Developer",
            "required_skills_from_db": ["React", "JavaScript", "TailwindCSS"],
            "requirements": "Thành thạo React, JavaScript, PostgreSQL. Kinh nghiệm 1 năm.",
        },
    })
    matched_names = [s["name"] for s in data["matched_skills"]]
    assert "react" in matched_names
    assert "javascript" in matched_names
    assert "tailwindcss" in matched_names
    assert "postgresql" in matched_names


def test_06_cv_vietnamese_language():
    data = _post_match({
        "cv": {
            "raw_text": (
                "Chuyên viên Phân tích nghiệp vụ\n"
                "Kinh nghiệm làm việc\n"
                "2 năm làm phân tích nghiệp vụ, viết tài liệu yêu cầu phần mềm, quản lý dự án theo mô hình Scrum và giao tiếp khách hàng.\n"
                "Học vấn\n"
                "Cử nhân Hệ thống thông tin.\n"
            )
        },
        "job": {
            "title": "Business Analyst",
            "requirements": "Yêu cầu kỹ năng Business Analysis, Scrum và Giao tiếp. Kinh nghiệm 1 năm.",
            "education_requirement": "Đại học trở lên",
        },
    })
    assert data["overall_score"] >= 70
    assert data["education_score"] == 100


def test_07_cv_english_language():
    data = _post_match({
        "cv": {
            "raw_text": (
                "John Doe - Python Backend Engineer\n"
                "Work Experience\n"
                "2 years of experience building scalable RESTful APIs using Python, FastAPI, PostgreSQL and Docker.\n"
                "Education\n"
                "Bachelor of Computer Science.\n"
            )
        },
        "job": {
            "title": "Python Developer",
            "required_skills_from_db": ["Python", "FastAPI", "PostgreSQL"],
            "requirements": "At least 1 year experience with Python, FastAPI, REST API. Docker is a plus.",
            "education_requirement": "Đại học trở lên",
        },
    })
    assert data["overall_score"] >= 85
    assert data["education_score"] == 100


def test_08_jd_distinguishes_required_vs_preferred():
    data = _post_match({
        "cv": {
            "raw_text": (
                "Backend Developer với 1 năm kinh nghiệm phát triển REST API bằng Java và Spring Boot."
            )
        },
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot"],
            "requirements": "Bắt buộc Java, Spring Boot. Ưu tiên biết thêm Docker và Redis.",
        },
    })
    req_names = [s["name"] for s in data["required_skills"]]
    pref_names = [s["name"] for s in data["preferred_skills"]]
    assert "java" in req_names
    assert "spring boot" in req_names
    assert "docker" in pref_names
    assert "redis" in pref_names
    assert data["skill_score"] == 100


def test_09_cv_unusually_long_keyword_stuffing():
    stuffed_cv = (
        "Backend Developer 1 năm kinh nghiệm Java Spring Boot. "
        + ("Dự án quản lý bán hàng dùng công nghệ mới. " * 250)
    )
    data = _post_match({
        "cv": {"raw_text": stuffed_cv},
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL", "Docker", "Kubernetes"],
            "requirements": "Yêu cầu Java, Spring Boot, PostgreSQL, Docker, Kubernetes, kinh nghiệm 3 năm.",
        },
    })
    # Không bị điểm cao giả tạo dù CV rất dài
    assert data["overall_score"] < 65


def test_10_cv_with_irrelevant_noise():
    noisy_cv = (
        "Sở thích: đá bóng, nghe nhạc, du lịch, nấu ăn, đọc sách trinh thám, sưu tầm tem.\n"
        "Hoạt động ngoại khóa: tình nguyện viên mùa hè xanh.\n"
        "Kinh nghiệm: 1 năm làm Backend Developer với Java, Spring Boot, PostgreSQL.\n"
    )
    data = _post_match({
        "cv": {"raw_text": noisy_cv},
        "job": {
            "title": "Backend Developer",
            "required_skills_from_db": ["Java", "Spring Boot", "PostgreSQL"],
            "requirements": "Kinh nghiệm 1 năm với Java, Spring Boot, PostgreSQL.",
        },
    })
    matched_names = [s["name"] for s in data["matched_skills"]]
    assert set(matched_names) == {"java", "spring boot", "postgresql"}
    assert data["overall_score"] >= 75
