from app.services.text_processor import TextProcessor


def test_strip_html_preserves_line_breaks():
    html = "<p>Phát triển hệ thống <strong>Backend</strong></p><ul><li>Java Spring Boot</li><li>PostgreSQL</li></ul>"
    cleaned = TextProcessor.clean_text(html)
    assert "Phát triển hệ thống Backend" in cleaned
    assert "Java Spring Boot" in cleaned
    assert "<" not in cleaned


def test_split_cv_sections_vietnamese_and_english():
    cv = """
    Nguyễn Văn A - Backend Developer
    Mục tiêu nghề nghiệp
    Mong muốn trở thành Senior Java Developer.
    Kỹ năng chuyên môn
    Java, Spring Boot, PostgreSQL, Docker
    Kinh nghiệm làm việc
    1 năm làm Backend Developer tại Công ty ABC.
    Dự án tiêu biểu
    Hệ thống tuyển dụng KiemJob dùng REST API.
    Học vấn
    Đại học Thủy Lợi - Công nghệ thông tin
    """
    sections = TextProcessor.split_cv_sections(cv)
    assert "Senior Java Developer" in sections["objective"]
    assert "Spring Boot" in sections["skills"]
    assert "Công ty ABC" in sections["experience"]
    assert "KiemJob" in sections["projects"]
    assert "Đại học Thủy Lợi" in sections["education"]


def test_preprocess_for_tfidf_keeps_tech_tokens_and_removes_stopwords():
    text = "Tôi có kinh nghiệm làm việc với C++, C#, Node.js và hệ thống PostgreSQL."
    processed = TextProcessor.preprocess_for_tfidf(text)
    assert "c++" in processed
    assert "c#" in processed
    assert "node.js" in processed
    assert "postgresql" in processed
    # Stopwords bị loại
    tokens = processed.split()
    assert "và" not in tokens
    assert "với" not in tokens
