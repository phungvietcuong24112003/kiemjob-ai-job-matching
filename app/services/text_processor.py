import os
import re
import unicodedata
from typing import Dict, List, Set
from bs4 import BeautifulSoup


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_STOPWORDS_PATH = os.path.join(_DATA_DIR, "stopwords_vi_en.txt")

_CV_SECTION_HEADERS: Dict[str, List[str]] = {
    "objective": [
        "mục tiêu nghề nghiệp", "giới thiệu bản thân", "tóm tắt", "career objective",
        "objective", "summary", "about me", "profile"
    ],
    "skills": [
        "kỹ năng", "kỹ năng chuyên môn", "công nghệ", "skills", "technical skills",
        "core competencies", "technologies"
    ],
    "experience": [
        "kinh nghiệm làm việc", "kinh nghiệm", "quá trình làm việc", "lịch sử công tác",
        "work experience", "experience", "employment history", "professional experience"
    ],
    "projects": [
        "dự án", "dự án tham gia", "dự án tiêu biểu", "dự án cá nhân",
        "projects", "personal projects", "key projects"
    ],
    "education": [
        "học vấn", "trình độ học vấn", "giáo dục", "đào tạo",
        "education", "academic background"
    ],
    "certificates": [
        "chứng chỉ", "giải thưởng", "bằng cấp", "certifications", "certificates", "awards"
    ],
}


def load_stopwords(path: str = _STOPWORDS_PATH) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}


STOPWORDS: Set[str] = load_stopwords()


class TextProcessor:
    """Bộ tiền xử lý văn bản CV và Job Description cho pipeline NLP."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def strip_html(text: str) -> str:
        if not text:
            return ""
        if "<" not in text and ">" not in text:
            return text
        html_with_breaks = re.sub(r"(?i)<br\s*/?>", "\n", text)
        html_with_breaks = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", html_with_breaks)
        soup = BeautifulSoup(html_with_breaks, "html.parser")
        return soup.get_text(separator=" ")

    @classmethod
    def clean_text(cls, text: str, max_chars: int = 20000) -> str:
        if not text:
            return ""
        cleaned = cls.strip_html(text)
        cleaned = cls.normalize_unicode(cleaned)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
        cleaned = cleaned.strip()
        if max_chars > 0 and len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip()
        return cleaned

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Tách văn bản thành các câu/dòng gốc (giữ nguyên dấu và chữ hoa để trích xuất bằng chứng)."""
        if not text:
            return []
        cleaned = cls.clean_text(text, max_chars=0)
        lines = cleaned.split("\n")
        sentences: List[str] = []
        for line in lines:
            line = line.strip("•-* \t")
            if not line:
                continue
            parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9])", line)
            for part in parts:
                p = part.strip()
                if len(p) >= 3:
                    sentences.append(p)
        return sentences

    @classmethod
    def split_cv_sections(cls, text: str) -> Dict[str, str]:
        """Phân tách văn bản CV thành các mục chuẩn để đánh giá theo ngữ cảnh."""
        sections: Dict[str, List[str]] = {
            "header": [],
            "objective": [],
            "skills": [],
            "experience": [],
            "projects": [],
            "education": [],
            "certificates": [],
        }
        cleaned = cls.clean_text(text, max_chars=0)
        if not cleaned:
            return {k: "" for k in sections}

        current_section = "header"
        for line in cleaned.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            lower_clean = re.sub(r"[:\-_=•*]+", "", stripped.lower()).strip()

            matched_sec = None
            if len(lower_clean) <= 45:
                for sec_name, headers in _CV_SECTION_HEADERS.items():
                    if any(lower_clean == h or lower_clean.startswith(h + " ") for h in headers):
                        matched_sec = sec_name
                        break

            if matched_sec:
                current_section = matched_sec
            else:
                sections[current_section].append(stripped)

        return {k: "\n".join(v) for k, v in sections.items()}

    @classmethod
    def preprocess_for_tfidf(cls, text: str) -> str:
        """Chuẩn hóa văn bản cho TfidfVectorizer: giữ token kỹ thuật (c++, c#, .net, node.js)."""
        if not text:
            return ""
        cleaned = cls.clean_text(text, max_chars=0).lower()
        cleaned = re.sub(r"[^\w\s+#.]", " ", cleaned)
        tokens = cleaned.split()
        filtered = [t.strip(".") for t in tokens if t.strip(".") and t.strip(".") not in STOPWORDS]
        return " ".join(filtered)
