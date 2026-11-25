from dataclasses import dataclass
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.text_processor import TextProcessor


@dataclass(frozen=True)
class TfidfSimilarityResult:
    sim_exp_raw: float
    sim_exp_score: int
    sim_role_raw: float
    sim_role_score: int
    sim_proj_raw: float
    sim_proj_score: int


class NlpMatcher:
    """
    Tính độ tương đồng văn bản giữa các phần của CV và JD bằng TF-IDF + Cosine Similarity.
    - Sử dụng ngram_range=(1, 2) để bắt được cả từ đơn lẫn cụm 2 từ (vd: 'rest api', 'tối ưu').
    - Sử dụng sublinear_tf=True (1 + log(tf)) để tránh CV nhồi nhét lặp từ khóa làm sai lệch điểm.
    - Chuẩn hóa điểm cosine thô về thang 0..100 (ngưỡng bão hòa 0.40 cho cặp văn bản CV-JD ngắn).
    """

    COSINE_SATURATION_THRESHOLD = 0.40

    @classmethod
    def compute_cosine(cls, text_a: str, text_b: str) -> float:
        prep_a = TextProcessor.preprocess_for_tfidf(text_a)
        prep_b = TextProcessor.preprocess_for_tfidf(text_b)

        if not prep_a or not prep_b:
            return 0.0
        if len(prep_a.split()) < 2 or len(prep_b.split()) < 2:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b[\w+#.]+\b",
            )
            tfidf_matrix = vectorizer.fit_transform([prep_a, prep_b])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return max(0.0, min(1.0, sim))
        except ValueError:
            return 0.0

    @classmethod
    def calibrate_cosine_to_score(cls, raw_cosine: float) -> int:
        if raw_cosine <= 0.0:
            return 0
        scaled = (raw_cosine / cls.COSINE_SATURATION_THRESHOLD) * 100.0
        return max(0, min(100, int(round(scaled))))

    @classmethod
    def match_cv_and_jd(
        cls,
        cv_sections: dict,
        cv_full_text: str,
        job_title: str,
        job_description: str,
        job_requirements: str,
    ) -> TfidfSimilarityResult:
        # 1. Kinh nghiệm & dự án CV vs Mô tả & yêu cầu JD
        cv_exp_proj = "\n".join(
            part for part in [cv_sections.get("experience", ""), cv_sections.get("projects", "")] if part
        ).strip()
        if len(cv_exp_proj) < 40:
            cv_exp_proj = cv_full_text

        jd_body = "\n".join(part for part in [job_description, job_requirements] if part).strip()
        sim_exp_raw = cls.compute_cosine(cv_exp_proj, jd_body)

        # 2. Chức danh & mục tiêu/tóm tắt CV vs Tiêu đề & mô tả JD
        cv_role_text = "\n".join(
            part
            for part in [
                cv_sections.get("header", ""),
                cv_sections.get("objective", ""),
            ]
            if part
        ).strip()
        if len(cv_role_text) < 20:
            cv_role_text = cv_full_text[:600]

        jd_role_text = "\n".join(part for part in [job_title, job_description] if part).strip()
        sim_role_raw = cls.compute_cosine(cv_role_text, jd_role_text)

        # 3. Dự án CV vs Yêu cầu JD
        cv_proj = cv_sections.get("projects", "").strip()
        sim_proj_raw = cls.compute_cosine(cv_proj, job_requirements or jd_body) if cv_proj else sim_exp_raw * 0.8

        return TfidfSimilarityResult(
            sim_exp_raw=round(sim_exp_raw, 4),
            sim_exp_score=cls.calibrate_cosine_to_score(sim_exp_raw),
            sim_role_raw=round(sim_role_raw, 4),
            sim_role_score=cls.calibrate_cosine_to_score(sim_role_raw),
            sim_proj_raw=round(sim_proj_raw, 4),
            sim_proj_score=cls.calibrate_cosine_to_score(sim_proj_raw),
        )
