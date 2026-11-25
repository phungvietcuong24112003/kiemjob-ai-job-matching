import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from app.schemas.match_schema import EducationEvaluation, ExperienceEvaluation
from app.services.skill_extractor import JdSkillRequirement, SkillExtractor
from app.services.text_processor import TextProcessor


_EDU_RANKS = {
    "khong_yeu_cau": 0,
    "thpt": 1,
    "trung_cap": 2,
    "cao_dang": 3,
    "dai_hoc": 4,
    "thac_si": 5,
    "tien_si": 6,
}

_EDU_LABELS = {
    "khong_yeu_cau": "Không yêu cầu",
    "thpt": "THPT",
    "trung_cap": "Trung cấp",
    "cao_dang": "Cao đẳng",
    "dai_hoc": "Đại học",
    "thac_si": "Thạc sĩ",
    "tien_si": "Tiến sĩ",
}

_MAJOR_GROUPS = {
    "it": [
        "công nghệ thông tin", "cntt", "khoa học máy tính", "kỹ thuật phần mềm",
        "hệ thống thông tin", "an toàn thông tin", "computer science", "software engineering",
        "information technology", "toán tin", "điện tử viễn thông"
    ],
    "economics_accounting": [
        "kế toán", "kiểm toán", "tài chính", "ngân hàng", "quản trị kinh doanh",
        "kinh tế", "accounting", "finance", "business administration"
    ],
    "marketing": [
        "marketing", "truyền thông", "thương mại điện tử", "digital marketing", "báo chí"
    ],
}


@dataclass
class ExpAnalysisOutput:
    score: int
    evaluation: ExperienceEvaluation


@dataclass
class EduAnalysisOutput:
    score: Optional[int]
    evaluation: EducationEvaluation


class ExpEduAnalyzer:
    """Phân tích kinh nghiệm (số năm + TF-IDF nhiệm vụ + công nghệ trong dự án) và học vấn."""

    @staticmethod
    def extract_years_from_cv(cv_text: str, parsed_years: Optional[float] = None) -> Optional[float]:
        if parsed_years is not None and parsed_years >= 0:
            return round(float(parsed_years), 1)

        if not cv_text:
            return None

        # 1. Tìm mẫu số năm rõ ràng: "1 năm kinh nghiệm", "2.5 years of experience", "18 tháng"
        year_matches = re.findall(
            r"(?i)(\d+(?:[.,]\d+)?)\s*\+?\s*(?:năm|years?|yrs?)\s*(?:kinh\s*nghiệm|of\s+experience|exp|làm\s*việc)?",
            cv_text,
        )
        valid_years: List[float] = []
        for m in year_matches:
            try:
                val = float(m.replace(",", "."))
                if 0.0 < val <= 35.0:
                    valid_years.append(val)
            except ValueError:
                continue

        if valid_years:
            return round(max(valid_years), 1)

        month_matches = re.findall(
            r"(?i)(\d+)\s*(?:tháng|months?)\s*(?:kinh\s*nghiệm|of\s+experience|thực\s*tập)?",
            cv_text,
        )
        if month_matches:
            try:
                max_months = max(int(m) for m in month_matches if 0 < int(m) <= 120)
                return round(max_months / 12.0, 1)
            except ValueError:
                pass

        # 2. Ước lượng theo khoảng thời gian MM/YYYY - MM/YYYY hoặc YYYY - Hiện tại
        current_year = datetime.now().year
        ranges = re.findall(
            r"(?i)(?:0?[1-9]|1[0-2])?[/\-]?(20[0-2]\d)\s*[-–—~đến]+\s*(?:(?:0?[1-9]|1[0-2])?[/\-]?(20[0-2]\d)|(hiện\s*tại|present|now|current))",
            cv_text,
        )
        total_span = 0.0
        for start_y, end_y, is_curr in ranges:
            try:
                sy = int(start_y)
                ey = current_year if is_curr else int(end_y)
                if 2000 <= sy <= ey <= current_year + 1:
                    total_span += max(0.5, float(ey - sy))
            except ValueError:
                continue

        if total_span > 0:
            return round(min(25.0, total_span), 1)

        return None

    @staticmethod
    def extract_required_years_from_jd(jd_text: str) -> Tuple[Optional[float], Optional[float]]:
        if not jd_text:
            return None, None

        # Mẫu khoảng: "1-2 năm", "1 - 3 years"
        range_m = re.search(
            r"(?i)(\d+(?:[.,]\d+)?)\s*[-–~đến]+\s*(\d+(?:[.,]\d+)?)\s*(?:năm|years?)",
            jd_text,
        )
        if range_m:
            try:
                y1 = float(range_m.group(1).replace(",", "."))
                y2 = float(range_m.group(2).replace(",", "."))
                return min(y1, y2), max(y1, y2)
            except ValueError:
                pass

        # Mẫu tối thiểu: "tối thiểu 2 năm", "từ 1 năm", "2+ years"
        min_m = re.search(
            r"(?i)(?:tối\s*thiểu|ít\s*nhất|từ|trên|minimum|at\s+least)?\s*(\d+(?:[.,]\d+)?)\s*\+?\s*(?:năm|years?)",
            jd_text,
        )
        if min_m:
            try:
                y = float(min_m.group(1).replace(",", "."))
                if 0.0 < y <= 20.0:
                    return y, None
            except ValueError:
                pass

        if re.search(r"(?i)\b(fresher|intern|thực\s*tập|không\s*yêu\s*cầu\s*kinh\s*nghiệm)\b", jd_text):
            return 0.0, None

        return None, None

    @classmethod
    def analyze_experience(
        cls,
        cv_text: str,
        cv_sections: dict,
        parsed_years: Optional[float],
        jd_text: str,
        jd_skills: List[JdSkillRequirement],
        sim_exp_score: int,
        sim_exp_raw: float,
        skill_extractor: SkillExtractor,
    ) -> ExpAnalysisOutput:
        candidate_years = cls.extract_years_from_cv(cv_text, parsed_years)
        req_min_years, _ = cls.extract_required_years_from_jd(jd_text)

        exp_proj_text = "\n".join(
            part for part in [cv_sections.get("experience", ""), cv_sections.get("projects", "")] if part
        ).strip()
        has_exp_section = len(exp_proj_text) >= 30

        # 1. Điểm số năm kinh nghiệm (35%)
        if req_min_years is None or req_min_years == 0.0:
            if candidate_years is not None and candidate_years > 0:
                years_ratio = 1.0
            elif has_exp_section:
                years_ratio = 0.85
            else:
                years_ratio = 0.40
        else:
            if candidate_years is not None:
                years_ratio = min(1.0, max(0.0, candidate_years / req_min_years))
            elif has_exp_section:
                years_ratio = 0.35
            else:
                years_ratio = 0.0

        # 2. Độ tương đồng nhiệm vụ qua TF-IDF (40%)
        sim_ratio = max(0.0, min(1.0, sim_exp_score / 100.0))

        # 3. Tỷ lệ kỹ năng JD xuất hiện trong bối cảnh kinh nghiệm / dự án CV (25%)
        context_text = exp_proj_text if has_exp_section else cv_text
        context_skills = {s.name for s in skill_extractor.extract_skills_from_text(context_text)}
        if jd_skills:
            matched_in_proj = sum(1 for s in jd_skills if s.name in context_skills)
            project_tech_ratio = matched_in_proj / len(jd_skills)
        else:
            project_tech_ratio = sim_ratio

        exp_score = int(round((0.35 * years_ratio + 0.40 * sim_ratio + 0.25 * project_tech_ratio) * 100))
        exp_score = max(0, min(100, exp_score))

        # Sinh nhận xét minh bạch, không suy diễn
        parts: List[str] = []
        if candidate_years is not None:
            if req_min_years is not None and req_min_years > 0:
                if candidate_years >= req_min_years:
                    parts.append(f"Ứng viên có {candidate_years} năm kinh nghiệm, đạt yêu cầu tối thiểu ({req_min_years} năm).")
                else:
                    parts.append(f"Ứng viên có {candidate_years} năm kinh nghiệm (thấp hơn mức yêu cầu {req_min_years} năm).")
            else:
                parts.append(f"Ghi nhận {candidate_years} năm kinh nghiệm trong CV.")
        else:
            if has_exp_section:
                parts.append("CV có mục kinh nghiệm/dự án nhưng chưa ghi rõ tổng số năm kinh nghiệm.")
            else:
                parts.append("Không tìm thấy thông tin rõ ràng về kinh nghiệm làm việc trong CV.")

        if jd_skills:
            parts.append(
                f"Mức độ áp dụng công nghệ vào dự án/công việc đạt {int(round(project_tech_ratio * 100))}% "
                f"(TF-IDF similarity: {sim_exp_raw:.2f})."
            )

        return ExpAnalysisOutput(
            score=exp_score,
            evaluation=ExperienceEvaluation(
                candidate_years=candidate_years,
                required_min_years=req_min_years,
                tfidf_similarity=sim_exp_raw,
                project_tech_ratio=round(project_tech_ratio, 2),
                assessment=" ".join(parts),
            ),
        )

    @staticmethod
    def parse_education_level(text: str) -> str:
        if not text:
            return "khong_yeu_cau"
        low = TextProcessor.normalize_unicode(text).lower()
        if re.search(r"(?i)(không\s*yêu\s*cầu|bất\s*kỳ|not\s+required|no\s+requirement)", low):
            return "khong_yeu_cau"
        if re.search(r"(?i)\b(tiến\s*sĩ|ph\.?d|doctorate)\b", low):
            return "tien_si"
        if re.search(r"(?i)\b(thạc\s*sĩ|master|m\.sc|mba)\b", low):
            return "thac_si"
        if re.search(r"(?i)\b(đại\s*học|cử\s*nhân|kỹ\s*sư|bachelor|university|b\.sc|b\.eng)\b", low):
            return "dai_hoc"
        if re.search(r"(?i)\b(cao\s*đẳng|college|associate)\b", low):
            return "cao_dang"
        if re.search(r"(?i)\b(trung\s*cấp|vocational)\b", low):
            return "trung_cap"
        if re.search(r"(?i)\b(thpt|trung\s*học\s*phổ\s*thông|high\s*school)\b", low):
            return "thpt"
        return "khong_yeu_cau"

    @staticmethod
    def detect_major_group(text: str) -> Optional[str]:
        if not text:
            return None
        low = TextProcessor.normalize_unicode(text).lower()
        for group, keywords in _MAJOR_GROUPS.items():
            if any(kw in low for kw in keywords):
                return group
        return None

    @classmethod
    def analyze_education(
        cls,
        cv_text: str,
        cv_sections: dict,
        parsed_edu: Optional[str],
        job_edu_req: Optional[str],
        jd_text: str,
    ) -> EduAnalysisOutput:
        req_level = cls.parse_education_level(job_edu_req or "")
        if req_level == "khong_yeu_cau":
            req_level = cls.parse_education_level(jd_text or "")

        edu_text = cv_sections.get("education", "").strip() or cv_text
        cand_level = parsed_edu.lower().strip() if parsed_edu and parsed_edu.lower().strip() in _EDU_RANKS else cls.parse_education_level(edu_text)

        req_label = _EDU_LABELS.get(req_level, "Không yêu cầu")
        cand_label = _EDU_LABELS.get(cand_level) if cand_level != "khong_yeu_cau" else None

        # Nếu JD không yêu cầu học vấn -> trả None để bỏ qua khi chuẩn hóa trọng số động
        if req_level == "khong_yeu_cau":
            return EduAnalysisOutput(
                score=None,
                evaluation=EducationEvaluation(
                    required_education="Không yêu cầu",
                    candidate_education=cand_label,
                    meets_requirement=True,
                    assessment="Tin tuyển dụng không yêu cầu bắt buộc bằng cấp; học vấn không ảnh hưởng bất lợi đến điểm tổng.",
                ),
            )

        req_rank = _EDU_RANKS[req_level]
        cand_rank = _EDU_RANKS.get(cand_level, 0)

        jd_major = cls.detect_major_group((job_edu_req or "") + "\n" + (jd_text or ""))
        cv_major = cls.detect_major_group(edu_text)

        if cand_rank == 0:
            score = 40
            meets = False
            msg = f"Yêu cầu trình độ {req_label}, nhưng chưa tìm thấy thông tin bằng cấp rõ ràng trong CV."
        elif cand_rank >= req_rank:
            if jd_major and cv_major and jd_major != cv_major:
                score = 70
                meets = True
                msg = f"Đạt bậc {cand_label} (yêu cầu {req_label}), nhưng chuyên ngành có thể khác nhóm ngành chính."
            else:
                score = 100
                meets = True
                msg = f"Trình độ {cand_label} đáp ứng yêu cầu ({req_label})."
        elif cand_rank == req_rank - 1:
            score = 50
            meets = False
            msg = f"Trình độ {cand_label} tiệm cận yêu cầu ({req_label})."
        else:
            score = 20
            meets = False
            msg = f"Trình độ {cand_label} thấp hơn mức yêu cầu ({req_label})."

        return EduAnalysisOutput(
            score=score,
            evaluation=EducationEvaluation(
                required_education=req_label,
                candidate_education=cand_label,
                meets_requirement=meets,
                assessment=msg,
            ),
        )
