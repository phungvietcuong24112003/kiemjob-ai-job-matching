import re
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.match_schema import (
    EducationEvaluation,
    ExperienceEvaluation,
    MatchRequest,
    MatchResponse,
    SkillCategory,
    SkillMatchItem,
)
from app.services.evidence_verifier import EvidenceVerifier
from app.services.exp_edu_analyzer import ExpEduAnalyzer
from app.services.nlp_matcher import NlpMatcher
from app.services.skill_extractor import SkillExtractor
from app.services.text_processor import TextProcessor


class ScoringEngine:
    """
    Bộ điều phối thuật toán NLP chấm điểm CV - Job xác định (Deterministic).
    - Trọng số chuẩn: Kỹ năng chuyên môn 35%, Kinh nghiệm 25%, Phù hợp công việc 15%,
      Công nghệ & công cụ 10%, Học vấn 10%, Kỹ năng bổ trợ 5%.
    - Sử dụng Chuẩn hóa trọng số động (Dynamic Weight Normalization) nếu Job không yêu cầu một tiêu chí.
    """

    WEIGHTS: Dict[str, float] = {
        "skill": 0.35,
        "experience": 0.25,
        "job_relevance": 0.15,
        "technology": 0.10,
        "education": 0.10,
        "supplementary_skill": 0.05,
    }

    STATUS_COEF: Dict[str, float] = {
        "evidenced": 1.0,
        "unclear_evidence": 0.5,
        "not_found": 0.0,
        "insufficient_data": 0.0,
    }

    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.evidence_verifier = EvidenceVerifier(self.skill_extractor)

    @classmethod
    def compute_skill_group_score(cls, items: List[SkillMatchItem]) -> Optional[int]:
        """Tính điểm cho 1 nhóm kỹ năng, phân biệt bắt buộc (80%) và ưu tiên (20%)."""
        if not items:
            return None

        required = [i for i in items if i.is_required]
        preferred = [i for i in items if not i.is_required]

        def avg_coef(group: List[SkillMatchItem]) -> float:
            if not group:
                return 0.0
            return sum(cls.STATUS_COEF.get(i.evidence_status, 0.0) for i in group) / len(group)

        if required and preferred:
            ratio = 0.80 * avg_coef(required) + 0.20 * avg_coef(preferred)
        elif required:
            ratio = avg_coef(required)
        else:
            ratio = avg_coef(preferred)

        return max(0, min(100, int(round(ratio * 100))))

    _TITLE_SYNONYMS: Dict[str, List[str]] = {
        "developer": ["engineer", "lập trình viên", "programmer", "dev"],
        "engineer": ["developer", "kỹ sư", "lập trình viên"],
        "business": ["nghiệp vụ"],
        "analyst": ["phân tích", "chuyên viên phân tích"],
        "accountant": ["kế toán"],
        "tester": ["qa", "qc", "kiểm thử"],
    }

    @classmethod
    def compute_title_match_ratio(
        cls,
        job_title: str,
        cv_sections: dict,
        cv_full_text: str,
        parsed_titles: Optional[List[str]] = None,
    ) -> float:
        if not job_title or not job_title.strip():
            return 0.5
        jt_low = TextProcessor.normalize_unicode(job_title).lower().strip()
        jt_tokens = [t for t in re.split(r"[\s,/\-|()]+", jt_low) if len(t) >= 2]

        search_space = "\n".join(
            [
                cv_sections.get("header", ""),
                cv_sections.get("objective", ""),
                " ".join(parsed_titles or []),
                cv_full_text[:500],
            ]
        ).lower()

        if jt_low in search_space:
            return 1.0
        if not jt_tokens:
            return 0.0

        hits = 0
        for tok in jt_tokens:
            candidates = [tok] + cls._TITLE_SYNONYMS.get(tok, [])
            if any(re.search(rf"(?<![a-z0-9à-ỹ]){re.escape(c)}(?![a-z0-9à-ỹ])", search_space) for c in candidates):
                hits += 1
        return round(hits / len(jt_tokens), 2)

    def evaluate(self, request: MatchRequest) -> MatchResponse:
        cv_raw = TextProcessor.clean_text(request.cv.raw_text, max_chars=settings.MAX_CV_CHARS)
        jd_desc = TextProcessor.clean_text(request.job.description or "", max_chars=settings.MAX_JOB_CHARS // 2)
        jd_reqs = TextProcessor.clean_text(request.job.requirements or "", max_chars=settings.MAX_JOB_CHARS // 2)
        jd_title = TextProcessor.clean_text(request.job.title, max_chars=300)

        jd_combined = "\n".join(part for part in [jd_title, jd_desc, jd_reqs] if part).strip()
        cv_sufficient = len(cv_raw.strip()) >= 50
        jd_sufficient = len(jd_combined.strip()) >= 20 or len(request.job.required_skills_from_db) > 0
        data_sufficient = cv_sufficient and jd_sufficient

        if not data_sufficient:
            return MatchResponse(
                overall_score=0,
                skill_score=None,
                experience_score=0,
                job_relevance_score=0,
                technology_score=None,
                education_score=None,
                supplementary_skill_score=None,
                data_sufficiency="insufficient",
                confidence="low",
                experience_evaluation=ExperienceEvaluation(
                    assessment="Không đủ dữ liệu văn bản từ CV hoặc tin tuyển dụng để đánh giá."
                ),
                education_evaluation=EducationEvaluation(
                    required_education=request.job.education_requirement or "Không yêu cầu",
                    assessment="Không đủ dữ liệu để đối chiếu học vấn.",
                ),
                weaknesses=["Nội dung CV quá ngắn (< 50 ký tự) hoặc tin tuyển dụng thiếu thông tin mô tả/yêu cầu."],
                recommendations=["Kiểm tra lại file PDF CV hoặc bổ sung đầy đủ mô tả/kỹ năng cho tin tuyển dụng."],
                analysis_summary="Không đủ dữ liệu để thực hiện đánh giá mức độ phù hợp.",
                algorithm_version=settings.ALGORITHM_VERSION,
            )

        cv_sections = TextProcessor.split_cv_sections(cv_raw)

        # 1. Trích xuất & phân nhóm kỹ năng JD (đã khử trùng lặp giữa các nhóm)
        jd_skills = self.skill_extractor.extract_jd_skills(
            description=jd_desc,
            requirements=jd_reqs,
            required_skills_from_db=request.job.required_skills_from_db,
        )

        # 2. Kiểm chứng từng kỹ năng trên CV gốc
        parsed_skills = request.cv.parsed_profile.skills if request.cv.parsed_profile else []
        skill_items, evidence_items = self.evidence_verifier.verify_jd_skills_against_cv(
            jd_skills=jd_skills,
            cv_raw_text=cv_raw,
            parsed_profile_skills=parsed_skills,
            data_sufficient=True,
        )

        # Phân chia vào 3 nhóm điểm không trùng lặp
        by_cat: Dict[SkillCategory, List[SkillMatchItem]] = {
            "core_skills": [],
            "tools_and_tech": [],
            "supplementary_skills": [],
        }
        for item in skill_items:
            by_cat[item.category].append(item)

        skill_score = self.compute_skill_group_score(by_cat["core_skills"])
        technology_score = self.compute_skill_group_score(by_cat["tools_and_tech"])
        supplementary_skill_score = self.compute_skill_group_score(by_cat["supplementary_skills"])

        # 3. Tính TF-IDF & Cosine Similarity
        tfidf_res = NlpMatcher.match_cv_and_jd(
            cv_sections=cv_sections,
            cv_full_text=cv_raw,
            job_title=jd_title,
            job_description=jd_desc,
            job_requirements=jd_reqs,
        )

        # 4. Đánh giá Kinh nghiệm (3 trục) & Học vấn
        parsed_years = request.cv.parsed_profile.years_experience if request.cv.parsed_profile else None
        exp_out = ExpEduAnalyzer.analyze_experience(
            cv_text=cv_raw,
            cv_sections=cv_sections,
            parsed_years=parsed_years,
            jd_text=jd_combined,
            jd_skills=jd_skills,
            sim_exp_score=tfidf_res.sim_exp_score,
            sim_exp_raw=tfidf_res.sim_exp_raw,
            skill_extractor=self.skill_extractor,
        )

        parsed_edu = request.cv.parsed_profile.education_level if request.cv.parsed_profile else None
        edu_out = ExpEduAnalyzer.analyze_education(
            cv_text=cv_raw,
            cv_sections=cv_sections,
            parsed_edu=parsed_edu,
            job_edu_req=request.job.education_requirement,
            jd_text=jd_combined,
        )

        # 5. Mức độ phù hợp công việc (Title match 60% + TF-IDF role 40%)
        parsed_titles = request.cv.parsed_profile.job_titles if request.cv.parsed_profile else []
        title_ratio = self.compute_title_match_ratio(jd_title, cv_sections, cv_raw, parsed_titles)
        job_relevance_score = max(
            0, min(100, int(round(0.60 * (title_ratio * 100.0) + 0.40 * tfidf_res.sim_role_score)))
        )

        # 6. Chuẩn hóa trọng số động (Dynamic Weight Normalization)
        active_scores: List[Tuple[float, int]] = [
            (self.WEIGHTS["experience"], exp_out.score),
            (self.WEIGHTS["job_relevance"], job_relevance_score),
        ]
        if skill_score is not None:
            active_scores.append((self.WEIGHTS["skill"], skill_score))
        if technology_score is not None:
            active_scores.append((self.WEIGHTS["technology"], technology_score))
        if edu_out.score is not None:
            active_scores.append((self.WEIGHTS["education"], edu_out.score))
        if supplementary_skill_score is not None:
            active_scores.append((self.WEIGHTS["supplementary_skill"], supplementary_skill_score))

        total_weight = sum(w for w, _ in active_scores)
        overall_score = int(round(sum(w * s for w, s in active_scores) / total_weight)) if total_weight > 0 else 0
        overall_score = max(0, min(100, overall_score))

        # Phân loại danh sách kết quả
        matched_skills = [i for i in skill_items if i.evidence_status in ("evidenced", "unclear_evidence")]
        missing_skills = [i for i in skill_items if i.evidence_status in ("not_found", "insufficient_data")]
        required_skills = [i for i in skill_items if i.is_required]
        preferred_skills = [i for i in skill_items if not i.is_required]

        # Sinh nhận xét minh bạch dựa trên dữ liệu đã tính
        strengths, weaknesses, recommendations, summary = self._build_narratives(
            overall_score=overall_score,
            jd_title=jd_title,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            required_skills=required_skills,
            exp_eval=exp_out.evaluation,
            edu_eval=edu_out.evaluation,
        )

        confidence = "high"
        if len(cv_raw) < 150 or not jd_skills:
            confidence = "medium"
        if any(i.evidence_status == "unclear_evidence" for i in matched_skills) and len(matched_skills) <= 1:
            confidence = "medium"

        return MatchResponse(
            overall_score=overall_score,
            skill_score=skill_score,
            experience_score=exp_out.score,
            job_relevance_score=job_relevance_score,
            technology_score=technology_score,
            education_score=edu_out.score,
            supplementary_skill_score=supplementary_skill_score,
            data_sufficiency="sufficient",
            confidence=confidence,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_evaluation=exp_out.evaluation,
            education_evaluation=edu_out.evaluation,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evidence_from_cv=evidence_items,
            analysis_summary=summary,
            algorithm_version=settings.ALGORITHM_VERSION,
        )

    @staticmethod
    def _build_narratives(
        overall_score: int,
        jd_title: str,
        matched_skills: List[SkillMatchItem],
        missing_skills: List[SkillMatchItem],
        required_skills: List[SkillMatchItem],
        exp_eval: ExperienceEvaluation,
        edu_eval: EducationEvaluation,
    ) -> Tuple[List[str], List[str], List[str], str]:
        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []

        evidenced_req = [s.name for s in matched_skills if s.is_required and s.evidence_status == "evidenced"]
        evidenced_pref = [s.name for s in matched_skills if not s.is_required and s.evidence_status == "evidenced"]
        missing_req = [s.name for s in missing_skills if s.is_required]
        missing_pref = [s.name for s in missing_skills if not s.is_required]

        if evidenced_req:
            strengths.append(
                f"Có bằng chứng trực tiếp cho {len(evidenced_req)}/{len(required_skills)} yêu cầu bắt buộc: {', '.join(evidenced_req)}."
            )
        if evidenced_pref:
            strengths.append(f"Đáp ứng thêm kỹ năng ưu tiên: {', '.join(evidenced_pref)}.")
        if exp_eval.candidate_years is not None and (
            exp_eval.required_min_years is None or exp_eval.candidate_years >= exp_eval.required_min_years
        ):
            strengths.append(exp_eval.assessment)

        if missing_req:
            weaknesses.append(f"Chưa tìm thấy bằng chứng trong CV cho yêu cầu bắt buộc: {', '.join(missing_req)}.")
            recommendations.append(f"Phỏng vấn làm rõ kinh nghiệm thực tế của ứng viên với: {', '.join(missing_req)}.")
        if missing_pref:
            weaknesses.append(f"Chưa thấy đề cập kỹ năng ưu tiên: {', '.join(missing_pref)}.")
            recommendations.append(f"Hỏi thêm xem ứng viên đã từng tìm hiểu hoặc sử dụng {', '.join(missing_pref)} chưa.")
        if not edu_eval.meets_requirement:
            weaknesses.append(edu_eval.assessment)

        if not recommendations:
            recommendations.append("Hồ sơ đáp ứng tốt các tiêu chí chính; nên mời phỏng vấn chuyên môn để kiểm tra độ sâu.")

        level_label = (
            "Phù hợp cao" if overall_score >= 75 else ("Phù hợp một phần" if overall_score >= 50 else "Mức độ phù hợp thấp")
        )
        summary = f"{level_label} ({overall_score}/100) cho vị trí {jd_title}. {exp_eval.assessment}"
        return strengths, weaknesses, recommendations, summary
