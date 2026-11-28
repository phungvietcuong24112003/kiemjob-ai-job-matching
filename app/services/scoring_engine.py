import re
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.match_schema import (
    ConfidenceDetail,
    EducationEvaluation,
    ExperienceEvaluation,
    MatchRequest,
    MatchResponse,
    MissingRequirementItem,
    RecommendationDetail,
    SkillCategory,
    SkillMatchItem,
    StructuredStrengthItem,
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
        """
        Tính điểm cho 1 nhóm kỹ năng:
        - Phân biệt bắt buộc (80%) và ưu tiên (20%).
        - Trong nhóm bắt buộc, ưu tiên kỹ năng NTD chỉ định trực tiếp từ bài đăng (source='db': 70%)
          so với kỹ năng AI tự trích xuất từ văn bản JD (source='jd_text': 30%).
        """
        if not items:
            return None

        required = [i for i in items if i.is_required]
        preferred = [i for i in items if not i.is_required]

        def avg_coef(group: List[SkillMatchItem]) -> float:
            if not group:
                return 0.0
            return sum(cls.STATUS_COEF.get(i.evidence_status, 0.0) for i in group) / len(group)

        def required_coef(group: List[SkillMatchItem]) -> float:
            explicit = [i for i in group if i.source == "db"]
            from_jd = [i for i in group if i.source != "db"]
            if explicit and from_jd:
                return 0.70 * avg_coef(explicit) + 0.30 * avg_coef(from_jd)
            return avg_coef(group)

        if required and preferred:
            ratio = 0.80 * required_coef(required) + 0.20 * avg_coef(preferred)
        elif required:
            ratio = required_coef(required)
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
        jd_sufficient = (
            len(jd_combined.strip()) >= 20
            or len(request.job.required_skills_from_db) > 0
            or len(request.job.explicit_required_skills) > 0
        )
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
                confidence_detail=ConfidenceDetail(
                    level="LOW",
                    reason="Nội dung CV quá ngắn (< 50 ký tự) hoặc tin tuyển dụng thiếu mô tả/yêu cầu.",
                ),
                recommendation=RecommendationDetail(
                    level="INSUFFICIENT_DATA",
                    label="Chưa có đủ thông tin để đánh giá",
                    description="Chưa có đủ dữ liệu văn bản từ CV hoặc JD. Nhà tuyển dụng nên mở file PDF gốc hoặc yêu cầu ứng viên bổ sung hồ sơ.",
                ),
                experience_evaluation=ExperienceEvaluation(
                    assessment="Không đủ dữ liệu văn bản từ CV hoặc tin tuyển dụng để đánh giá."
                ),
                education_evaluation=EducationEvaluation(
                    required_education=request.job.education_requirement or "Không yêu cầu",
                    assessment="Không đủ dữ liệu để đối chiếu học vấn.",
                ),
                weaknesses=["Nội dung CV quá ngắn (< 50 ký tự) hoặc tin tuyển dụng thiếu thông tin mô tả/yêu cầu."],
                recommendations=["Kiểm tra lại file PDF CV hoặc bổ sung đầy đủ mô tả/kỹ năng cho tin tuyển dụng."],
                analysis_summary="Chưa có đủ dữ liệu để thực hiện đánh giá mức độ phù hợp.",
                algorithm_version=settings.ALGORITHM_VERSION,
            )

        cv_sections = TextProcessor.split_cv_sections(cv_raw)

        # 1. Trích xuất & phân nhóm kỹ năng JD (đã khử trùng lặp giữa các nhóm)
        jd_skills = self.skill_extractor.extract_jd_skills(
            description=jd_desc,
            requirements=jd_reqs,
            required_skills_from_db=request.job.required_skills_from_db,
            explicit_required_skills=request.job.explicit_required_skills,
        )

        # 2. Kiểm chứng từng kỹ năng trên CV gốc
        parsed_skills = request.cv.parsed_profile.skills if request.cv.parsed_profile else []
        skill_items, evidence_items = self.evidence_verifier.verify_jd_skills_against_cv(
            jd_skills=jd_skills,
            cv_raw_text=cv_raw,
            parsed_profile_skills=parsed_skills,
            data_sufficient=True,
            cv_sections=cv_sections,
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
        explicit_required_skills = [i for i in skill_items if i.source == "db"]

        jd_sentence_map = {s.name: s.jd_sentence for s in jd_skills}

        # Tính độ tin cậy độc lập với điểm tổng
        conf_detail = self._evaluate_confidence(
            cv_raw=cv_raw,
            cv_sections=cv_sections,
            jd_skills=jd_skills,
            matched_skills=matched_skills,
            exp_eval=exp_out.evaluation,
        )
        confidence = conf_detail.level.lower()

        # Sinh nhận xét có cấu trúc, bằng chứng và đề xuất cho NTD
        structured_strengths, missing_requirements, recommendation, strengths, weaknesses, recommendations, summary = (
            self._build_structured_explanations(
                overall_score=overall_score,
                jd_title=jd_title,
                skill_items=skill_items,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                required_skills=required_skills,
                jd_sentence_map=jd_sentence_map,
                exp_eval=exp_out.evaluation,
                edu_eval=edu_out.evaluation,
                conf_detail=conf_detail,
            )
        )

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
            confidence_detail=conf_detail,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            explicit_required_skills=explicit_required_skills,
            experience_evaluation=exp_out.evaluation,
            education_evaluation=edu_out.evaluation,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            structured_strengths=structured_strengths,
            missing_requirements=missing_requirements,
            recommendation=recommendation,
            evidence_from_cv=evidence_items,
            analysis_summary=summary,
            algorithm_version=settings.ALGORITHM_VERSION,
        )

    @staticmethod
    def _evaluate_confidence(
        cv_raw: str,
        cv_sections: dict,
        jd_skills: list,
        matched_skills: List[SkillMatchItem],
        exp_eval: ExperienceEvaluation,
    ) -> ConfidenceDetail:
        reasons: List[str] = []
        has_exp_or_proj = bool(
            (cv_sections.get("experience", "") or "").strip()
            or (cv_sections.get("projects", "") or "").strip()
        )
        unclear_count = sum(1 for s in matched_skills if s.evidence_status == "unclear_evidence")

        if len(cv_raw) < 120:
            reasons.append("Nội dung CV quá ngắn, thiếu chi tiết về quá trình làm việc.")
        if exp_eval.candidate_years is None and not has_exp_or_proj:
            reasons.append("Chưa xác định được kinh nghiệm làm việc hoặc dự án thực tế trong CV.")
        if not jd_skills:
            reasons.append("Tin tuyển dụng chưa liệt kê rõ kỹ năng yêu cầu.")

        if reasons:
            return ConfidenceDetail(level="LOW", reason=" ".join(reasons))

        med_reasons: List[str] = []
        if len(cv_raw) < 280:
            med_reasons.append("CV khá ngắn, thông tin ngữ cảnh còn hạn chế.")
        if exp_eval.candidate_years is None:
            med_reasons.append("Chưa ghi rõ tổng số năm kinh nghiệm.")
        if unclear_count > 0:
            med_reasons.append(f"Có {unclear_count} kỹ năng chỉ xuất hiện ở dạng liệt kê/gần đúng, chưa rõ bối cảnh sử dụng.")

        if med_reasons:
            return ConfidenceDetail(level="MEDIUM", reason=" ".join(med_reasons))

        return ConfidenceDetail(
            level="HIGH",
            reason="CV và JD đều cung cấp đầy đủ thông tin kỹ năng, kinh nghiệm và có minh chứng rõ ràng.",
        )

    @staticmethod
    def _build_structured_explanations(
        overall_score: int,
        jd_title: str,
        skill_items: List[SkillMatchItem],
        matched_skills: List[SkillMatchItem],
        missing_skills: List[SkillMatchItem],
        required_skills: List[SkillMatchItem],
        jd_sentence_map: Dict[str, Optional[str]],
        exp_eval: ExperienceEvaluation,
        edu_eval: EducationEvaluation,
        conf_detail: ConfidenceDetail,
    ) -> Tuple[
        List[StructuredStrengthItem],
        List[MissingRequirementItem],
        RecommendationDetail,
        List[str],
        List[str],
        List[str],
        str,
    ]:
        structured_strengths: List[StructuredStrengthItem] = []
        missing_requirements: List[MissingRequirementItem] = []
        strengths_txt: List[str] = []
        weaknesses_txt: List[str] = []
        recs_txt: List[str] = []

        # 1. Điểm mạnh có căn cứ
        evidenced_items = [s for s in matched_skills if s.evidence_status == "evidenced"]
        for item in evidenced_items[:6]:
            structured_strengths.append(
                StructuredStrengthItem(
                    title=f"Kỹ năng {item.name.upper() if len(item.name) <= 3 else item.name.title()}",
                    description=item.comment or f"CV có đề cập và minh chứng sử dụng {item.name}, đáp ứng yêu cầu trong JD.",
                    evidence=item.evidence,
                )
            )

        evidenced_req = [s.name for s in evidenced_items if s.is_required]
        evidenced_pref = [s.name for s in evidenced_items if not s.is_required]
        if evidenced_req:
            strengths_txt.append(
                f"CV có minh chứng trực tiếp cho {len(evidenced_req)}/{len(required_skills)} yêu cầu bắt buộc: {', '.join(evidenced_req)}."
            )
        if evidenced_pref:
            strengths_txt.append(f"Đáp ứng thêm kỹ năng ưu tiên: {', '.join(evidenced_pref)}.")

        if exp_eval.candidate_years is not None and (
            exp_eval.required_min_years is None or exp_eval.candidate_years >= exp_eval.required_min_years
        ):
            structured_strengths.append(
                StructuredStrengthItem(
                    title="Kinh nghiệm làm việc phù hợp",
                    description=exp_eval.assessment,
                    evidence=None,
                )
            )
            strengths_txt.append(exp_eval.assessment)

        if edu_eval.meets_requirement and edu_eval.required_education != "Không yêu cầu":
            structured_strengths.append(
                StructuredStrengthItem(
                    title="Học vấn đạt yêu cầu",
                    description=edu_eval.assessment,
                    evidence=None,
                )
            )

        # 2. Yêu cầu còn thiếu hoặc chưa rõ minh chứng (KHÔNG quy chụp không biết)
        unclear_items = [s for s in matched_skills if s.evidence_status == "unclear_evidence"]
        for item in unclear_items:
            missing_requirements.append(
                MissingRequirementItem(
                    title=f"{item.name} (Chưa rõ mức độ sử dụng)",
                    description=item.comment or f"CV có đề cập đến {item.name} nhưng chưa cung cấp đủ thông tin về bối cảnh sử dụng.",
                    jd_requirement=jd_sentence_map.get(item.name),
                    status="unclear_evidence",
                )
            )
            weaknesses_txt.append(
                f"CV có đề cập {item.name} nhưng chưa có minh chứng rõ ràng về bối cảnh/mức độ sử dụng."
            )

        for item in missing_skills:
            req_tag = "bắt buộc" if item.is_required else "ưu tiên"
            missing_requirements.append(
                MissingRequirementItem(
                    title=f"{item.name} ({req_tag})",
                    description=f"Chưa tìm thấy minh chứng về {item.name} trong CV (tiêu chí {req_tag}).",
                    jd_requirement=jd_sentence_map.get(item.name),
                    status="not_found",
                )
            )
            weaknesses_txt.append(f"Chưa tìm thấy minh chứng về {item.name} trong CV (tiêu chí {req_tag}).")

        if exp_eval.required_min_years is not None and exp_eval.required_min_years > 0:
            if exp_eval.candidate_years is None:
                missing_requirements.append(
                    MissingRequirementItem(
                        title="Thời gian kinh nghiệm",
                        description=f"JD yêu cầu tối thiểu {exp_eval.required_min_years} năm kinh nghiệm nhưng chưa xác định được số năm từ CV.",
                        jd_requirement=f"Yêu cầu tối thiểu {exp_eval.required_min_years} năm kinh nghiệm",
                        status="insufficient_data",
                    )
                )
                weaknesses_txt.append(f"Chưa xác định được số năm kinh nghiệm từ CV (JD yêu cầu {exp_eval.required_min_years} năm).")
            elif exp_eval.candidate_years < exp_eval.required_min_years:
                missing_requirements.append(
                    MissingRequirementItem(
                        title="Số năm kinh nghiệm",
                        description=f"Kinh nghiệm ghi nhận trong CV ({exp_eval.candidate_years} năm) thấp hơn ngưỡng JD ({exp_eval.required_min_years} năm).",
                        jd_requirement=f"Yêu cầu tối thiểu {exp_eval.required_min_years} năm kinh nghiệm",
                        status="not_found",
                    )
                )

        if not edu_eval.meets_requirement:
            missing_requirements.append(
                MissingRequirementItem(
                    title="Yêu cầu học vấn",
                    description=edu_eval.assessment,
                    jd_requirement=edu_eval.required_education,
                    status="not_found",
                )
            )
            weaknesses_txt.append(edu_eval.assessment)

        # 3. Đề xuất cho Nhà tuyển dụng (kết hợp điểm số + độ tin cậy + kỹ năng bắt buộc)
        missing_req_names = [s.name for s in missing_skills if s.is_required]
        if conf_detail.level == "LOW":
            rec = RecommendationDetail(
                level="INSUFFICIENT_DATA",
                label="Chưa có đủ thông tin để đánh giá",
                description=(
                    f"Độ tin cậy thấp ({conf_detail.reason}). Nhà tuyển dụng nên đọc trực tiếp file PDF gốc "
                    "hoặc trao đổi thêm với ứng viên trước khi kết luận."
                ),
            )
            recs_txt.append(rec.description)
        elif overall_score >= 70 and not missing_req_names and not unclear_items:
            rec = RecommendationDetail(
                level="CONSIDER_INTERVIEW",
                label="Nên cân nhắc phỏng vấn",
                description=(
                    f"Ứng viên đạt {overall_score}/100 và có minh chứng trực tiếp cho các yêu cầu cốt lõi. "
                    "Nên cân nhắc mời phỏng vấn kỹ thuật để đánh giá chiều sâu."
                ),
            )
            recs_txt.append(rec.description)
        elif overall_score >= 60:
            focus = ", ".join((missing_req_names + [u.name for u in unclear_items])[:4])
            rec = RecommendationDetail(
                level="CONSIDER_INTERVIEW",
                label="Cân nhắc phỏng vấn (cần xác minh thêm)",
                description=(
                    f"Ứng viên đạt {overall_score}/100. Nên cân nhắc phỏng vấn và tập trung xác minh thêm về: {focus}."
                    if focus
                    else f"Ứng viên đạt {overall_score}/100. Nên phỏng vấn để kiểm tra kinh nghiệm thực tế."
                ),
            )
            recs_txt.append(rec.description)
        else:
            rec = RecommendationDetail(
                level="NEED_MORE_REVIEW",
                label="Cần xem xét thêm",
                description=(
                    f"Mức độ khớp hiện tại là {overall_score}/100 do chưa tìm thấy nhiều minh chứng phù hợp với JD. "
                    "Nhà tuyển dụng nên xem xét thêm hồ sơ gốc nếu muốn cân nhắc cho vị trí khác hoặc đào tạo lại."
                ),
            )
            recs_txt.append(rec.description)

        # 4. Nhận định tổng quan nhất quán với điểm số & độ tin cậy
        level_label = (
            "mức độ phù hợp cao"
            if overall_score >= 75
            else ("mức độ phù hợp khá" if overall_score >= 55 else "mức độ phù hợp thấp")
        )
        req_total = len(required_skills)
        req_hit = len(evidenced_req)
        skill_part = f"Đáp ứng có minh chứng {req_hit}/{req_total} yêu cầu bắt buộc" if req_total > 0 else "Đã đối chiếu nội dung CV với JD"
        summary = (
            f"Ứng viên có {level_label} ({overall_score}/100) với vị trí {jd_title}. "
            f"{skill_part}. {exp_eval.assessment}"
        )

        return structured_strengths, missing_requirements, rec, strengths_txt, weaknesses_txt, recs_txt, summary
