import re
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz

from app.schemas.match_schema import EvidenceItem, SkillMatchItem
from app.services.skill_extractor import JdSkillRequirement, SkillExtractor
from app.services.text_processor import TextProcessor


class EvidenceVerifier:
    """
    Kiểm chứng bằng chứng từ CV gốc để đảm bảo mọi nhận xét đều có căn cứ xác thực.
    - Không tự tạo câu trích dẫn không tồn tại trong CV.
    - Phân biệt rõ bằng chứng chính xác (direct_quote) và khớp gần đúng (fuzzy_match).
    """

    FUZZY_EVIDENCE_THRESHOLD = 88

    def __init__(self, skill_extractor: SkillExtractor):
        self.skill_extractor = skill_extractor

        # Tạo map canonical -> danh sách toàn bộ surface (canonical + aliases)
        self._canonical_surfaces: Dict[str, List[str]] = {}
        for surface, canonical in skill_extractor._surface_to_canonical.items():
            self._canonical_surfaces.setdefault(canonical, []).append(surface)

        self._surface_patterns: Dict[str, List[re.Pattern]] = {}
        for canonical, surfaces in self._canonical_surfaces.items():
            patterns: List[re.Pattern] = []
            for s in sorted(surfaces, key=len, reverse=True):
                escaped = re.escape(s)
                patterns.append(
                    re.compile(rf"(?<![a-z0-9à-ỹ+#.]){escaped}(?![a-z0-9à-ỹ+#]|\.[a-z0-9à-ỹ])", re.IGNORECASE)
                )
            self._surface_patterns[canonical] = patterns

    def find_evidence_for_skill(
        self,
        skill_name: str,
        original_sentences: List[str],
        cv_lower_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Tìm câu nguyên văn trong CV chứng minh kỹ năng `skill_name`.
        Trả về (quote, evidence_type) với evidence_type in {"direct_quote", "fuzzy_match"} hoặc (None, None).
        """
        norm = self.skill_extractor.normalize_skill(skill_name)
        patterns = self._surface_patterns.get(norm)
        if not patterns:
            escaped = re.escape(norm)
            patterns = [
                re.compile(rf"(?<![a-z0-9à-ỹ+#.]){escaped}(?![a-z0-9à-ỹ+#]|\.[a-z0-9à-ỹ])", re.IGNORECASE)
            ]

        is_short = norm in self.skill_extractor.short_ambiguous

        # 1. Tìm kiếm chính xác trong danh sách câu gốc
        for sent in original_sentences:
            sent_low = sent.lower()
            for pat in patterns:
                m = pat.search(sent_low)
                if not m:
                    continue
                if is_short and not self.skill_extractor._has_short_skill_context(
                    cv_lower_text, norm, m.start(), m.end()
                ):
                    continue
                return sent, "direct_quote"

        # 2. Nếu không khớp chính xác và tên kỹ năng đủ dài (>= 5 ký tự), thử fuzzy match làm gợi ý
        if len(norm) >= 5 and not is_short:
            best_sent = None
            best_score = 0.0
            for sent in original_sentences:
                score = fuzz.partial_ratio(norm, sent.lower())
                if score >= self.FUZZY_EVIDENCE_THRESHOLD and score > best_score:
                    best_score = score
                    best_sent = sent
            if best_sent:
                return best_sent, "fuzzy_match"

        return None, None

    def verify_jd_skills_against_cv(
        self,
        jd_skills: List[JdSkillRequirement],
        cv_raw_text: str,
        parsed_profile_skills: Optional[List[str]] = None,
        data_sufficient: bool = True,
    ) -> Tuple[List[SkillMatchItem], List[EvidenceItem]]:
        """
        Đối chiếu từng kỹ năng của JD với văn bản CV gốc, gắn trạng thái bằng chứng trung thực:
        - evidenced: có trích dẫn nguyên văn trong CV (direct_quote)
        - unclear_evidence: chỉ khớp gần đúng (fuzzy_match) hoặc có trong parsed_profile nhưng không tìm được câu rõ ràng
        - not_found: không tìm thấy trong CV (không quy chụp là không biết)
        - insufficient_data: dữ liệu CV không đủ để đánh giá
        """
        original_sentences = TextProcessor.split_sentences(cv_raw_text)
        cv_lower = TextProcessor.clean_text(cv_raw_text, max_chars=0).lower()
        parsed_norm_set = {self.skill_extractor.normalize_skill(s) for s in (parsed_profile_skills or []) if s}

        results: List[SkillMatchItem] = []
        evidences: List[EvidenceItem] = []

        for req in jd_skills:
            if not data_sufficient:
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        evidence_status="insufficient_data",
                        evidence_type=None,
                        evidence=None,
                        verified=False,
                        comment="Dữ liệu CV quá ngắn hoặc thiếu nội dung để đánh giá kỹ năng này.",
                    )
                )
                continue

            quote, ev_type = self.find_evidence_for_skill(req.name, original_sentences, cv_lower)

            if quote and ev_type == "direct_quote":
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        evidence_status="evidenced",
                        evidence_type="direct_quote",
                        evidence=quote,
                        verified=True,
                        comment="Tìm thấy bằng chứng trực tiếp trong CV.",
                    )
                )
                evidences.append(
                    EvidenceItem(
                        skill=req.name,
                        quote=quote,
                        evidence_type="direct_quote",
                        verified=True,
                    )
                )
            elif quote and ev_type == "fuzzy_match":
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        evidence_status="unclear_evidence",
                        evidence_type="fuzzy_match",
                        evidence=quote,
                        verified=True,
                        comment="Tìm thấy cụm từ gần đúng trong CV, cần nhà tuyển dụng kiểm tra lại ngữ cảnh.",
                    )
                )
                evidences.append(
                    EvidenceItem(
                        skill=req.name,
                        quote=quote,
                        evidence_type="fuzzy_match",
                        verified=True,
                    )
                )
            elif req.name in parsed_norm_set:
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        evidence_status="unclear_evidence",
                        evidence_type=None,
                        evidence=None,
                        verified=False,
                        comment="Có ghi nhận trong hồ sơ phân tích trước đó nhưng chưa trích xuất được câu nguyên văn rõ ràng.",
                    )
                )
            else:
                req_label = "bắt buộc" if req.is_required else "ưu tiên"
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        evidence_status="not_found",
                        evidence_type=None,
                        evidence=None,
                        verified=True,
                        comment=f"Không tìm thấy thông tin đề cập đến {req.name} trong CV (tiêu chí {req_label}).",
                    )
                )

        return results, evidences
