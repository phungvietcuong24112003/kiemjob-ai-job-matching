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

    _USAGE_CONTEXT_RE = re.compile(
        r"(?i)\b(kinh\s*nghiệm|dự\s*án|phát\s*triển|xây\s*dựng|sử\s*dụng|triển\s*khai|"
        r"làm\s*việc|tích\s*hợp|thiết\s*kế|quản\s*lý|lập\s*trình|vận\s*hành|tối\s*ưu|"
        r"developed|built|using|used|experience|project|projects|implemented|deployed|"
        r"designed|managed|worked|created|năm|years?|tháng|months?)\b"
    )

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
        exp_proj_sentences: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Tìm câu nguyên văn trong CV chứng minh kỹ năng `skill_name`.
        Trả về (quote, evidence_type, has_usage_context).
        """
        norm = self.skill_extractor.normalize_skill(skill_name)
        patterns = self._surface_patterns.get(norm)
        if not patterns:
            escaped = re.escape(norm)
            patterns = [
                re.compile(rf"(?<![a-z0-9à-ỹ+#.]){escaped}(?![a-z0-9à-ỹ+#]|\.[a-z0-9à-ỹ])", re.IGNORECASE)
            ]

        is_short = norm in self.skill_extractor.short_ambiguous
        exp_proj_set = {s.strip() for s in (exp_proj_sentences or [])}

        # 1. Ưu tiên tìm câu vừa khớp chính xác vừa có ngữ cảnh sử dụng / nằm trong experience/projects
        first_exact_sent: Optional[str] = None
        for sent in original_sentences:
            sent_low = sent.lower()
            matched = False
            for pat in patterns:
                m = pat.search(sent_low)
                if not m:
                    continue
                if is_short and not self.skill_extractor._has_short_skill_context(
                    cv_lower_text, norm, m.start(), m.end()
                ):
                    continue
                matched = True
                break
            if matched:
                if first_exact_sent is None:
                    first_exact_sent = sent
                has_ctx = (sent.strip() in exp_proj_set) or bool(self._USAGE_CONTEXT_RE.search(sent))
                if has_ctx:
                    return sent, "direct_quote", True

        if first_exact_sent is not None:
            return first_exact_sent, "direct_quote", False

        # 2. Nếu không khớp chính xác và tên kỹ năng đủ dài (>= 5 ký tự), thử fuzzy match làm gợi ý
        if len(norm) >= 5 and not is_short:
            best_sent = None
            best_score = 0.0
            is_multi_word = len(norm.split()) >= 2
            for sent in original_sentences:
                sent_low = sent.lower()
                score = fuzz.partial_ratio(norm, sent_low)
                if is_multi_word:
                    score = max(score, fuzz.token_set_ratio(norm, sent_low))
                if score >= self.FUZZY_EVIDENCE_THRESHOLD and score > best_score:
                    best_score = score
                    best_sent = sent
            if best_sent:
                return best_sent, "fuzzy_match", False

        return None, None, False

    def verify_jd_skills_against_cv(
        self,
        jd_skills: List[JdSkillRequirement],
        cv_raw_text: str,
        parsed_profile_skills: Optional[List[str]] = None,
        data_sufficient: bool = True,
        cv_sections: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[SkillMatchItem], List[EvidenceItem]]:
        """
        Đối chiếu từng kỹ năng của JD với văn bản CV gốc, gắn trạng thái bằng chứng trung thực:
        - evidenced: có trích dẫn nguyên văn trong CV kèm ngữ cảnh sử dụng (kinh nghiệm/dự án/hành động)
        - unclear_evidence: chỉ xuất hiện trong danh sách từ khóa rời rạc chưa rõ cách dùng hoặc khớp gần đúng
        - not_found: không tìm thấy trong CV (không quy chụp là không biết)
        - insufficient_data: dữ liệu CV không đủ để đánh giá
        """
        original_sentences = TextProcessor.split_sentences(cv_raw_text)
        cv_lower = TextProcessor.clean_text(cv_raw_text, max_chars=0).lower()
        parsed_norm_set = {self.skill_extractor.normalize_skill(s) for s in (parsed_profile_skills or []) if s}

        exp_proj_text = ""
        if cv_sections:
            exp_proj_text = "\n".join(
                p for p in [cv_sections.get("experience", ""), cv_sections.get("projects", "")] if p
            )
        exp_proj_sentences = TextProcessor.split_sentences(exp_proj_text) if exp_proj_text else []

        results: List[SkillMatchItem] = []
        evidences: List[EvidenceItem] = []

        for req in jd_skills:
            if not data_sufficient:
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        source=req.source,
                        evidence_status="insufficient_data",
                        evidence_type=None,
                        evidence=None,
                        verified=False,
                        comment="Dữ liệu CV quá ngắn hoặc thiếu nội dung để đánh giá kỹ năng này.",
                    )
                )
                continue

            quote, ev_type, has_usage_ctx = self.find_evidence_for_skill(
                req.name, original_sentences, cv_lower, exp_proj_sentences
            )

            if quote and ev_type == "direct_quote" and has_usage_ctx:
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        source=req.source,
                        evidence_status="evidenced",
                        evidence_type="direct_quote",
                        evidence=quote,
                        verified=True,
                        comment=f"CV có đề cập và minh chứng sử dụng {req.name}, đáp ứng yêu cầu trong JD.",
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
            elif quote and ev_type == "direct_quote" and not has_usage_ctx:
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        source=req.source,
                        evidence_status="unclear_evidence",
                        evidence_type="direct_quote",
                        evidence=quote,
                        verified=True,
                        comment=f"CV có đề cập đến {req.name} nhưng chưa cung cấp đủ thông tin về bối cảnh hoặc mức độ sử dụng.",
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
                        source=req.source,
                        evidence_status="unclear_evidence",
                        evidence_type="fuzzy_match",
                        evidence=quote,
                        verified=True,
                        comment=f"Tìm thấy cụm từ gần đúng liên quan đến {req.name} trong CV, cần nhà tuyển dụng kiểm tra lại ngữ cảnh.",
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
                        source=req.source,
                        evidence_status="unclear_evidence",
                        evidence_type=None,
                        evidence=None,
                        verified=False,
                        comment=f"Có ghi nhận {req.name} trong hồ sơ phân tích trước đó nhưng chưa trích xuất được câu nguyên văn rõ ràng.",
                    )
                )
            else:
                req_label = "bắt buộc" if req.is_required else "ưu tiên"
                results.append(
                    SkillMatchItem(
                        name=req.name,
                        category=req.category,
                        is_required=req.is_required,
                        source=req.source,
                        evidence_status="not_found",
                        evidence_type=None,
                        evidence=None,
                        verified=True,
                        comment=f"Chưa tìm thấy minh chứng về {req.name} trong CV (tiêu chí {req_label}).",
                    )
                )

        return results, evidences
