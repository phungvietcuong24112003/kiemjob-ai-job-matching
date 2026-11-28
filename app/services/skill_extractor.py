import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from rapidfuzz import fuzz, process

from app.schemas.match_schema import SkillCategory
from app.services.text_processor import TextProcessor


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_TAXONOMY_PATH = os.path.join(_DATA_DIR, "skill_taxonomy.json")

_PREFERRED_PATTERNS = re.compile(
    r"(?i)\b(ưu\s*tiên|lợi\s*thế|điểm\s*cộng|là\s+một\s+lợi\s*thế|biết\s+thêm|"
    r"preferred|nice\s+to\s+have|nice-to-have|plus|bonus|optional|advantage)\b"
)

_REQUIRED_HEADER_RE = re.compile(
    r"(?i)^(yêu\s*cầu\s*(bắt\s*buộc|công\s*việc|chuyên\s*môn)?|required|requirements|must\s+have)\s*:?$"
)
_PREFERRED_HEADER_RE = re.compile(
    r"(?i)^(ưu\s*tiên|điểm\s*cộng|lợi\s*thế|preferred|nice\s+to\s+have|bonus)\s*:?$"
)

_SUPPLEMENTARY_CUSTOM_KEYWORDS = (
    "giao tiếp", "teamwork", "làm việc nhóm", "thuyết trình", "đàm phán",
    "quản lý thời gian", "chịu áp lực", "giải quyết vấn đề", "tin học văn phòng",
    "tiếng anh", "tiếng nhật", "tiếng trung", "tiếng hàn", "ngoại ngữ", "agile", "scrum"
)
_TOOLS_CUSTOM_KEYWORDS = (
    "misa", "sap", "erp", "crm", "spss", "stata", "power bi", "tableau",
    "figma", "canva", "photoshop", "illustrator", "autocad", "solidworks",
    "excel", "word", "powerpoint", "google analytics", "jira", "notion"
)


@dataclass(frozen=True)
class ExtractedSkill:
    name: str
    category: SkillCategory
    match_type: str  # "exact" | "synonym" | "fuzzy"
    surface: str


@dataclass(frozen=True)
class JdSkillRequirement:
    name: str
    category: SkillCategory
    is_required: bool
    source: str  # "db" | "jd_text"
    jd_sentence: Optional[str] = None


class SkillExtractor:
    """
    Trích xuất, chuẩn hóa và phân nhóm kỹ năng từ CV và JD.
    Đảm bảo tính loại trừ lẫn nhau (Mutual Exclusivity): mỗi kỹ năng chỉ thuộc 1 nhóm duy nhất.
    """

    def __init__(self, taxonomy_path: str = _TAXONOMY_PATH):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.synonyms: Dict[str, str] = {
            k.strip().lower(): v.strip().lower()
            for k, v in data.get("synonyms", {}).items()
        }
        self.short_ambiguous: Dict[str, List[str]] = {
            k.strip().lower(): [c.lower() for c in v]
            for k, v in data.get("short_ambiguous_skills", {}).items()
        }

        self.skill_to_category: Dict[str, SkillCategory] = {}
        priority_order: List[SkillCategory] = ["core_skills", "tools_and_tech", "supplementary_skills"]
        taxonomy = data.get("taxonomy", {})

        for cat in priority_order:
            subgroups = taxonomy.get(cat, {})
            for _, skill_list in subgroups.items():
                for skill in skill_list:
                    norm = skill.strip().lower()
                    # Đảm bảo không ghi đè nếu đã xuất hiện ở nhóm ưu tiên cao hơn
                    if norm not in self.skill_to_category:
                        self.skill_to_category[norm] = cat

        # Tập hợp toàn bộ từ khóa có thể nhận diện (canonical + synonyms)
        self._surface_to_canonical: Dict[str, str] = {}
        for canonical in self.skill_to_category:
            self._surface_to_canonical[canonical] = canonical
        for alias, canonical in self.synonyms.items():
            if canonical in self.skill_to_category:
                self._surface_to_canonical[alias] = canonical

        # Sắp xếp từ dài đến ngắn để match cụm dài trước (vd: "spring boot" trước "spring")
        sorted_surfaces = sorted(self._surface_to_canonical.keys(), key=len, reverse=True)
        self._patterns: List[Tuple[str, str, re.Pattern]] = []
        for surface in sorted_surfaces:
            canonical = self._surface_to_canonical[surface]
            escaped = re.escape(surface)
            pattern = re.compile(
                rf"(?<![a-z0-9à-ỹ+#.]){escaped}(?![a-z0-9à-ỹ+#]|\.[a-z0-9à-ỹ])",
                re.IGNORECASE,
            )
            self._patterns.append((surface, canonical, pattern))

        self._fuzzy_candidates: List[str] = [
            s for s in self._surface_to_canonical.keys()
            if len(s) >= 5 and s not in self.short_ambiguous
        ]

    def normalize_skill(self, raw_skill: str) -> str:
        """Chuẩn hóa tên kỹ năng về dạng chuẩn (canonical)."""
        if not raw_skill:
            return ""
        cleaned = re.sub(r"\s+", " ", TextProcessor.normalize_unicode(raw_skill).strip().lower())
        if cleaned in self._surface_to_canonical:
            return self._surface_to_canonical[cleaned]
        if cleaned in self.synonyms:
            return self.synonyms[cleaned]

        # Fuzzy match chỉ áp dụng cho chuỗi >= 5 ký tự để tránh match nhầm
        if len(cleaned) >= 5 and self._fuzzy_candidates:
            match = process.extractOne(cleaned, self._fuzzy_candidates, scorer=fuzz.ratio)
            if match and match[1] >= 88:
                return self._surface_to_canonical[match[0]]

        return cleaned

    def get_category(self, canonical_skill: str) -> SkillCategory:
        """Trả về đúng 1 nhóm duy nhất của kỹ năng; phân loại thông minh cho kỹ năng tùy chỉnh từ DB."""
        norm = self.normalize_skill(canonical_skill)
        if norm in self.skill_to_category:
            return self.skill_to_category[norm]
        if any(kw in norm for kw in _SUPPLEMENTARY_CUSTOM_KEYWORDS):
            return "supplementary_skills"
        if any(kw in norm for kw in _TOOLS_CUSTOM_KEYWORDS):
            return "tools_and_tech"
        return "core_skills"

    def _has_short_skill_context(self, text_lower: str, skill: str, match_start: int, match_end: int) -> bool:
        """Kiểm tra ngữ cảnh cho các kỹ năng ngắn dễ nhầm (c, r, go, qa, ba, cv)."""
        contexts = self.short_ambiguous.get(skill)
        if not contexts:
            return True
        win_start = max(0, match_start - 80)
        win_end = min(len(text_lower), match_end + 80)
        window = text_lower[win_start:win_end]
        return any(ctx in window for ctx in contexts)

    def extract_skills_from_text(self, text: str) -> List[ExtractedSkill]:
        """Trích xuất toàn bộ kỹ năng xuất hiện trong văn bản, đã khử trùng lặp."""
        if not text:
            return []
        cleaned = TextProcessor.clean_text(text, max_chars=0)
        lower_text = cleaned.lower()

        found: Dict[str, ExtractedSkill] = {}
        for surface, canonical, pattern in self._patterns:
            if canonical in found:
                continue
            m = pattern.search(lower_text)
            if not m:
                continue
            if surface in self.short_ambiguous or canonical in self.short_ambiguous:
                key = surface if surface in self.short_ambiguous else canonical
                if not self._has_short_skill_context(lower_text, key, m.start(), m.end()):
                    continue

            match_type = "exact" if surface == canonical else "synonym"
            found[canonical] = ExtractedSkill(
                name=canonical,
                category=self.get_category(canonical),
                match_type=match_type,
                surface=m.group(0),
            )

        return list(found.values())

    def extract_jd_skills(
        self,
        description: str = "",
        requirements: str = "",
        required_skills_from_db: Optional[List[str]] = None,
        explicit_required_skills: Optional[List[str]] = None,
    ) -> List[JdSkillRequirement]:
        """
        Trích xuất và phân loại kỹ năng yêu cầu từ JD (Required vs Preferred)
        kết hợp với danh sách kỹ năng có cấu trúc từ bài đăng (`required_skills` / `job_skills`).
        """
        skill_map: Dict[str, JdSkillRequirement] = {}

        # 1. Kỹ năng bắt buộc NTD chỉ định trực tiếp (ưu tiên explicit_required_skills, fallback required_skills_from_db)
        db_inputs = explicit_required_skills if explicit_required_skills else required_skills_from_db
        if db_inputs:
            for raw in db_inputs:
                norm = self.normalize_skill(raw)
                if not norm:
                    continue
                skill_map[norm] = JdSkillRequirement(
                    name=norm,
                    category=self.get_category(norm),
                    is_required=True,
                    source="db",
                )

        # 2. Quét từng câu trong requirements và description (nhận diện ngữ cảnh mục Ưu tiên)
        combined_sentences = (
            TextProcessor.split_sentences(requirements or "")
            + TextProcessor.split_sentences(description or "")
        )

        in_preferred_section = False
        for sent in combined_sentences:
            stripped = sent.strip()
            if _PREFERRED_HEADER_RE.match(stripped):
                in_preferred_section = True
                continue
            if _REQUIRED_HEADER_RE.match(stripped):
                in_preferred_section = False
                continue

            is_preferred_sentence = in_preferred_section or bool(_PREFERRED_PATTERNS.search(sent))
            extracted = self.extract_skills_from_text(sent)
            for item in extracted:
                is_req = not is_preferred_sentence
                existing = skill_map.get(item.name)
                if existing is None:
                    skill_map[item.name] = JdSkillRequirement(
                        name=item.name,
                        category=item.category,
                        is_required=is_req,
                        source="jd_text",
                        jd_sentence=sent,
                    )
                else:
                    # Nếu xuất hiện ở db nhưng chưa có câu mô tả JD -> bổ sung jd_sentence để giải thích
                    if existing.jd_sentence is None:
                        skill_map[item.name] = JdSkillRequirement(
                            name=existing.name,
                            category=existing.category,
                            is_required=existing.is_required or is_req,
                            source=existing.source,
                            jd_sentence=sent,
                        )
                    elif existing.source == "jd_text" and is_req and not existing.is_required:
                        skill_map[item.name] = JdSkillRequirement(
                            name=item.name,
                            category=item.category,
                            is_required=True,
                            source="jd_text",
                            jd_sentence=sent,
                        )

        return list(skill_map.values())
