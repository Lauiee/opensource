"""Faster-Whisper 전사 후처리 파이프라인.

의료 STT 텍스트에 대한 포괄적 후처리:
  1. 환각(hallucination) 탐지 및 제거
  2. 필러(filler) 단어 정리
  3. 숫자/날짜/용량 정규화
  4. 구두점 및 문장 경계 교정
  5. 의료 용어 교정 (medterm 엔진 연동)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정 및 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class PostProcessConfig:
    """후처리 파이프라인 설정."""

    # 기능 토글
    remove_hallucinations: bool = True   # 환각 제거
    clean_fillers: bool = True           # 필러 단어 정리
    normalize_numbers: bool = True       # 숫자/용량/혈압 정규화
    fix_punctuation: bool = True         # 구두점 교정
    apply_medical_correction: bool = True  # 의료 용어 교정 (medterm 엔진)

    # 세그먼트 필터링
    min_segment_length: int = 1          # 최소 세그먼트 글자 수

    # 필러 옵션
    keep_fillers: bool = False           # True이면 필러를 [filler] 마커로 표시, False이면 삭제

    # 환각 탐지 임계값
    repeat_threshold: int = 3            # 동일 구문 반복 횟수 임계값

    # 정규화 옵션
    use_arabic_numbers: bool = True      # 한글 숫자를 아라비아 숫자로 변환


@dataclass
class PostProcessResult:
    """후처리 결과."""

    original: str                         # 원본 텍스트
    processed: str                        # 처리된 텍스트
    changes: list[dict] = field(default_factory=list)   # 변경 이력 [{type, before, after, detail}]
    stats: dict = field(default_factory=dict)           # 통계 {removed_hallucinations, cleaned_fillers, ...}


# ──────────────────────────────────────────────
# 상수: 환각 패턴
# ──────────────────────────────────────────────

# Whisper가 자주 생성하는 환각 문구 (정확 일치용)
_WHISPER_ARTIFACT_PHRASES: list[str] = [
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    "구독과 좋아요",
    "구독과 좋아요를 눌러주세요",
    "좋아요와 구독 부탁드립니다",
    "자막 by",
    "자막 제공",
    "영상 제공",
    "다음 시간에 만나요",
    "다음 영상에서 만나요",
    "채널에 오신 것을 환영합니다",
    "오늘도 시청해주셔서 감사합니다",
    "끝까지 시청해주셔서 감사합니다",
]

# 방송국/미디어 환각 키워드
_MEDIA_KEYWORDS: list[str] = [
    "MBC", "KBS", "SBS", "JTBC", "YTN", "MBN", "TV조선",
    "MBC 뉴스", "KBS 뉴스", "SBS 뉴스",
]

# 음악/소리 마커 패턴 (정규식)
_SOUND_MARKER_PATTERNS: list[str] = [
    r"\[음악\]",
    r"\(음악\)",
    r"\[박수\]",
    r"\(박수\)",
    r"\[웃음\]",
    r"\(웃음\)",
    r"\[노래\]",
    r"\(노래\)",
    r"\[소리\]",
    r"\(소리\)",
    r"\[잡음\]",
    r"\(잡음\)",
    r"♪+",
    r"♫+",
    r"🎵+",
    r"🎶+",
    r"\*음악\*",
    r"\*박수\*",
]

# ──────────────────────────────────────────────
# 상수: 필러 패턴
# ──────────────────────────────────────────────

# 단독 필러 단어 (문장 내에서 삭제 대상)
_FILLER_PATTERNS: list[tuple[str, str]] = [
    # (정규식 패턴, 설명)
    (r"음\.{2,}", "음..."),
    (r"어\.{2,}", "어..."),
    (r"그\.{2,}", "그..."),
    (r"아\.{2,}", "아..."),
    (r"저\.{2,}", "저..."),
    (r"에\.{2,}", "에..."),
    (r"으?음+\.{0,}", "음"),
    (r"(?<!\S)음(?=[,.\s]|$)", "음"),
    (r"(?<!\S)어(?=[,.\s]|$)", "어"),
    (r"(?<!\S)그(?=[,.\s]|$)", "그"),
]

# 반복 필러 (무의미한 반복)
_REPEATED_FILLER_PATTERNS: list[tuple[str, str]] = [
    (r"네{3,}", "반복 '네'"),
    (r"예{3,}", "반복 '예'"),
    (r"아{4,}", "반복 '아'"),
    (r"어{4,}", "반복 '어'"),
    (r"으{4,}", "반복 '으'"),
    (r"음{3,}", "반복 '음'"),
    (r"(네\s*){3,}", "반복 '네 '"),
    (r"(예\s*){3,}", "반복 '예 '"),
    (r"(네네)+", "네네 반복"),
    (r"(예예)+", "예예 반복"),
]

# ──────────────────────────────────────────────
# 상수: 한글 숫자 매핑
# ──────────────────────────────────────────────

# 기본 한글 숫자
_KOREAN_DIGITS: dict[str, int] = {
    "영": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
    "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
}

# 고유어 숫자 (관형사형)
_NATIVE_KOREAN_NUMBERS: dict[str, int] = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
    "스물": 20, "서른": 30, "마흔": 40, "쉰": 50,
    "예순": 60, "일흔": 70, "여든": 80, "아흔": 90,
}

# 자릿수 단위
_KOREAN_UNITS: dict[str, int] = {
    "십": 10, "백": 100, "천": 1000, "만": 10000,
}

# 의료 단위 매핑 (한글 → 약어)
_MEDICAL_UNIT_MAP: dict[str, str] = {
    "밀리그램": "mg",
    "마이크로그램": "μg",
    "그램": "g",
    "킬로그램": "kg",
    "밀리리터": "mL",
    "리터": "L",
    "시시": "cc",
    "밀리몰": "mmol",
    "유닛": "U",
    "아이유": "IU",
    "퍼센트": "%",
    "밀리미터": "mm",
    "센티미터": "cm",
}

# ──────────────────────────────────────────────
# 레거시 의료 사전 (medterm 엔진 폴백용)
# ──────────────────────────────────────────────

_LEGACY_MEDICAL_DICT: list[tuple[str, str]] = [
    ("전체 환수로", "전치환술 후"),
    ("전체 환술의", "전치환술의"),
    ("전체 환술을", "전치환술을"),
    ("전체 환술", "전치환술"),
    ("전체환술", "전치환술"),
    ("전체 환수", "전치환술"),
    ("전체환수", "전치환술"),
    ("무혈설 계세사", "무혈성 괴사"),
    ("무혈성 계세사", "무혈성 괴사"),
    ("무혈성계세사", "무혈성 괴사"),
    ("무혈성 계서", "무혈성 괴사"),
    ("계세사증", "괴사증"),
    ("계세사", "괴사"),
    ("관절념", "관절염"),
    ("관절륨", "관절염"),
    ("관질염", "관절염"),
    ("고관질", "고관절"),
    ("이용성증", "이형성증"),
    ("이영성증", "이형성증"),
    ("스테로지를", "스테로이드를"),
    ("스테로지", "스테로이드"),
    ("스테로에즈", "스테로이드"),
    ("릴리카", "리리카"),
    ("니리카", "리리카"),
    ("트리드로", "트리돌"),
    ("세레네스", "세레브렉스"),
    ("코피바", "본비바"),
    ("연고라 골절", "연골하 골절"),
    ("손통제", "진통제"),
    ("고기 고른 즙", "고름집"),
    ("구름집같이", "고름집같이"),
    ("구름집", "고름집"),
    ("액저리", "엑스레이"),
    ("이명옥", "임영욱"),
]


# ──────────────────────────────────────────────
# 메인 후처리 클래스
# ──────────────────────────────────────────────

class TextPostProcessor:
    """STT 텍스트 후처리 파이프라인.

    처리 순서:
      1. 환각 탐지 및 제거
      2. 필러 단어 정리
      3. 숫자/용량/혈압 정규화
      4. 구두점 및 문장 경계 교정
      5. 공백 정리
      6. 의료 용어 교정 (medterm 엔진 또는 레거시 사전)
    """

    def __init__(self, config: Optional[PostProcessConfig] = None):
        self.config = config or PostProcessConfig()

        # 환각 문구 패턴 사전 컴파일
        self._artifact_pattern = self._compile_artifact_pattern()
        self._sound_pattern = self._compile_sound_pattern()

    # ──────────────────────────────────────────
    # 패턴 컴파일 (초기화 시 1회)
    # ──────────────────────────────────────────

    def _compile_artifact_pattern(self) -> re.Pattern:
        """Whisper 환각 문구를 하나의 정규식으로 컴파일."""
        # 특수문자 이스케이프 후 OR 결합
        escaped = [re.escape(phrase) for phrase in _WHISPER_ARTIFACT_PHRASES]
        # 미디어 키워드도 추가
        escaped.extend(re.escape(kw) for kw in _MEDIA_KEYWORDS)
        combined = "|".join(escaped)
        return re.compile(rf"(?:{combined})[\s.!?]*", flags=re.IGNORECASE)

    def _compile_sound_pattern(self) -> re.Pattern:
        """음악/소리 마커 패턴을 하나의 정규식으로 컴파일."""
        combined = "|".join(_SOUND_MARKER_PATTERNS)
        return re.compile(rf"(?:{combined})\s*", flags=re.IGNORECASE)

    # ──────────────────────────────────────────
    # 메인 처리 인터페이스
    # ──────────────────────────────────────────

    def process(self, text: str) -> PostProcessResult:
        """텍스트 전체 후처리 파이프라인 실행.

        Args:
            text: 원본 전사 텍스트

        Returns:
            PostProcessResult: 처리 결과 (원본, 처리 후, 변경 이력, 통계)
        """
        if not text or not text.strip():
            return PostProcessResult(
                original=text,
                processed="",
                changes=[],
                stats={"empty_input": True},
            )

        original = text
        all_changes: list[dict] = []
        stats: dict = {
            "removed_hallucinations": 0,
            "cleaned_fillers": 0,
            "normalized_numbers": 0,
            "fixed_punctuation": 0,
            "medical_corrections": 0,
        }

        # 1단계: 환각 제거
        if self.config.remove_hallucinations:
            text, changes = self.remove_hallucinations(text)
            all_changes.extend(changes)
            stats["removed_hallucinations"] = len(changes)

        # 2단계: 필러 정리
        if self.config.clean_fillers:
            text, changes = self.clean_fillers(text)
            all_changes.extend(changes)
            stats["cleaned_fillers"] = len(changes)

        # 3단계: 숫자 정규화
        if self.config.normalize_numbers:
            text, changes = self.normalize_numbers(text)
            all_changes.extend(changes)
            stats["normalized_numbers"] = len(changes)

        # 4단계: 구두점 교정
        if self.config.fix_punctuation:
            text, changes = self.fix_punctuation(text)
            all_changes.extend(changes)
            stats["fixed_punctuation"] = len(changes)

        # 5단계: 공백 정리 (항상 실행)
        text = self.clean_whitespace(text)

        # 6단계: 의료 용어 교정
        if self.config.apply_medical_correction:
            text, med_count = self._apply_medical_correction(text)
            stats["medical_corrections"] = med_count

        return PostProcessResult(
            original=original,
            processed=text,
            changes=all_changes,
            stats=stats,
        )

    def process_segments(self, segments: list[dict]) -> list[dict]:
        """세그먼트 목록 후처리.

        각 세그먼트의 "text" 필드를 개별 처리하고,
        유효하지 않은 세그먼트는 필터링한다.
        중복 세그먼트도 제거한다.

        Args:
            segments: [{"text": "...", "start": 0.0, "end": 1.0, ...}, ...]

        Returns:
            처리된 세그먼트 목록
        """
        if not segments:
            return segments

        processed_segments: list[dict] = []

        for seg in segments:
            text = seg.get("text", "")
            if not text:
                continue

            # 개별 세그먼트 후처리
            result = self.process(text)
            processed_text = result.processed

            # 유효성 검사
            if not self.is_valid_segment(processed_text):
                continue

            # 세그먼트 복사 후 텍스트 교체
            new_seg = dict(seg)
            new_seg["text"] = processed_text
            # 변경 이력을 세그먼트에 첨부 (디버깅용)
            if result.changes:
                new_seg["_postprocess_changes"] = result.changes

            processed_segments.append(new_seg)

        # 연속 중복 세그먼트 제거
        processed_segments = self._deduplicate_segments(processed_segments)

        return processed_segments

    # ──────────────────────────────────────────
    # 1. 환각 탐지 및 제거
    # ──────────────────────────────────────────

    def remove_hallucinations(self, text: str) -> tuple[str, list[dict]]:
        """환각 패턴을 탐지하고 제거.

        처리 항목:
          - 동일 구문 반복 (3회 이상)
          - Whisper 환각 문구 (감사합니다, 구독과 좋아요 등)
          - 방송국/미디어 키워드
          - 음악/소리 마커 ([음악], ♪ 등)
          - 구두점만으로 이루어진 의미 없는 구간
          - 문장 단위 연속 중복

        Returns:
            (처리된 텍스트, 변경 이력 목록)
        """
        changes: list[dict] = []
        threshold = self.config.repeat_threshold

        # --- (a) 긴 구문 반복 제거 (10자 이상 구문이 threshold회 이상 반복) ---
        pattern_long_repeat = re.compile(r"(.{10,}?)\1{" + str(threshold - 1) + r",}")
        match = pattern_long_repeat.search(text)
        while match:
            repeated = match.group(0)
            single = match.group(1)
            text = text.replace(repeated, single, 1)
            changes.append({
                "type": "hallucination",
                "subtype": "long_repeat",
                "before": repeated[:80] + ("..." if len(repeated) > 80 else ""),
                "after": single[:80] + ("..." if len(single) > 80 else ""),
                "detail": f"긴 구문 {len(repeated) // len(single)}회 반복 → 1회로 축소",
            })
            match = pattern_long_repeat.search(text)

        # --- (b) 짧은 구문 반복 제거 (2~9자 구문이 threshold회 이상 반복) ---
        pattern_short_repeat = re.compile(r"(.{2,9}?)\1{" + str(threshold - 1) + r",}")
        match = pattern_short_repeat.search(text)
        while match:
            repeated = match.group(0)
            single = match.group(1)
            # "네"와 같은 의미 있는 단어 1회 보존, 나머지 삭제
            text = text.replace(repeated, single, 1)
            changes.append({
                "type": "hallucination",
                "subtype": "short_repeat",
                "before": repeated,
                "after": single,
                "detail": f"짧은 구문 반복 제거",
            })
            match = pattern_short_repeat.search(text)

        # --- (c) Whisper 환각 문구 제거 ---
        for m in self._artifact_pattern.finditer(text):
            changes.append({
                "type": "hallucination",
                "subtype": "whisper_artifact",
                "before": m.group(),
                "after": "",
                "detail": "Whisper 환각 문구 제거",
            })
        text = self._artifact_pattern.sub("", text)

        # --- (d) 음악/소리 마커 제거 ---
        for m in self._sound_pattern.finditer(text):
            changes.append({
                "type": "hallucination",
                "subtype": "sound_marker",
                "before": m.group(),
                "after": "",
                "detail": "음악/소리 마커 제거",
            })
        text = self._sound_pattern.sub("", text)

        # --- (e) 감사합니다 반복 (2회 이상) ---
        pattern_thanks = re.compile(r"(감사합니다\.?\s*){2,}", flags=re.IGNORECASE)
        m = pattern_thanks.search(text)
        if m:
            changes.append({
                "type": "hallucination",
                "subtype": "thanks_repeat",
                "before": m.group(),
                "after": "",
                "detail": "'감사합니다' 반복 제거",
            })
            text = pattern_thanks.sub("", text)

        # --- (f) 문장 단위 연속 중복 제거 ---
        sentences = re.split(r"(?<=[.?!])\s+", text)
        if len(sentences) > 1:
            deduped = [sentences[0]]
            dup_count = 0
            for s in sentences[1:]:
                if s.strip() == deduped[-1].strip():
                    dup_count += 1
                else:
                    deduped.append(s)
            if dup_count > 0:
                changes.append({
                    "type": "hallucination",
                    "subtype": "sentence_dedup",
                    "before": f"{dup_count}개 중복 문장",
                    "after": "제거됨",
                    "detail": f"연속 동일 문장 {dup_count}개 제거",
                })
            text = " ".join(deduped)

        # --- (g) 구두점만으로 이루어진 잔여물 제거 ---
        pattern_punct_only = re.compile(r"^\s*[.,!?;:\-–—…·]+\s*$")
        if pattern_punct_only.match(text):
            changes.append({
                "type": "hallucination",
                "subtype": "punct_only",
                "before": text,
                "after": "",
                "detail": "구두점만으로 이루어진 텍스트 제거",
            })
            text = ""

        return text.strip(), changes

    # ──────────────────────────────────────────
    # 2. 필러 단어 처리
    # ──────────────────────────────────────────

    def clean_fillers(self, text: str) -> tuple[str, list[dict]]:
        """필러(filler) 단어를 정리.

        - 반복 필러 (네네네, 아아아아) → 제거 또는 1회로 축소
        - 단독 필러 (음..., 어..., 그...) → 제거 또는 마커 표시
        - 의미 있는 "네" (확인/동의)는 보존

        Returns:
            (처리된 텍스트, 변경 이력 목록)
        """
        changes: list[dict] = []
        keep = self.config.keep_fillers

        # --- (a) 반복 필러 제거 ---
        for pattern_str, desc in _REPEATED_FILLER_PATTERNS:
            pattern = re.compile(pattern_str)
            for m in pattern.finditer(text):
                matched = m.group()
                if keep:
                    replacement = "[filler]"
                else:
                    replacement = ""
                changes.append({
                    "type": "filler",
                    "subtype": "repeated",
                    "before": matched,
                    "after": replacement,
                    "detail": f"{desc} 제거",
                })
            text = pattern.sub("[filler]" if keep else "", text)

        # --- (b) 단독 필러 (음..., 어..., 그..., 아..., 저..., 에...) 제거 ---
        for pattern_str, desc in _FILLER_PATTERNS:
            pattern = re.compile(pattern_str)
            for m in pattern.finditer(text):
                matched = m.group()
                # 의미 있는 "그"(대명사/접속사 역할)는 보존
                # → "그래서", "그런데" 등의 일부가 아닌 독립 "그"만 처리
                if keep:
                    replacement = "[filler]"
                else:
                    replacement = ""
                changes.append({
                    "type": "filler",
                    "subtype": "single",
                    "before": matched,
                    "after": replacement,
                    "detail": f"필러 '{desc}' 제거",
                })
            text = pattern.sub("[filler]" if keep else "", text)

        # --- (c) 문장 끝 반복 필러 제거 ("네네네", "아아아" 등이 문장 끝에) ---
        pattern_trailing = re.compile(r"([네예아어으음])\1{2,}\s*([.!?]?\s*)$")
        m = pattern_trailing.search(text)
        if m:
            matched = m.group()
            changes.append({
                "type": "filler",
                "subtype": "trailing",
                "before": matched,
                "after": "",
                "detail": "문장 끝 반복 필러 제거",
            })
            text = pattern_trailing.sub("", text)

        return text.strip(), changes

    # ──────────────────────────────────────────
    # 3. 숫자 및 단위 정규화
    # ──────────────────────────────────────────

    def normalize_numbers(self, text: str) -> tuple[str, list[dict]]:
        """한글 숫자를 아라비아 숫자로, 의료 단위를 약어로 변환.

        처리 항목:
          - 한자어 숫자: "삼 개월" → "3개월"
          - 고유어 숫자: "두 달" → "2달", "세 번" → "3번"
          - 용량: "오백 밀리그램" → "500mg"
          - 혈압: "백이십 / 팔십" → "120/80"
          - 복용 빈도: "하루 세 번" → "하루 3번", "일일이회" → "1일 2회"

        Returns:
            (처리된 텍스트, 변경 이력 목록)
        """
        changes: list[dict] = []

        if not self.config.use_arabic_numbers:
            return text, changes

        # --- (a) 혈압 패턴: "백이십 / 팔십" → "120/80" ---
        text, bp_changes = self._normalize_blood_pressure(text)
        changes.extend(bp_changes)

        # --- (b) 복용 빈도 패턴 ---
        text, freq_changes = self._normalize_frequency(text)
        changes.extend(freq_changes)

        # --- (c) 한자어 숫자 + 의료 단위 ---
        text, dose_changes = self._normalize_dosage(text)
        changes.extend(dose_changes)

        # --- (d) 한자어 숫자 + 일반 단위 (개월, 일, 주, 년 등) ---
        text, unit_changes = self._normalize_sino_with_unit(text)
        changes.extend(unit_changes)

        # --- (e) 고유어 숫자 + 단위 (번, 달, 개, 알, 정, 캡슐 등) ---
        text, native_changes = self._normalize_native_with_unit(text)
        changes.extend(native_changes)

        return text, changes

    def _korean_sino_to_int(self, text: str) -> Optional[int]:
        """한자어 숫자 문자열을 정수로 변환.

        예: "삼백이십오" → 325, "천이백" → 1200, "오십" → 50
        """
        if not text:
            return None

        text = text.strip().replace(" ", "")
        result = 0
        current = 0

        i = 0
        while i < len(text):
            char = text[i]

            if char in _KOREAN_DIGITS:
                current = _KOREAN_DIGITS[char]
                i += 1
            elif char in _KOREAN_UNITS:
                unit_val = _KOREAN_UNITS[char]
                if current == 0:
                    current = 1  # "백" → 100 ("일백"이 아니라 "백"만 쓴 경우)
                if unit_val >= 10000:  # 만 단위
                    result = (result + current) * unit_val
                    current = 0
                else:
                    result += current * unit_val
                    current = 0
                i += 1
            else:
                # 인식 불가 문자 → 변환 실패
                return None

        result += current
        return result if result > 0 else None

    def _normalize_blood_pressure(self, text: str) -> tuple[str, list[dict]]:
        """혈압 패턴 정규화: "백이십 / 팔십" → "120/80"."""
        changes: list[dict] = []

        # 한자어 숫자 문자 클래스
        sino_chars = "".join(_KOREAN_DIGITS.keys()) + "".join(_KOREAN_UNITS.keys())
        # 혈압 패턴: <한자어숫자> (/ 또는 에) <한자어숫자>
        bp_pattern = re.compile(
            rf"([{sino_chars}]+)\s*[/에]\s*([{sino_chars}]+)"
        )

        def bp_replacer(m: re.Match) -> str:
            systolic = self._korean_sino_to_int(m.group(1))
            diastolic = self._korean_sino_to_int(m.group(2))
            if systolic and diastolic and 60 <= systolic <= 300 and 30 <= diastolic <= 200:
                replacement = f"{systolic}/{diastolic}"
                changes.append({
                    "type": "number",
                    "subtype": "blood_pressure",
                    "before": m.group(),
                    "after": replacement,
                    "detail": "혈압 정규화",
                })
                return replacement
            return m.group()

        text = bp_pattern.sub(bp_replacer, text)
        return text, changes

    def _normalize_frequency(self, text: str) -> tuple[str, list[dict]]:
        """복용 빈도 정규화."""
        changes: list[dict] = []

        # 한자어만 숫자 변환 (고유어 한두세네는 한국어 유지)
        freq_patterns = [
            # "일일 이회" / "일 일 이 회" → "1일 2회"
            (r"일\s*일\s*([일이삼사오육칠팔구])\s*회", self._freq_sino_replacer),
        ]

        for pattern_str, replacer_func in freq_patterns:
            pattern = re.compile(pattern_str)

            def make_replacer(func):
                def replacer(m: re.Match) -> str:
                    result = func(m)
                    if result != m.group():
                        changes.append({
                            "type": "number",
                            "subtype": "frequency",
                            "before": m.group(),
                            "after": result,
                            "detail": "복용 빈도 정규화",
                        })
                    return result
                return replacer

            text = pattern.sub(make_replacer(replacer_func), text)

        return text, changes

    def _freq_native_replacer(self, m: re.Match) -> str:
        """고유어 빈도 변환: "하루 세 번" → "하루 3번"."""
        num_str = m.group(1)
        num_val = _NATIVE_KOREAN_NUMBERS.get(num_str)
        if num_val:
            return f"하루 {num_val}번"
        return m.group()

    def _freq_sino_replacer(self, m: re.Match) -> str:
        """한자어 빈도 변환: "일일이회" → "1일 2회"."""
        num_str = m.group(1)
        num_val = _KOREAN_DIGITS.get(num_str)
        if num_val is not None:
            return f"1일 {num_val}회"
        return m.group()

    def _normalize_dosage(self, text: str) -> tuple[str, list[dict]]:
        """용량 정규화: "오백 밀리그램" → "500mg"."""
        changes: list[dict] = []
        sino_chars = "".join(_KOREAN_DIGITS.keys()) + "".join(_KOREAN_UNITS.keys())

        # 한자어 숫자 + 의료 단위
        for unit_kr, unit_abbr in _MEDICAL_UNIT_MAP.items():
            pattern = re.compile(rf"([{sino_chars}]+)\s*{re.escape(unit_kr)}")

            def make_replacer(abbr):
                def replacer(m: re.Match) -> str:
                    num = self._korean_sino_to_int(m.group(1))
                    if num:
                        result = f"{num}{abbr}"
                        changes.append({
                            "type": "number",
                            "subtype": "dosage",
                            "before": m.group(),
                            "after": result,
                            "detail": f"용량 정규화 ({abbr})",
                        })
                        return result
                    return m.group()
                return replacer

            text = pattern.sub(make_replacer(unit_abbr), text)

        return text, changes

    def _normalize_sino_with_unit(self, text: str) -> tuple[str, list[dict]]:
        """한자어 숫자 + 일반 단위 정규화: "삼 개월" → "3개월"."""
        changes: list[dict] = []
        sino_chars = "".join(_KOREAN_DIGITS.keys()) + "".join(_KOREAN_UNITS.keys())

        # 일반 단위 목록
        general_units = [
            "개월", "주", "년", "일", "시간", "분", "초",
            "세", "살", "번째", "회차", "차", "단계", "기",
            "도", "호",
        ]

        for unit in general_units:
            pattern = re.compile(rf"([{sino_chars}]+)\s*{re.escape(unit)}")

            def make_replacer(u):
                def replacer(m: re.Match) -> str:
                    num = self._korean_sino_to_int(m.group(1))
                    if num:
                        result = f"{num}{u}"
                        changes.append({
                            "type": "number",
                            "subtype": "sino_unit",
                            "before": m.group(),
                            "after": result,
                            "detail": f"한자어 숫자+단위 정규화 ({u})",
                        })
                        return result
                    return m.group()
                return replacer

            text = pattern.sub(make_replacer(unit), text)

        return text, changes

    def _normalize_native_with_unit(self, text: str) -> tuple[str, list[dict]]:
        """고유어 숫자 + 단위 정규화: "두 달" → "2달", "세 알" → "3알"."""
        changes: list[dict] = []

        # 고유어(한두세네)는 숫자 변환 안 함 — 한국어 유지
        native_units: list[str] = []

        # 고유어 숫자 패턴 (키 길이 역순으로 정렬 — 긴 것 먼저 매칭)
        sorted_native = sorted(_NATIVE_KOREAN_NUMBERS.keys(), key=len, reverse=True)
        native_pattern_str = "|".join(re.escape(k) for k in sorted_native)

        for unit in native_units:
            pattern = re.compile(rf"({native_pattern_str})\s*{re.escape(unit)}")

            def make_replacer(u):
                def replacer(m: re.Match) -> str:
                    num_str = m.group(1)
                    num_val = _NATIVE_KOREAN_NUMBERS.get(num_str)
                    if num_val:
                        result = f"{num_val}{u}"
                        changes.append({
                            "type": "number",
                            "subtype": "native_unit",
                            "before": m.group(),
                            "after": result,
                            "detail": f"고유어 숫자+단위 정규화 ({u})",
                        })
                        return result
                    return m.group()
                return replacer

            text = pattern.sub(make_replacer(unit), text)

        return text, changes

    # ──────────────────────────────────────────
    # 4. 구두점 및 문장 경계 교정
    # ──────────────────────────────────────────

    def fix_punctuation(self, text: str) -> tuple[str, list[dict]]:
        """구두점 및 문장 경계를 교정.

        처리 항목:
          - 연속 마침표 정리 ("..." → ".", ".." → ".")
          - 연속 동일 구두점 축소 ("!!" → "!", "??" → "?")
          - 구두점 앞 불필요 공백 제거 (" ." → ".")
          - 구두점 뒤 공백 추가 ("문장.다음" → "문장. 다음")
          - 문장 끝 마침표 추가 (마지막 문자가 한글이면)

        Returns:
            (처리된 텍스트, 변경 이력 목록)
        """
        changes: list[dict] = []
        original = text

        # --- (a) 연속 마침표 정리 ("..." "....." → ".") ---
        text = re.sub(r"\.{2,}", ".", text)

        # --- (b) 연속 동일 구두점 축소 ---
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r",{2,}", ",", text)

        # --- (c) 구두점 앞 불필요 공백 제거 ---
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)

        # --- (d) 구두점 뒤 공백 추가 (한글/영문 문자가 바로 이어지면) ---
        # 단, 숫자 사이의 "/"는 건너뜀 (혈압 120/80 등)
        text = re.sub(r"([.!?])\s*([가-힣a-zA-Z])", r"\1 \2", text)

        # --- (e) 쉼표 뒤 공백 추가 ---
        text = re.sub(r",([가-힣a-zA-Z])", r", \1", text)

        # --- (f) 문장 끝 마침표 추가 ---
        stripped = text.rstrip()
        if stripped and re.match(r"[가-힣a-zA-Z0-9]", stripped[-1]):
            text = stripped + "."

        # 변경 사항 기록
        if text != original:
            changes.append({
                "type": "punctuation",
                "subtype": "fix",
                "before": original[-50:] if len(original) > 50 else original,
                "after": text[-50:] if len(text) > 50 else text,
                "detail": "구두점 교정",
            })

        return text, changes

    # ──────────────────────────────────────────
    # 5. 공백 정리
    # ──────────────────────────────────────────

    def clean_whitespace(self, text: str) -> str:
        """불필요한 공백 정리.

        - 연속 공백을 단일 공백으로 축소
        - 줄바꿈 전후 공백 정리
        - 앞뒤 공백 제거
        """
        # 연속 공백 → 단일 공백
        text = re.sub(r"[ \t]+", " ", text)
        # 줄바꿈 전후 공백 정리
        text = re.sub(r" *\n *", "\n", text)
        # 연속 줄바꿈 축소
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ──────────────────────────────────────────
    # 6. 세그먼트 유효성 검사
    # ──────────────────────────────────────────

    def is_valid_segment(self, text: str) -> bool:
        """세그먼트가 유효한 콘텐츠를 포함하는지 판별.

        무효 판정 기준:
          - 빈 문자열
          - 최소 길이 미만
          - 구두점/공백/특수문자만으로 구성
          - 음악/소리 마커만으로 구성
        """
        if not text:
            return False

        cleaned = text.strip()
        if len(cleaned) < self.config.min_segment_length:
            return False

        # 구두점, 공백, 특수문자만으로 구성된 경우
        if re.match(r"^[\s.,!?;:\-–—…·♪♫🎵🎶\[\]\(\)\*\"\']+$", cleaned):
            return False

        # [filler] 마커만으로 구성된 경우
        remaining = re.sub(r"\[filler\]", "", cleaned).strip()
        if not remaining:
            return False

        return True

    # ──────────────────────────────────────────
    # 내부 유틸리티
    # ──────────────────────────────────────────

    def _deduplicate_segments(self, segments: list[dict]) -> list[dict]:
        """연속 중복 세그먼트 제거."""
        if not segments:
            return segments
        deduped = [segments[0]]
        for seg in segments[1:]:
            if seg.get("text", "").strip() != deduped[-1].get("text", "").strip():
                deduped.append(seg)
        return deduped

    def _apply_medical_correction(self, text: str) -> tuple[str, int]:
        """의료 용어 교정 적용 (medterm 엔진 우선, 레거시 폴백).

        Returns:
            (교정된 텍스트, 교정 횟수)
        """
        correction_count = 0

        try:
            from app.medterm.engine import get_engine
            engine = get_engine()
            if engine is not None:
                result = engine.correct(text)
                correction_count = len(result.logs) if hasattr(result, "logs") else 0
                return result.text, correction_count
        except Exception:
            logger.debug("medterm 엔진 사용 불가, 레거시 폴백 적용")

        # 레거시 폴백
        for wrong, correct in _LEGACY_MEDICAL_DICT:
            if wrong in text:
                text = text.replace(wrong, correct)
                correction_count += 1

        return text, correction_count


# ──────────────────────────────────────────────
# 편의 함수 (기존 인터페이스 호환)
# ──────────────────────────────────────────────

# 모듈 레벨 기본 프로세서 (지연 초기화)
_default_processor: Optional[TextPostProcessor] = None


def _get_default_processor() -> TextPostProcessor:
    """기본 프로세서 인스턴스 반환 (싱글톤)."""
    global _default_processor
    if _default_processor is None:
        _default_processor = TextPostProcessor()
    return _default_processor


def postprocess_text(text: str) -> str:
    """기존 인터페이스 호환 함수: 텍스트 후처리 후 결과 문자열 반환.

    기존 코드에서 postprocess_text()를 호출하던 부분이 변경 없이 동작하도록 유지.
    """
    processor = _get_default_processor()
    result = processor.process(text)
    return result.processed


def deduplicate_segments(segments: list) -> list:
    """기존 인터페이스 호환 함수: 연속 중복 세그먼트 제거."""
    processor = _get_default_processor()
    return processor._deduplicate_segments(segments)


def postprocess_segments(segments: list[dict]) -> list[dict]:
    """세그먼트 목록 후처리 편의 함수."""
    processor = _get_default_processor()
    return processor.process_segments(segments)


def postprocess_with_details(text: str) -> PostProcessResult:
    """후처리 결과를 상세 정보와 함께 반환하는 편의 함수."""
    processor = _get_default_processor()
    return processor.process(text)
