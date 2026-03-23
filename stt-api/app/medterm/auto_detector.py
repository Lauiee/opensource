"""Tier 2: 미등록 의료 용어 오류 자동 탐지 — 캐시 및 통계 지원.

v2: 보호 단어(Safe Words) + 조사 인식으로 거짓 양성(false positive) 대폭 감소.
"""

import logging
import re
from collections import OrderedDict

from app.medterm.models import DictEntry, MatchStrategy
from app.medterm.reference_db import ReferenceDB
from app.medterm.specialty_detector import detect_specialty
from app.medterm.store import DictionaryStore

logger = logging.getLogger(__name__)

# 한글 단어 추출 패턴 (2글자 이상)
_WORD_PATTERN = re.compile(r"[가-힣]{2,}")

# ──────────────────────────────────────────────
# 보호 단어 목록 (Safe Words)
# 이 단어들은 의료 용어와 자모 유사도가 높지만,
# 일상적으로 흔히 사용되므로 교정 대상에서 제외.
# ──────────────────────────────────────────────
SAFE_WORDS: set[str] = {
    # 일상 고빈도 단어 (의료 용어와 혼동 위험)
    "영상", "사진", "화면", "소리", "시간", "기간", "방법", "방향",
    "상태", "상황", "관계", "관련", "부분", "부위", "기능", "기본",
    "정도", "정상", "주변", "주위", "내용", "내부", "외부", "전체",
    "현재", "이전", "이후", "이상", "이하", "결과", "검사", "치료",
    "수술", "설명", "운동", "생활", "식사", "음식", "감사", "인사",
    "남성", "여성", "나이", "연세", "비용", "자기", "본인", "우리",
    "거기", "여기", "저기", "어디", "언제", "어떻게", "다음", "처음",
    "병원", "선생님", "환자", "보호자", "가족", "아버지", "어머니",
    "아들", "딸", "남편", "아내", "형제", "자매",
    # 동사/형용사 어간
    "하고", "하면", "하는", "하는데", "한다", "했어", "했는데",
    "같은", "같이", "있는", "없는", "되는", "많은", "좋은",
    "이런", "그런", "저런", "아직", "지금", "오늘", "내일",
    # 의료 용어와 자모 유사도 높은 일상 단어
    "영상",      # ↔ 열상 (laceration)
    "상담",      # ↔ 담낭
    "비타민",    # 단독으로 충분히 쓰임
    "환경",      # ↔ 환경 (환경안전 등 일상 복합어)
    "안전",      # 일상 단어
    "환경안전",  # 일상 복합어
    "자기공명영상",  # MRI — 이 자체가 정상 의학 용어
    "공명",      # 일상/물리 용어
    "보험",      # 일상 단어
    "급여",      # 일상 단어
    "비급여",    # 일상 단어
    "서류",      # 일상 단어
    "진료",      # 진료 자체는 정상 용어
    "예약",      # 일상 단어
    "접수",      # 일상 단어
    "대기",      # 일상 단어
    "처방",      # 처방 자체가 정상 용어
    "입원",      # 정상 용어
    "퇴원",      # 정상 용어
    "외래",      # 정상 용어
    "재활",      # 정상 용어
    "통증",      # 정상 용어
    "증상",      # 정상 용어
    "약물",      # 정상 용어
    "복용",      # 정상 용어
    "경과",      # 정상 용어
    "검진",      # 정상 용어
    "진단",      # 정상 용어
    # 신체 부위 (조사 붙여도 오교정 안 되게)
    "골반", "무릎", "허리", "어깨", "발목", "손목", "팔꿈치",
    "척추", "등뼈", "갈비뼈", "목뼈", "허벅지", "종아리",
    "엉덩이", "고관절", "손가락", "발가락", "가슴", "배",
    "머리", "이마", "턱", "코", "귀", "눈", "입", "목",
    "팔", "다리", "손", "발", "등", "배꼽",
    # 흔한 의학 용어 (그 자체가 정상)
    "골절", "연골", "인대", "관절", "근육", "힘줄", "뼈",
    "혈관", "신경", "세포", "조직", "장기", "피부", "점막",
    "혈액", "소변", "대변", "호흡", "맥박", "혈압",
    "수액", "주사", "약", "캡슐", "알약", "연고",
    "수술", "시술", "마취", "봉합", "절개", "절제",
    "방사선", "초음파", "내시경", "심전도",
    # v2.0 Tier2 오교정 방지 (일상 고빈도 단어)
    "수도", "이건", "일이", "정신", "정신이",
    "수술하면", "수술하고", "수술을", "수술이",
    "무기폐", "무기폐나",  # 이미 정확한 의료 용어
    "합병증", "합병증이", "합병증에",
    "재발", "재발률", "재발이",
    "혈전", "혈전이",
    "소변줄", "소변",
    "스텐트", "스텐트를",
    "마취", "전신마취",
    "종양", "종양이",
    "외래", "외래로", "외래를",
    "추가적", "추가적인",
    "불가피한", "불가피",
    "서명", "서명해",
    "퇴원", "퇴원할",
    # 일반 조동사/부사 (의료용어로 오변환 방지)
    "수도", "것도", "거고", "거예요", "되고",
    "하게", "될", "해야", "아직", "당연히",
    "보통", "원래", "계속", "굉장히", "정기적",
    "너무", "많이", "좀", "또", "그냥",
    "제가", "본인", "직접",
}

# ──────────────────────────────────────────────
# 한국어 조사 목록 (suffix particles)
# "하지불안증후군도" → "하지불안증후군" + "도" (조사)
# 조사만 다른 단어를 오교정하면 안 됨.
# ──────────────────────────────────────────────
_PARTICLES = [
    "에서부터", "에서는", "에서도", "에서의",
    "으로는", "으로도", "으로서", "으로의",
    "에게는", "에게도", "에게서",
    "까지는", "까지도", "부터는",
    "이라는", "이라고", "이라면",
    "으로", "에서", "에게", "까지", "부터",
    "처럼", "같이", "보다", "에는", "에도",
    "이라", "이나", "이든", "이고",
    "인데", "이면", "이랑",
    "과는", "와는",
    "에", "을", "를", "은", "는", "이", "가",
    "도", "와", "과", "의", "로", "만",
]


def _strip_particle(word: str) -> tuple[str, str]:
    """단어에서 조사를 분리. (어근, 조사) 반환.

    예: "광장공포증이" → ("광장공포증", "이")
        "골반에서" → ("골반", "에서")
        "통증" → ("통증", "")
    """
    for p in _PARTICLES:
        if word.endswith(p) and len(word) > len(p) + 1:
            return word[:-len(p)], p
    return word, ""


def _is_particle_only_diff(word: str, candidate_term: str) -> bool:
    """두 단어의 차이가 조사/접미어뿐인지 확인.

    예: "광장공포증이" vs "광장공포증" → True (조사 "이" 제거)
        "골반이" vs "골반위" → True (어근 "골반" 동일, 1글자 차이)
        "하지불안증후군도" vs "하지불안증후군" → True (조사 "도")
        "백래장" vs "백내장" → False (실제 오타)
    """
    stem_word, particle_word = _strip_particle(word)
    stem_cand, particle_cand = _strip_particle(candidate_term)

    # 어근이 같으면 조사만 다른 것
    if stem_word == stem_cand:
        return True
    if stem_word == candidate_term:
        return True
    if word == stem_cand:
        return True

    # ★ 어근이 후보의 접두사이고, 차이가 1글자면 조사 유사 차이
    #    예: "골반이" → stem "골반", candidate "골반위" → "골반"이 접두사, 차이 1글자
    if particle_word and candidate_term.startswith(stem_word):
        suffix_diff = len(candidate_term) - len(stem_word)
        if suffix_diff <= 1:
            return True

    # ★ 전체 단어가 후보의 접두사+1이면 (조사 포함)
    #    예: "골반이"(3자) vs "골반위"(3자) → 앞 2글자 같고 끝 1자만 다름
    if len(word) == len(candidate_term) and len(word) >= 3:
        common_prefix = 0
        for a, b in zip(word, candidate_term):
            if a == b:
                common_prefix += 1
            else:
                break
        # 공통 접두사가 단어 길이-1이면 마지막 글자만 다름
        # 그리고 마지막 글자가 조사면 → 조사 차이
        if common_prefix == len(word) - 1:
            last_char = word[-1]
            # 단일 조사인지 체크
            if last_char in "이가를을은는에도와과의로만":
                return True

    return False

# 탐지 캐시 기본 크기
_DETECTION_CACHE_SIZE = 512


class DetectionResult:
    """탐지 결과."""

    def __init__(self, word: str, position: int, candidates: list[dict], action: str):
        self.word = word
        self.position = position
        self.candidates = candidates  # [{"term": str, "similarity": float, "specialty": str}]
        self.action = action  # "auto_correct" | "needs_review" | "ignore"

    def best_candidate(self) -> dict | None:
        """최고 유사도 후보 반환."""
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        """직렬화용 딕셔너리 변환."""
        best = self.best_candidate()
        return {
            "word": self.word,
            "position": self.position,
            "action": self.action,
            "candidates": self.candidates,
            "best_term": best["term"] if best else None,
            "best_similarity": best["similarity"] if best else None,
        }


class _DetectionCache:
    """탐지 결과 캐시 — 동일 단어 반복 탐지 방지."""

    def __init__(self, maxsize: int = _DETECTION_CACHE_SIZE):
        self._cache: OrderedDict[str, list[dict] | None] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, word: str, specialty: str | None) -> tuple[bool, list[dict] | None]:
        """캐시에서 탐지 결과 조회.

        Returns:
            (found, candidates) — found=True면 캐시 히트
        """
        key = f"{word}|{specialty or 'ALL'}"
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return True, self._cache[key]
        self._misses += 1
        return False, None

    def put(self, word: str, specialty: str | None, candidates: list[dict] | None) -> None:
        """캐시에 탐지 결과 저장."""
        key = f"{word}|{specialty or 'ALL'}"
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = candidates
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = candidates

    def invalidate(self) -> None:
        """캐시 전체 무효화 (사전 변경 시 호출)."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


class AutoDetector:
    """미등록 의료 용어 오류를 자동 탐지 — 캐시 + 통계 지원."""

    def __init__(
        self,
        ref_db: ReferenceDB,
        store: DictionaryStore,
        auto_threshold: float = 0.85,
        review_threshold: float = 0.70,
        cache_size: int = _DETECTION_CACHE_SIZE,
    ):
        self._ref_db = ref_db
        self._store = store
        self._auto_threshold = auto_threshold
        self._review_threshold = review_threshold

        # Tier 1 사전의 wrong 값들 캐시 (이미 교정 대상인 것 스킵)
        self._known_wrongs: set[str] = set()
        self._known_corrects: set[str] = set()
        self._refresh_known()

        # 탐지 결과 캐시
        self._cache = _DetectionCache(maxsize=cache_size)

        # 통계 추적
        self._stats = {
            "total_detections": 0,       # 총 탐지 호출 수
            "total_words_checked": 0,    # 총 검사 단어 수
            "auto_corrections": 0,       # 자동 교정 결과 수
            "needs_review": 0,           # 검증 대기 결과 수
            "ignored": 0,                # 무시된 결과 수
            "skipped_known": 0,          # Tier 1 이미 처리된 단어 스킵 수
            "skipped_exact": 0,          # 참조 DB에 정확히 존재하여 스킵된 수
            "skipped_safe": 0,           # 보호 단어로 스킵된 수
            "skipped_particle": 0,       # 조사만 다른 단어 스킵 수
        }

    def _refresh_known(self) -> None:
        """Tier 1 사전의 wrong/correct 값 목록 갱신."""
        entries = self._store.get_entries(enabled_only=True)
        self._known_wrongs = {e.wrong for e in entries}
        self._known_corrects = {e.correct for e in entries}

    def invalidate_cache(self) -> None:
        """캐시 무효화 (사전 변경 후 호출)."""
        self._cache.invalidate()
        self._refresh_known()

    def detect(
        self,
        text: str,
        context_hint: str | None = None,
    ) -> list[DetectionResult]:
        """텍스트에서 미등록 오류 탐지.

        Args:
            text: 분석할 텍스트
            context_hint: 진료과 힌트 (예: "정형외과", "내과")

        절차:
        1. 한글 단어 추출
        2. 보호 단어(Safe Words)이면 skip
        3. Tier 1 사전에 있으면 skip
        4. KOSTOM 참조 DB에 정확히 있으면 skip
        5. 조사만 다른 경우 skip
        6. KOSTOM에 유사한 용어가 있으면 → 임계값에 따라 분류
        """
        self._stats["total_detections"] += 1
        self._refresh_known()

        # 진료과 추정 (context_hint 우선, 없으면 텍스트에서 자동 추정)
        primary_specialty = context_hint
        if not primary_specialty:
            specialties = detect_specialty(text, self._ref_db)
            primary_specialty = specialties[0][0] if specialties else None

        results: list[DetectionResult] = []
        seen_words: set[str] = set()

        for match in _WORD_PATTERN.finditer(text):
            word = match.group()
            pos = match.start()

            # 중복 스킵
            if word in seen_words:
                continue
            seen_words.add(word)
            self._stats["total_words_checked"] += 1

            # ★ 보호 단어 스킵 — 일상 한국어 단어를 교정하지 않음
            if word in SAFE_WORDS:
                self._stats["skipped_safe"] += 1
                continue

            # ★ 조사를 제거한 어근이 보호 단어이면 스킵
            stem, particle = _strip_particle(word)
            if particle and stem in SAFE_WORDS:
                self._stats["skipped_safe"] += 1
                continue

            # Tier 1에서 이미 처리되는 단어 스킵
            if word in self._known_wrongs or word in self._known_corrects:
                self._stats["skipped_known"] += 1
                continue

            # 참조 DB에 정확히 존재하면 정상 → 스킵
            if self._ref_db.has_exact(word):
                self._stats["skipped_exact"] += 1
                continue

            # ★ 조사 제거 후 어근이 참조 DB에 있으면 정상 → 스킵
            if particle and self._ref_db.has_exact(stem):
                self._stats["skipped_exact"] += 1
                continue

            # 캐시 확인
            found, cached_candidates = self._cache.get(word, primary_specialty)
            if found:
                candidates = cached_candidates
            else:
                # 참조 DB에서 유사 용어 검색 (진료과 우선)
                candidates = self._ref_db.search(
                    word, specialty=primary_specialty, top_n=3
                )
                if not candidates:
                    # 전체 진료과에서 다시 검색
                    candidates = self._ref_db.search(word, specialty=None, top_n=3)

                # 캐시에 저장 (빈 결과도 저장하여 불필요한 재검색 방지)
                self._cache.put(word, primary_specialty, candidates if candidates else None)

            if not candidates:
                continue

            best = candidates[0]
            best_sim = best["similarity"]

            # 정확히 같은 단어가 후보에 있으면 스킵
            if any(c["term"] == word for c in candidates):
                continue

            # ★ 조사만 다른 경우 스킵 — "광장공포증이"→"광장공포증" 방지
            if _is_particle_only_diff(word, best["term"]):
                self._stats["skipped_particle"] += 1
                continue

            # ★ 후보 용어가 보호 단어이면 스킵 (역방향 보호)
            if best["term"] in SAFE_WORDS:
                self._stats["skipped_safe"] += 1
                continue

            # 임계값에 따라 액션 결정
            if best_sim >= self._auto_threshold:
                action = "auto_correct"
                self._stats["auto_corrections"] += 1
            elif best_sim >= self._review_threshold:
                action = "needs_review"
                self._stats["needs_review"] += 1
            else:
                action = "ignore"
                self._stats["ignored"] += 1

            if action != "ignore":
                results.append(DetectionResult(
                    word=word,
                    position=pos,
                    candidates=candidates,
                    action=action,
                ))

        return results

    def get_stats(self) -> dict:
        """탐지 통계 반환."""
        return {
            **self._stats,
            "cache_size": self._cache.size,
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
            "auto_threshold": self._auto_threshold,
            "review_threshold": self._review_threshold,
            "known_wrongs_count": len(self._known_wrongs),
            "known_corrects_count": len(self._known_corrects),
        }
