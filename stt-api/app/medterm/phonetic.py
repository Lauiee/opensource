"""한글 자모 분해 및 음성학적 유사도 계산."""

# 초성 19자
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
# 중성 21자
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
# 종성 28자 (0번은 종성 없음)
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_NUM_JUNGSEONG = 21
_NUM_JONGSEONG = 28


def is_hangul(ch: str) -> bool:
    """한글 음절인지 확인."""
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_END


def decompose_char(ch: str) -> list[str]:
    """한글 한 글자를 자모로 분해. 한글이 아니면 그대로 반환."""
    if not is_hangul(ch):
        return [ch]
    code = ord(ch) - _HANGUL_BASE
    cho = code // (_NUM_JUNGSEONG * _NUM_JONGSEONG)
    jung = (code % (_NUM_JUNGSEONG * _NUM_JONGSEONG)) // _NUM_JONGSEONG
    jong = code % _NUM_JONGSEONG
    result = [CHOSEONG[cho], JUNGSEONG[jung]]
    if jong > 0:
        result.append(JONGSEONG[jong])
    return result


def decompose(text: str) -> list[str]:
    """문자열 전체를 자모 시퀀스로 분해."""
    jamo: list[str] = []
    for ch in text:
        jamo.extend(decompose_char(ch))
    return jamo


def _levenshtein(s1: list[str], s2: list[str]) -> int:
    """자모 시퀀스 간 Levenshtein 편집 거리."""
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 삭제
                curr[j - 1] + 1,   # 삽입
                prev[j - 1] + cost, # 치환
            )
        prev, curr = curr, prev

    return prev[m]


def jamo_similarity(text1: str, text2: str) -> float:
    """두 한글 문자열의 자모 유사도 (0.0 ~ 1.0)."""
    jamo1 = decompose(text1)
    jamo2 = decompose(text2)
    max_len = max(len(jamo1), len(jamo2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein(jamo1, jamo2)
    return 1.0 - (distance / max_len)
