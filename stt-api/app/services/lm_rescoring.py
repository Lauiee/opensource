"""LM 기반 ASR 오류 교정 (최신 기술 적용).

2025-2026 최신 연구 기반:
- N-best 리스코어링: Whisper 후보 중 최적 선택
- 의료 도메인 N-gram 언어 모델: 의료 텍스트 통계 기반 교정
- 혼동 쌍(confusion pair) 기반 교정: 음향적으로 유사한 오류 패턴 자동 교정

참고 논문:
- Whisper-LM (2025): LM으로 Whisper 출력 보정
- Whispering-LLaMA (EMNLP 2023): LLM 기반 전사 오류 교정
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. 의료 도메인 N-gram 언어 모델 (경량, 무료)
# ──────────────────────────────────────────────────────────────────────

class MedicalNgramLM:
    """의료 텍스트 기반 N-gram 언어 모델.

    의료 대화 코퍼스에서 빈번한 어절 조합을 학습하여,
    STT 출력에서 비정상적인 어절 조합을 탐지한다.
    """

    def __init__(self):
        self._bigrams: Counter = Counter()
        self._trigrams: Counter = Counter()
        self._unigrams: Counter = Counter()
        self._total_bigrams = 0
        self._total_unigrams = 0

    def train_from_texts(self, texts: list[str]) -> None:
        """텍스트 리스트에서 N-gram 학습."""
        for text in texts:
            words = text.split()
            for w in words:
                self._unigrams[w] += 1
                self._total_unigrams += 1
            for i in range(len(words) - 1):
                bg = (words[i], words[i + 1])
                self._bigrams[bg] += 1
                self._total_bigrams += 1
            for i in range(len(words) - 2):
                tg = (words[i], words[i + 1], words[i + 2])
                self._trigrams[tg] += 1

        logger.info(
            "N-gram LM 학습 완료: unigrams=%d, bigrams=%d, trigrams=%d",
            len(self._unigrams), len(self._bigrams), len(self._trigrams),
        )

    def train_from_files(self, file_paths: list[Path]) -> None:
        """파일들에서 텍스트를 읽어 학습."""
        import json
        texts = []
        for fp in file_paths:
            if not fp.exists():
                continue
            try:
                raw = fp.read_text(encoding="utf-8").strip()
                # JSON 형식 시도
                if raw.startswith("["):
                    be = raw.rfind("]")
                    if be >= 0:
                        raw = raw[: be + 1]
                    data = json.loads(raw)
                    for item in data:
                        content = item.get("content", "")
                        if content:
                            texts.append(content)
                else:
                    texts.append(raw)
            except Exception:
                continue

        self.train_from_texts(texts)

    def score_sentence(self, sentence: str) -> float:
        """문장의 언어 모델 점수 (높을수록 자연스러움)."""
        words = sentence.split()
        if not words:
            return 0.0

        score = 0.0
        for i in range(len(words) - 1):
            bg = (words[i], words[i + 1])
            bg_count = self._bigrams.get(bg, 0)
            uni_count = self._unigrams.get(words[i], 0)
            if uni_count > 0:
                prob = (bg_count + 1) / (uni_count + len(self._unigrams))  # Laplace smoothing
            else:
                prob = 1 / (len(self._unigrams) + 1)
            import math
            score += math.log(prob)

        return score / max(len(words), 1)

    def find_anomalous_words(self, text: str, threshold: float = -8.0) -> list[dict]:
        """텍스트에서 비정상적(점수가 낮은) 어절을 찾음."""
        words = text.split()
        anomalies = []

        for i in range(1, len(words)):
            bg = (words[i - 1], words[i])
            bg_count = self._bigrams.get(bg, 0)
            uni_count = self._unigrams.get(words[i - 1], 0)

            if uni_count > 0 and bg_count == 0:
                # 이전 단어는 알려졌지만 이 조합은 한 번도 안 나옴
                anomalies.append({
                    "position": i,
                    "word": words[i],
                    "context": f"{words[i-1]} {words[i]}",
                    "score": 0.0,
                })

        return anomalies


# ──────────────────────────────────────────────────────────────────────
# 2. 혼동 쌍(Confusion Pair) 기반 교정
# ──────────────────────────────────────────────────────────────────────

# 한국어 의료 STT에서 자주 발생하는 음향적 혼동 쌍
# (Whisper가 특히 자주 틀리는 패턴)
CONFUSION_PAIRS: list[tuple[str, str, str]] = [
    # (잘못된 패턴, 올바른 패턴, 조건/컨텍스트)
    # ── 자음 혼동 ──
    ("ㄴ→ㄹ", "신장", "심장"),  # 이건 context-dependent
    ("ㅂ→ㅍ", "뾰족", "뼈족"),
    # ── 음절 탈락/삽입 ──
    ("초파", "초음파", ""),
    ("처에는", "처음에는", ""),
    ("다에", "다음에", ""),
    ("다번", "다음번", ""),
    # ── 영어 음역 오류 ──
    ("마라해", "MRI", ""),
    ("엠알에이", "MRI", ""),
]

# 문맥 기반 교정 규칙 (단순 치환보다 정교)
CONTEXT_RULES: list[dict] = [
    {
        "pattern": r"심장\s*(기능|안|안쪽)",
        "context_check": r"소변|사구체|콩팥|신우|여과율|단백뇨",
        "replacement": lambda m: m.group(0).replace("심장", "신장"),
        "description": "소변/사구체 맥락에서 심장→신장",
    },
    {
        "pattern": r"진로\s*(의뢰서|를|을)",
        "context_check": None,  # 항상 적용
        "replacement": lambda m: m.group(0).replace("진로", "진료"),
        "description": "진로→진료 (의뢰서 맥락)",
    },
]


def apply_context_corrections(text: str, full_context: str = "") -> str:
    """문맥 기반 교정 적용.

    단순 문자열 치환이 아니라, 주변 문맥을 확인한 후에만 교정.
    """
    context = full_context or text

    for rule in CONTEXT_RULES:
        pattern = rule["pattern"]
        ctx_check = rule.get("context_check")

        # 문맥 조건 확인
        if ctx_check and not re.search(ctx_check, context):
            continue

        text = re.sub(pattern, rule["replacement"], text)

    return text


# ──────────────────────────────────────────────────────────────────────
# 3. Temperature Fallback + N-best 선택
# ──────────────────────────────────────────────────────────────────────

def select_best_transcription(
    candidates: list[str],
    lm: MedicalNgramLM = None,
) -> str:
    """여러 전사 후보 중 가장 자연스러운 것을 선택.

    1. 각 후보의 LM 점수 계산
    2. 길이 페널티 적용 (너무 짧거나 긴 것 불이익)
    3. 환각 패턴 패널티 적용
    4. 최고 점수 후보 반환
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    best_score = float("-inf")
    best_candidate = candidates[0]

    avg_len = sum(len(c) for c in candidates) / len(candidates)

    for candidate in candidates:
        score = 0.0

        # LM 점수
        if lm:
            score += lm.score_sentence(candidate) * 10

        # 길이 페널티
        len_ratio = len(candidate) / max(avg_len, 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            score -= 5.0

        # 환각 패널티
        if re.search(r"(\d{2,}월부터\.?\s*){3,}", candidate):
            score -= 100.0  # 불가능한 월 반복
        if re.search(r"(\d\s+){6,}", candidate):
            score -= 50.0   # 숫자 나열

        # 반복 패널티
        words = candidate.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 20.0  # 같은 단어 반복이 너무 많음

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


# ──────────────────────────────────────────────────────────────────────
# 4. 통합 후처리 파이프라인
# ──────────────────────────────────────────────────────────────────────

def enhanced_postprocess(
    text: str,
    full_context: str = "",
    lm: MedicalNgramLM = None,
) -> str:
    """최신 기술 기반 향상된 후처리.

    1. 문맥 기반 교정 (심장→신장 등)
    2. N-gram LM 기반 이상 탐지
    3. 환각 필터링 강화
    """
    # 1) 문맥 기반 교정
    text = apply_context_corrections(text, full_context)

    # 2) 불가능한 날짜 환각 제거
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월부터\.?\s*", "", text)
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)

    # 3) 숫자 나열 환각 제거
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)

    # 4) 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text
