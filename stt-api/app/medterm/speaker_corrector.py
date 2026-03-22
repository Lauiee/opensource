"""화자분리(Speaker Diarization) 후처리 교정 모듈.

2단계 파이프라인:
  1단계 (AB): 호칭 + 내용 패턴 → 즉시 교정 (0ms)
  2단계 (GPT): 전체 대화 문맥 기반 GPT-4o-mini 검증 → 정밀 교정

GPT 호출 기준 (needs_gpt_review):
  - 역할 비율 불균형 (한쪽 >70%)
  - 단일 화자 (전부 같은 role)
  - 연속 동일 화자 3회 이상 구간 존재
  - AB 신호 없는 세그먼트가 50% 이상

추가 분석 엔진:
  - Context Window Analysis: 주변 세그먼트 기반 문맥 분석
  - Conversation Flow Analysis: 진료 단계별 화자 패턴 분석
  - 다중 신호 가중 합산 신뢰도 스코어링
"""

from __future__ import annotations

import json
import re
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 신뢰도 등급 상수
# ─────────────────────────────────────────────

CONFIDENCE_HIGH = 0.8      # 높은 신뢰도 임계값
CONFIDENCE_MEDIUM = 0.5    # 중간 신뢰도 임계값
CONFIDENCE_LOW = 0.0       # 낮은 신뢰도 (0.5 미만)

# 교정 적용 최소 신뢰도 임계값
CORRECTION_THRESHOLD = 0.3


class Role(str, Enum):
    DOCTOR = "원장님"
    PATIENT = "환자"


class ConfidenceLevel(str, Enum):
    """신뢰도 등급."""
    HIGH = "HIGH"       # ≥ 0.8
    MEDIUM = "MEDIUM"   # 0.5 ~ 0.8
    LOW = "LOW"         # < 0.5


def get_confidence_level(confidence: float) -> ConfidenceLevel:
    """신뢰도 수치를 등급으로 변환."""
    if confidence >= CONFIDENCE_HIGH:
        return ConfidenceLevel.HIGH
    elif confidence >= CONFIDENCE_MEDIUM:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# ─────────────────────────────────────────────
# 진료 대화 단계 정의
# ─────────────────────────────────────────────

class ConversationPhase(str, Enum):
    """진료 대화의 단계별 구분."""
    GREETING = "greeting"               # 인사
    CHIEF_COMPLAINT = "chief_complaint"  # 주호소 (왜 왔는지)
    HISTORY = "history"                 # 병력 청취
    EXAMINATION = "examination"         # 검사/진찰
    DIAGNOSIS = "diagnosis"             # 진단 설명
    TREATMENT = "treatment"             # 치료/처방 설명
    FAREWELL = "farewell"               # 마무리 인사


# 각 단계에서 기대되는 화자 패턴 (의사 발화 비율)
# 0.5 = 반반, 0.7 = 의사가 70%, 0.3 = 환자가 70%
PHASE_EXPECTED_DOCTOR_RATIO = {
    ConversationPhase.GREETING: 0.5,           # 인사: 반반
    ConversationPhase.CHIEF_COMPLAINT: 0.3,    # 주호소: 환자가 주로 말함
    ConversationPhase.HISTORY: 0.4,            # 병력: 의사 질문 + 환자 대답
    ConversationPhase.EXAMINATION: 0.6,        # 검사: 의사가 주도
    ConversationPhase.DIAGNOSIS: 0.8,          # 진단: 의사가 주로 설명
    ConversationPhase.TREATMENT: 0.7,          # 치료: 의사가 주로 설명
    ConversationPhase.FAREWELL: 0.5,           # 마무리: 반반
}

# 단계 판별용 키워드 (컴파일된 정규식은 아래에서 별도 처리)
PHASE_KEYWORDS = {
    ConversationPhase.GREETING: [
        "안녕하세요", "반갑습니다", "어서 오세요", "들어오세요", "오셨어요",
    ],
    ConversationPhase.CHIEF_COMPLAINT: [
        "어디가 아프", "어디가 불편", "오늘 어떻게", "뭐 때문에", "무엇 때문에",
        "어떻게 오셨", "어디 불편", "증상이 뭐",
    ],
    ConversationPhase.HISTORY: [
        "언제부터", "전에도", "예전에", "과거에", "수술 받은", "병력",
        "가족 중에", "약 드시는", "복용하는 약", "알레르기",
    ],
    ConversationPhase.EXAMINATION: [
        "검사를", "찍어보", "확인해", "체크해", "측정", "혈압",
        "X-ray", "엑스레이", "MRI", "CT", "초음파", "혈액검사", "소변검사",
    ],
    ConversationPhase.DIAGNOSIS: [
        "진단", "소견", "결과가", "결과를", "나왔", "보이고", "판단",
        "의심", "가능성이", "확인됐", "발견됐",
    ],
    ConversationPhase.TREATMENT: [
        "처방", "약을", "주사", "수술", "시술", "물리치료", "치료를",
        "복용", "드시고", "바르시고", "오시면", "다음에 오",
    ],
    ConversationPhase.FAREWELL: [
        "수고하세요", "감사합니다", "안녕히", "조심히", "다음에 뵐",
        "잘 가세요", "건강하세요",
    ],
}


@dataclass
class SpeakerSignal:
    """하나의 세그먼트에서 감지된 화자 신호."""
    doctor_score: float = 0.0
    patient_score: float = 0.0
    signals: list[str] = field(default_factory=list)

    @property
    def predicted_role(self) -> Role | None:
        """예측된 역할. 점수 차이가 임계값 미만이면 None."""
        diff = self.doctor_score - self.patient_score
        if abs(diff) < 0.3:
            return None
        return Role.DOCTOR if diff > 0 else Role.PATIENT

    @property
    def confidence(self) -> float:
        """신뢰도 (0.0 ~ 1.0). 두 점수의 차이 비율."""
        total = self.doctor_score + self.patient_score
        if total == 0:
            return 0.0
        return abs(self.doctor_score - self.patient_score) / total

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """신뢰도 등급."""
        return get_confidence_level(self.confidence)

    @property
    def has_signal(self) -> bool:
        """신호가 하나라도 있는지."""
        return (self.doctor_score + self.patient_score) > 0

    def merge(self, other: SpeakerSignal, weight: float = 1.0) -> None:
        """다른 신호를 가중치와 함께 병합."""
        self.doctor_score += other.doctor_score * weight
        self.patient_score += other.patient_score * weight
        self.signals.extend(other.signals)


@dataclass
class CorrectionResult:
    """교정 결과."""
    original_role: str
    corrected_role: str
    changed: bool
    strategy: str
    confidence: float
    signals: list[str]

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """신뢰도 등급."""
        return get_confidence_level(self.confidence)


# ─────────────────────────────────────────────
# 정규식 사전 컴파일 (성능 최적화)
# ─────────────────────────────────────────────

def _compile_patterns(raw_patterns: list[tuple[str, float, str]]) -> list[tuple[re.Pattern, float, str]]:
    """패턴 리스트를 사전 컴파일하여 반환."""
    compiled = []
    for pattern_str, score, desc in raw_patterns:
        compiled.append((re.compile(pattern_str), score, desc))
    return compiled


# ─────────────────────────────────────────────
# 전략 A: 호칭 기반 교정 (확장)
# ─────────────────────────────────────────────

# 의사가 환자/보호자를 부를 때 사용하는 호칭 → 의사 발화
_RAW_HONORIFIC_PATTERNS_DOCTOR = [
    # 환자 호칭
    (r"환자분", 1.5, "호칭: '환자분' → 의사"),
    # 가족 호칭 (의사가 환자/보호자에게)
    (r"어머님", 1.2, "호칭: '어머님' → 의사"),
    (r"어머니", 1.2, "호칭: '어머니' → 의사"),
    (r"아버지", 1.2, "호칭: '아버지' → 의사"),
    (r"아버님", 1.2, "호칭: '아버님' → 의사"),
    (r"할머니", 1.2, "호칭: '할머니' → 의사"),
    (r"할아버지", 1.2, "호칭: '할아버지' → 의사"),
    (r"할머님", 1.2, "호칭: '할머님' → 의사"),
    (r"할아버님", 1.2, "호칭: '할아버님' → 의사"),
    # 보호자 호칭
    (r"보호자분", 1.3, "호칭: '보호자분' → 의사"),
    (r"보호자", 1.0, "호칭: '보호자' → 의사"),
    # 어르신 호칭
    (r"어르신", 1.2, "호칭: '어르신' → 의사"),
    # 의사 특유의 인사/안내 표현
    (r"\b뵐게요\b", 0.8, "호칭: '뵐게요' → 의사 (예약)"),
    (r"들어오세요", 0.8, "호칭: '들어오세요' → 의사"),
    (r"안녕히\s*가세요", 0.8, "호칭: '안녕히 가세요' → 의사"),
    (r"조심히\s*가세요", 0.8, "호칭: '조심히 가세요' → 의사"),
    (r"어서\s*오세요", 0.7, "호칭: '어서 오세요' → 의사"),
]

# 환자가 의사를 부를 때 사용하는 호칭 → 환자 발화
_RAW_HONORIFIC_PATTERNS_PATIENT = [
    # 의사 호칭
    (r"^선생님", 1.5, "호칭: 문두 '선생님' → 환자"),
    (r"선생님[,\s]", 1.0, "호칭: '선생님,' → 환자"),
    (r"원장님", 1.5, "호칭: '원장님' → 환자"),
    (r"의사\s*선생님", 1.5, "호칭: '의사선생님' → 환자"),
    (r"교수님", 1.5, "호칭: '교수님' → 환자"),
    (r"과장님", 1.3, "호칭: '과장님' → 환자"),
    # 환자 특유의 인사/마무리 표현
    (r"수고하세요", 0.9, "호칭: '수고하세요' → 환자"),
    (r"수고하셨", 0.9, "호칭: '수고하셨' → 환자"),
    (r"잘\s*부탁", 0.8, "호칭: '잘 부탁' → 환자"),
    (r"잘\s*부탁드", 0.9, "호칭: '잘 부탁드립니다' → 환자"),
    (r"감사합니다\s*선생님", 1.2, "호칭: '감사합니다 선생님' → 환자"),
]

# 사전 컴파일
HONORIFIC_PATTERNS_DOCTOR = _compile_patterns(_RAW_HONORIFIC_PATTERNS_DOCTOR)
HONORIFIC_PATTERNS_PATIENT = _compile_patterns(_RAW_HONORIFIC_PATTERNS_PATIENT)


def strategy_a_honorific(content: str) -> SpeakerSignal:
    """호칭 기반 화자 판별. 사전 컴파일된 정규식 사용."""
    sig = SpeakerSignal()
    for compiled_re, score, desc in HONORIFIC_PATTERNS_DOCTOR:
        if compiled_re.search(content):
            sig.doctor_score += score
            sig.signals.append(desc)
    for compiled_re, score, desc in HONORIFIC_PATTERNS_PATIENT:
        if compiled_re.search(content):
            sig.patient_score += score
            sig.signals.append(desc)
    return sig


# ─────────────────────────────────────────────
# 전략 B: 발화 내용 분석 (확장)
# ─────────────────────────────────────────────

_RAW_CONTENT_PATTERNS_DOCTOR = [
    # ── 진단/검사 관련 ──
    (r"(진단|소견|검사\s*결과)", 1.2, "내용: 진단/검사결과 설명"),
    (r"MRI\s*(결과|사진|소견|에서|를)", 1.3, "내용: MRI 결과 설명 → 의사"),
    (r"CT\s*(결과|사진|소견|에서|를)", 1.3, "내용: CT 결과 설명 → 의사"),
    (r"(X-ray|엑스레이)\s*(결과|사진|소견|에서|를)", 1.3, "내용: X-ray 결과 설명 → 의사"),
    (r"초음파\s*(결과|소견|에서|를)", 1.2, "내용: 초음파 결과 설명 → 의사"),
    (r"혈액\s*검사\s*(결과|수치|에서)", 1.2, "내용: 혈액검사 결과 설명 → 의사"),
    (r"(수치가|수치는)\s*(높|낮|정상|비정상)", 1.0, "내용: 검사 수치 설명 → 의사"),

    # ── 처방/투약 관련 ──
    (r"(처방|투약|복용)", 1.0, "내용: 처방/투약 지시"),
    (r"약을\s*(드시|드셔|먹|복용|처방)", 1.2, "내용: 약 처방 설명 → 의사"),
    (r"주사\s*(를|한\s*번|맞|놓)", 1.0, "내용: 주사 관련 설명 → 의사"),
    (r"(하루에|하루\s*한\s*번|하루\s*두\s*번|하루\s*세\s*번)\s*(드시|드셔|먹|복용)", 1.0, "내용: 복용법 설명 → 의사"),
    (r"식후에?\s*(드시|드셔|먹|복용)", 0.9, "내용: 복용 시점 안내 → 의사"),

    # ── 수술/시술 관련 ──
    (r"(수술|시술)을?\s*(하|진행|예정)", 1.0, "내용: 수술/시술 설명"),
    (r"(수술|시술)\s*(방법|과정|시간|위험)", 1.0, "내용: 수술 상세 설명 → 의사"),
    (r"물리치료\s*(를|받으시|하시)", 1.0, "내용: 물리치료 안내 → 의사"),
    (r"재활\s*(운동|치료|프로그램)", 0.9, "내용: 재활 안내 → 의사"),

    # ── 치료/회복 관련 ──
    (r"(치료|회복).*(될|됩니다|거예요|거에요|겁니다)", 1.0, "내용: 치료 예후 설명"),
    (r"(예약|다음에|뒤에|뒤로)\s*(해|잡아|뵐)", 0.8, "내용: 예약 안내"),
    (r"(생길\s*수|발생할\s*수|올\s*수).*(있|있어요|있습니다)", 0.7, "내용: 가능성 설명"),
    (r"(합병증|부작용|후유증)", 0.8, "내용: 합병증 설명"),
    (r"(좋아지|괜찮아지|회복|낫)", 0.5, "내용: 회복 전망"),

    # ── ~하셔야, ~드릴게요 어투 (의사 특유) ──
    (r"(해\s*드리|놔\s*드리|해드리|놔드리|드릴게|드리겠)", 1.2, "내용: '~해드리다' 서비스 어투"),
    (r"(하셔야|드셔야|받으셔야|오셔야|쉬셔야)", 1.0, "내용: '~하셔야' 지시 어투 → 의사"),
    (r"(주셔야|해주셔야|해\s*주셔야)", 0.7, "내용: 환자에게 지시"),
    (r"(하시면\s*됩니다|하시면\s*돼요|하시면\s*돼)", 0.9, "내용: '~하시면 됩니다' 안내 어투 → 의사"),

    # ── 환자에게 질문 ──
    (r"(어디가\s*아프|어디\s*아프|어디가\s*불편)", 1.2, "내용: 증상 질문 → 의사"),
    (r"(언제부터|언제\s*부터)\s*(아프|불편|증상|시작)", 1.0, "내용: 시점 질문 → 의사"),
    (r"(통증이|통증은)\s*(어떠|어때|어디|언제)", 1.0, "내용: 통증 질문 → 의사"),
    (r"(어떠신|어떠세요|어떤\s*증상)", 0.8, "내용: 증상 질문 → 의사"),
    (r"(몇\s*번|얼마나\s*자주|자주\s*그러)", 0.7, "내용: 빈도 질문 → 의사"),
    (r"(다른\s*곳은|다른\s*데는|다른\s*증상)", 0.7, "내용: 추가 증상 질문 → 의사"),
    (r"(알레르기|과거\s*병력|수술\s*받으신)", 0.8, "내용: 병력 질문 → 의사"),

    # ── 안심/격려 ──
    (r"(걱정|염려).*(마세요|않으셔도|마시고)", 0.8, "내용: 안심시키기"),
    (r"(크게\s*걱정|많이\s*걱정).*(않으셔도|마세요)", 0.9, "내용: 안심시키기 강조 → 의사"),

    # ── 안내/지시 ──
    (r"(수납|처방전|약국)", 0.8, "내용: 수납/처방 안내"),
    (r"(확인|체크|검사).*(하겠|할게|해볼)", 0.8, "내용: 의사 행동 표현"),
    (r"오시면\s*돼", 0.8, "내용: '오시면 돼요' 지시"),
    (r"넣으시면\s*돼", 0.8, "내용: 약 사용법 설명"),
    (r"(나가.*계세요|앉아.*계세요|기다려)", 0.6, "내용: 대기 지시"),
    (r"준비할게요", 0.7, "내용: 시술 준비"),
    (r"(한\s*번\s*보겠|한번\s*볼게|봐\s*드릴)", 0.9, "내용: 진찰 시작 → 의사"),
    (r"(누워\s*보세요|앉아\s*보세요|일어나\s*보세요)", 0.9, "내용: 진찰 자세 지시 → 의사"),
    (r"(옷을\s*좀|윗옷을|상의를)\s*(올려|벗어|걷어)", 0.8, "내용: 진찰 준비 지시 → 의사"),
]

_RAW_CONTENT_PATTERNS_PATIENT = [
    # ── 증상 호소 ──
    (r"(아파요|아픕니다|아프|아팠)", 1.0, "내용: 증상 호소 (통증)"),
    (r"(쑤셔요|쑤시고|쑤시는)", 1.0, "내용: 증상 호소 (쑤심)"),
    (r"(저리고|저려요|저린|저릿저릿)", 1.0, "내용: 증상 호소 (저림)"),
    (r"(당겨요|당기고|당기는|당김)", 1.0, "내용: 증상 호소 (당김)"),
    (r"(뻣뻣|뻣뻣해|뻑뻑|뻑뻑해)", 1.0, "내용: 증상 호소 (뻣뻣함)"),
    (r"(욱신|욱신거|욱신욱신|지끈|지끈지끈)", 1.0, "내용: 증상 호소 (욱신/지끈)"),
    (r"(결리|결려요|결리고)", 0.9, "내용: 증상 호소 (결림)"),
    (r"(붓고|부어서|부었|부기가)", 0.9, "내용: 증상 호소 (부종)"),
    (r"(열이\s*나|열나|고열|미열)", 0.9, "내용: 증상 호소 (발열)"),
    (r"(기침|가래|콧물|재채기)", 0.8, "내용: 증상 호소 (호흡기)"),
    (r"(어지러|어지럽|두통|머리가\s*아프)", 0.9, "내용: 증상 호소 (두통/어지러움)"),
    (r"(소화가\s*안|속이\s*안\s*좋|메스꺼|구토|토할)", 0.9, "내용: 증상 호소 (소화기)"),
    (r"(잠이\s*안|못\s*자|불면)", 0.7, "내용: 증상 호소 (수면)"),

    # ── 기간/시점 표현 ──
    (r"(며칠\s*전부터|며칠\s*됐|며칠째)", 1.0, "내용: 기간 표현 (며칠) → 환자"),
    (r"(한\s*달\s*전|한달\s*전|한\s*달\s*됐)", 1.0, "내용: 기간 표현 (한 달) → 환자"),
    (r"(일주일\s*전|일주일\s*됐|한\s*주)", 1.0, "내용: 기간 표현 (일주일) → 환자"),
    (r"(어제부터|오늘\s*아침|그저께|엊그제)", 0.9, "내용: 기간 표현 (최근) → 환자"),
    (r"(작년|몇\s*년\s*전|몇달\s*전|몇\s*개월)", 0.8, "내용: 기간 표현 (장기) → 환자"),

    # ── 과거 병력/경험 ──
    (r"(전에도|전에\s*한\s*번|전에\s*있었)", 0.9, "내용: 과거 경험 → 환자"),
    (r"(예전에|옛날에|과거에)", 0.8, "내용: 과거 언급 → 환자"),
    (r"(수술\s*받았|수술\s*했|입원\s*했|입원한\s*적)", 0.9, "내용: 과거 수술/입원 → 환자"),
    (r"(다른\s*병원|동네\s*병원|대학\s*병원)\s*(에서|다녀|갔)", 0.8, "내용: 타 병원 경험 → 환자"),

    # ── 걱정/불안 ──
    (r"(걱정이|걱정되|걱정돼|걱정스러)", 0.9, "내용: 걱정 표현 → 환자"),
    (r"(괜찮을까|괜찮은\s*건가|괜찮은\s*거)", 0.9, "내용: 불안 질문 → 환자"),
    (r"(나을\s*수|나을까|낫|나아질)", 0.8, "내용: 회복 질문 → 환자"),
    (r"(심각한\s*건|큰\s*병|큰\s*거)", 0.8, "내용: 질환 걱정 → 환자"),
    (r"(무섭|두렵|겁나|겁이\s*나)", 0.7, "내용: 두려움 표현 → 환자"),

    # ── 약/치료 순응 보고 ──
    (r"약을?\s*(먹었|먹고\s*있|복용\s*하고|복용\s*했)", 1.0, "내용: 약 복용 보고 → 환자"),
    (r"(운동을?\s*했|운동\s*하고\s*있|걷기를?\s*했)", 0.8, "내용: 운동 이행 보고 → 환자"),
    (r"(찜질|파스|연고)\s*(했|붙였|발랐)", 0.7, "내용: 자가 치료 보고 → 환자"),
    (r"(안\s*먹었|못\s*먹었|빼먹|깜빡)", 0.8, "내용: 약 미복용 보고 → 환자"),

    # ── 1인칭/감정 ──
    (r"(저는|제가|저도|저한테|저를|저의)", 0.5, "내용: 1인칭 '저' → 환자"),
    (r"(죽고\s*싶|살고\s*싶|포기하지)", 1.0, "내용: 감정 호소"),
    (r"(힘들|우울|불안|무섭)", 0.6, "내용: 감정 표현"),
    (r"살고\s*싶습니다", 1.5, "내용: '살고 싶습니다' → 환자"),

    # ── 수용/확인 ──
    (r"(알겠습니다|알겠어요|알았어요)", 0.5, "내용: 수용 표현"),
    (r"(그렇군요|그렇구나|아\s*그렇|아\s*네)", 0.4, "내용: 이해 표현 → 환자"),

    # ── 질문 (의사에게) ──
    (r"(언제|몇\s*번|얼마나).*(와야|가야|해야|받아야)", 0.6, "내용: 치료 일정 질문"),
    (r"(뭐\s*하는|어떤\s*식으로|어떻게).*(거예요|건가요|거에요|거야)", 0.6, "내용: 치료 방법 질문"),
    (r"(보험|실비|비용|돈|원짜리)", 0.3, "내용: 비용 관련 (약한 신호)"),
    (r"(얼마나\s*걸리|시간이\s*얼마)", 0.5, "내용: 소요 시간 질문 → 환자"),
]

# 사전 컴파일
CONTENT_PATTERNS_DOCTOR = _compile_patterns(_RAW_CONTENT_PATTERNS_DOCTOR)
CONTENT_PATTERNS_PATIENT = _compile_patterns(_RAW_CONTENT_PATTERNS_PATIENT)

# 의학 용어 정규식 사전 컴파일 (장문 분석용)
_MEDICAL_TERMS_RE = re.compile(
    r"(수술|치료|진단|검사|처방|증상|합병증|마취|절제|봉합|주사|약물|투여|소견|종양|신경|혈관|조직|"
    r"방사선|항생제|진통제|소변|혈압|혈당|담즙|담관|담도|척추|디스크|관절|골절|인대|힘줄|근육|"
    r"신장|간|폐|심장|위|장|방광|요도|점막|염증|감염|출혈|부종|경화|괴사|암|종|낭종|석회|"
    r"골다공증|당뇨|고혈압|저혈압|빈혈|갑상선|류마티스|통풍|대상포진|심전도|내시경|위내시경|"
    r"대장내시경|조직검사|생검|병리|세포|림프|항암|방사선치료|화학요법|면역|백신)"
)

# Q&A 패턴 판별용 정규식 (질문 감지)
_QUESTION_RE = re.compile(
    r"([\?？]|"  # 물음표
    r"(인가요|건가요|을까요|ㄹ까요|나요|던가요|은가요|가요)\s*$|"  # 질문 어미
    r"(어떠|어때|어디|언제|몇|얼마|뭐|무엇|누가|왜|어떻게|어디서))"  # 의문사
)


def strategy_b_content(content: str) -> SpeakerSignal:
    """발화 내용 기반 화자 판별. 사전 컴파일된 정규식 사용."""
    sig = SpeakerSignal()

    # 의사 패턴 매칭
    for compiled_re, score, desc in CONTENT_PATTERNS_DOCTOR:
        if compiled_re.search(content):
            sig.doctor_score += score
            sig.signals.append(desc)

    # 환자 패턴 매칭
    for compiled_re, score, desc in CONTENT_PATTERNS_PATIENT:
        if compiled_re.search(content):
            sig.patient_score += score
            sig.signals.append(desc)

    # 장문 + 의학 용어 다수 → 의사일 가능성 높음
    if len(content) > 80:
        medical_terms = len(_MEDICAL_TERMS_RE.findall(content))
        if medical_terms >= 3:
            sig.doctor_score += 1.0
            sig.signals.append(f"내용: 의학용어 다수({medical_terms}개) + 장문")

    return sig


# ─────────────────────────────────────────────
# 진료 단계 판별
# ─────────────────────────────────────────────

# 단계 키워드 정규식 사전 컴파일
_PHASE_COMPILED: dict[ConversationPhase, list[re.Pattern]] = {}
for _phase, _keywords in PHASE_KEYWORDS.items():
    _PHASE_COMPILED[_phase] = [re.compile(re.escape(kw)) for kw in _keywords]


def _detect_phase(content: str) -> ConversationPhase | None:
    """세그먼트 내용에서 진료 단계를 판별. 해당 없으면 None."""
    best_phase = None
    best_count = 0
    for phase, patterns in _PHASE_COMPILED.items():
        count = sum(1 for p in patterns if p.search(content))
        if count > best_count:
            best_count = count
            best_phase = phase
    # 최소 1개 이상 매칭되어야 단계 판별 유효
    return best_phase if best_count >= 1 else None


def _map_conversation_phases(segments: list[dict]) -> list[ConversationPhase | None]:
    """전체 세그먼트 리스트를 진료 단계로 매핑.

    각 세그먼트의 내용을 분석하여 해당하는 진료 단계를 반환.
    단계가 명확하지 않은 세그먼트는 인접 세그먼트의 단계를 참고하여 보간.
    """
    n = len(segments)
    if n == 0:
        return []

    # 1차: 직접 판별
    phases: list[ConversationPhase | None] = []
    for seg in segments:
        content = seg.get("content", "")
        phase = _detect_phase(content)
        phases.append(phase)

    # 2차: 보간 (None인 세그먼트는 가장 가까운 앞쪽 단계를 계승)
    last_known = None
    for i in range(n):
        if phases[i] is not None:
            last_known = phases[i]
        elif last_known is not None:
            phases[i] = last_known

    # 3차: 앞쪽에 None이 남아있으면 뒤쪽에서 역방향 보간
    last_known = None
    for i in range(n - 1, -1, -1):
        if phases[i] is not None:
            last_known = phases[i]
        elif last_known is not None:
            phases[i] = last_known

    return phases


# ─────────────────────────────────────────────
# Context Window Analysis (문맥 창 분석)
# ─────────────────────────────────────────────

def _analyze_context(
    segments: list[dict],
    index: int,
    ab_signals: list[SpeakerSignal],
    window: int = 3,
) -> SpeakerSignal:
    """주변 세그먼트를 분석하여 현재 세그먼트의 화자 신호를 추정.

    3가지 문맥 신호를 종합:
    1) Q&A 패턴: 질문 뒤에는 다른 화자가 대답
    2) 역할 모멘텀: 연속 동일 화자 → 다음도 같은 화자일 가능성
    3) 주제 연속성: 동일 주제면 같은 화자가 계속 말할 가능성

    Args:
        segments: 전체 세그먼트 리스트
        index: 분석 대상 세그먼트 인덱스
        ab_signals: 각 세그먼트의 AB 분석 결과
        window: 앞뒤로 살펴볼 세그먼트 수 (기본 3)

    Returns:
        문맥 기반 화자 신호
    """
    sig = SpeakerSignal()
    n = len(segments)
    current_content = segments[index].get("content", "")
    current_role = segments[index].get("role", "")

    # ── 1) Q&A 패턴 분석 ──
    # 바로 앞 세그먼트가 질문이면, 현재는 대답 → 다른 화자
    if index > 0:
        prev_content = segments[index - 1].get("content", "")
        prev_role = segments[index - 1].get("role", "")
        prev_signal = ab_signals[index - 1] if index - 1 < len(ab_signals) else None

        # 이전 발화가 질문인지 확인
        is_prev_question = bool(_QUESTION_RE.search(prev_content))

        if is_prev_question:
            # 질문 뒤의 대답은 다른 화자
            if prev_signal and prev_signal.predicted_role is not None:
                if prev_signal.predicted_role == Role.DOCTOR:
                    # 의사가 질문 → 환자가 대답
                    sig.patient_score += 0.6
                    sig.signals.append("문맥: 의사 질문 뒤 대답 → 환자")
                else:
                    # 환자가 질문 → 의사가 대답
                    sig.doctor_score += 0.6
                    sig.signals.append("문맥: 환자 질문 뒤 대답 → 의사")
            elif prev_role == Role.DOCTOR.value:
                sig.patient_score += 0.4
                sig.signals.append("문맥: 의사(role) 질문 뒤 대답 → 환자")
            elif prev_role == Role.PATIENT.value:
                sig.doctor_score += 0.4
                sig.signals.append("문맥: 환자(role) 질문 뒤 대답 → 의사")

    # 현재가 질문이고 바로 뒤 세그먼트가 있으면, 대답 관계 확인
    if index < n - 1:
        is_current_question = bool(_QUESTION_RE.search(current_content))
        next_signal = ab_signals[index + 1] if index + 1 < len(ab_signals) else None

        if is_current_question and next_signal and next_signal.predicted_role is not None:
            # 내가 질문하면, 다음은 다른 화자 → 나는 반대 화자
            if next_signal.predicted_role == Role.DOCTOR:
                sig.patient_score += 0.4
                sig.signals.append("문맥: 질문자 → 다음 의사 대답 → 환자")
            else:
                sig.doctor_score += 0.4
                sig.signals.append("문맥: 질문자 → 다음 환자 대답 → 의사")

    # ── 2) 역할 모멘텀 분석 ──
    # 앞쪽 연속 동일 화자 패턴 확인 (최대 window 개)
    consecutive_same = 0
    momentum_role = None

    for i in range(index - 1, max(index - window - 1, -1), -1):
        if i < 0:
            break
        prev_sig = ab_signals[i] if i < len(ab_signals) else None
        if prev_sig and prev_sig.predicted_role is not None:
            if momentum_role is None:
                momentum_role = prev_sig.predicted_role
                consecutive_same = 1
            elif prev_sig.predicted_role == momentum_role:
                consecutive_same += 1
            else:
                break
        else:
            # 신호 없는 세그먼트는 role 기반으로 판단
            seg_role = segments[i].get("role", "")
            if momentum_role is None:
                if seg_role == Role.DOCTOR.value:
                    momentum_role = Role.DOCTOR
                elif seg_role == Role.PATIENT.value:
                    momentum_role = Role.PATIENT
                consecutive_same = 1
            elif (momentum_role == Role.DOCTOR and seg_role == Role.DOCTOR.value) or \
                 (momentum_role == Role.PATIENT and seg_role == Role.PATIENT.value):
                consecutive_same += 1
            else:
                break

    # 3회 이상 연속 동일 화자 → 모멘텀 신호 (약한 가중치)
    if consecutive_same >= 3 and momentum_role is not None:
        momentum_weight = min(0.5, consecutive_same * 0.15)  # 최대 0.5
        if momentum_role == Role.DOCTOR:
            sig.doctor_score += momentum_weight
            sig.signals.append(f"문맥: 의사 모멘텀({consecutive_same}연속, +{momentum_weight:.2f})")
        else:
            sig.patient_score += momentum_weight
            sig.signals.append(f"문맥: 환자 모멘텀({consecutive_same}연속, +{momentum_weight:.2f})")

    # ── 3) 대화 교대 패턴 (Turn-taking) ──
    # 일반적인 대화는 화자가 번갈아 바뀜
    # 앞 세그먼트와 다른 화자일 가능성 (약한 신호)
    if index > 0:
        prev_role_value = segments[index - 1].get("role", "")
        # 앞과 같은 역할이면 약한 반대 신호 (대화는 교대하는 게 자연스러움)
        # 단, 모멘텀이 강하면 이 신호는 무시됨
        if consecutive_same < 2:  # 모멘텀이 약할 때만
            if prev_role_value == Role.DOCTOR.value:
                sig.patient_score += 0.2
                sig.signals.append("문맥: 대화 교대 패턴 → 환자")
            elif prev_role_value == Role.PATIENT.value:
                sig.doctor_score += 0.2
                sig.signals.append("문맥: 대화 교대 패턴 → 의사")

    return sig


# ─────────────────────────────────────────────
# Conversation Flow Analysis (대화 흐름 분석)
# ─────────────────────────────────────────────

def _analyze_conversation_flow(
    segments: list[dict],
    phases: list[ConversationPhase | None],
    index: int,
) -> SpeakerSignal:
    """진료 대화 흐름(단계)에 기반하여 화자 신호를 생성.

    각 진료 단계에서 기대되는 화자 비율과 현재 역할을 비교하여 신호를 제공.

    Args:
        segments: 전체 세그먼트 리스트
        phases: 각 세그먼트의 진료 단계
        index: 분석 대상 세그먼트 인덱스

    Returns:
        대화 흐름 기반 화자 신호
    """
    sig = SpeakerSignal()
    phase = phases[index] if index < len(phases) else None

    if phase is None:
        return sig

    expected_doctor_ratio = PHASE_EXPECTED_DOCTOR_RATIO.get(phase, 0.5)

    # 단계별 기대 비율이 극단적일 때만 신호 부여
    if expected_doctor_ratio >= 0.7:
        # 이 단계에서는 의사가 주로 발화
        sig.doctor_score += 0.4
        sig.signals.append(f"흐름: {phase.value} 단계 → 의사 주도 (기대 {expected_doctor_ratio:.0%})")
    elif expected_doctor_ratio <= 0.35:
        # 이 단계에서는 환자가 주로 발화
        sig.patient_score += 0.4
        sig.signals.append(f"흐름: {phase.value} 단계 → 환자 주도 (기대 {1-expected_doctor_ratio:.0%})")

    # 단계 전환 감지: 단계가 바뀌면 화자도 바뀔 가능성
    if index > 0:
        prev_phase = phases[index - 1] if index - 1 < len(phases) else None
        if prev_phase is not None and phase != prev_phase:
            # 단계 전환 시 화자 변화 가능성 (약한 신호)
            sig.signals.append(f"흐름: 단계 전환 ({prev_phase.value} → {phase.value})")

    return sig


# ─────────────────────────────────────────────
# GPT 검증 프롬프트
# ─────────────────────────────────────────────

GPT_SYSTEM_PROMPT = """당신은 의료 대화 화자분리 전문가입니다.
병원 진료 대화의 STT(음성→텍스트) 결과에서 화자(role)가 잘못 배정된 것을 교정합니다.

판단 기준:
1. 진단, 치료 설명, 처방, 검사 안내, "~해 드리겠습니다" → 원장님
2. 증상 호소, "선생님/원장님" 호칭, 감정 표현, 질문 → 환자
3. "환자분", "어머님", "아버지", "할머니", "어르신" 호칭 사용 → 원장님
4. 짧은 응답("네", "감사합니다")은 앞뒤 대화 흐름으로 판단
5. 의학 용어를 설명하는 긴 발화 → 원장님
6. 인사("안녕하세요")는 문맥상 누가 먼저 하는지로 판단
7. 대화 흐름: 주호소(환자)→병력청취(교대)→진찰(의사)→진단(의사)→치료설명(의사)→마무리(교대)
8. Q&A 패턴: 질문 뒤에는 다른 화자가 대답하는 것이 자연스러움

반드시 아래 JSON 형식으로만 응답하세요:
[{"index": 세그먼트index, "role": "원장님" 또는 "환자"}]
- 변경이 필요한 세그먼트만 포함
- 변경 없으면 빈 배열 []
- JSON 외 다른 텍스트 금지"""


# ─────────────────────────────────────────────
# GPT 검증 필요성 판단
# ─────────────────────────────────────────────

@dataclass
class GPTReviewDecision:
    """GPT 검증 필요성 판단 결과."""
    needs_review: bool
    reasons: list[str]
    scope: str  # "full" (파일 전체) | "partial" (일부 구간) | "none"
    target_indices: list[int]  # partial일 때 대상 인덱스


def assess_gpt_need(
    segments: list[dict],
    ab_signals: list[SpeakerSignal],
) -> GPTReviewDecision:
    """GPT 2차 검증이 필요한지, 어느 범위로 할지 판단."""
    reasons = []
    n = len(segments)
    if n == 0:
        return GPTReviewDecision(False, [], "none", [])

    # 1) 역할 비율 분석
    doc_count = sum(1 for s in segments if s.get("role") == "원장님")
    ratio = doc_count / n
    is_imbalanced = ratio < 0.25 or ratio > 0.75
    is_single = ratio == 0.0 or ratio == 1.0

    if is_single:
        reasons.append(f"단일화자: 전체 {n}개가 모두 같은 역할")
    elif is_imbalanced:
        reasons.append(f"역할 불균형: 의사 {ratio:.0%} / 환자 {1-ratio:.0%}")

    # 2) 연속 동일 화자 분석
    max_consec = 1
    cur_consec = 1
    consec_ranges = []  # (start, end) 3회+ 연속 구간
    for i in range(1, n):
        if segments[i].get("role") == segments[i-1].get("role"):
            cur_consec += 1
        else:
            if cur_consec >= 3:
                consec_ranges.append((i - cur_consec, i - 1))
            max_consec = max(max_consec, cur_consec)
            cur_consec = 1
    if cur_consec >= 3:
        consec_ranges.append((n - cur_consec, n - 1))
    max_consec = max(max_consec, cur_consec)

    if max_consec >= 5:
        reasons.append(f"연속 동일화자 {max_consec}회 감지")
    elif max_consec >= 3:
        reasons.append(f"연속 동일화자 {max_consec}회 감지 (경미)")

    # 3) AB 신호 커버리지
    no_signal_count = sum(1 for sig in ab_signals if not sig.has_signal)
    no_signal_ratio = no_signal_count / n
    if no_signal_ratio > 0.6:
        reasons.append(f"AB 신호 없는 세그먼트 {no_signal_ratio:.0%} ({no_signal_count}/{n})")

    # 4) 낮은 신뢰도 세그먼트 비율
    low_conf_count = sum(
        1 for sig in ab_signals
        if sig.has_signal and sig.confidence_level == ConfidenceLevel.LOW
    )
    if n > 0 and low_conf_count / n > 0.4:
        reasons.append(f"낮은 신뢰도 세그먼트 {low_conf_count}/{n} ({low_conf_count/n:.0%})")

    # 판단
    if not reasons:
        return GPTReviewDecision(False, [], "none", [])

    # 불균형/단일화자 → 전체 파일
    if is_imbalanced or is_single:
        return GPTReviewDecision(True, reasons, "full", list(range(n)))

    # 연속 구간만 → 구간 + 앞뒤 문맥 2개씩
    if consec_ranges:
        indices = set()
        for start, end in consec_ranges:
            for i in range(max(0, start - 2), min(n, end + 3)):
                indices.add(i)
        return GPTReviewDecision(True, reasons, "partial", sorted(indices))

    # 신호 부족 → 전체
    if no_signal_ratio > 0.6:
        return GPTReviewDecision(True, reasons, "full", list(range(n)))

    # 낮은 신뢰도 다수 → 해당 인덱스들만
    if low_conf_count > 0:
        low_indices = [
            i for i, sig in enumerate(ab_signals)
            if sig.has_signal and sig.confidence_level == ConfidenceLevel.LOW
        ]
        return GPTReviewDecision(True, reasons, "partial", low_indices)

    return GPTReviewDecision(False, reasons, "none", [])


# ─────────────────────────────────────────────
# GPT 배치 호출 유틸리티
# ─────────────────────────────────────────────

# GPT API 한 번에 보낼 최대 세그먼트 수
GPT_BATCH_SIZE = 40


def _chunk_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    """인덱스 리스트를 chunk_size 단위로 분할."""
    chunks = []
    for i in range(0, len(indices), chunk_size):
        chunks.append(indices[i:i + chunk_size])
    return chunks


# ─────────────────────────────────────────────
# 통합 교정기
# ─────────────────────────────────────────────

class SpeakerCorrector:
    """화자분리 후처리 교정기.

    2단계: AB 패턴 → GPT-4o-mini 검증.

    추가 분석:
    - Context Window Analysis: 주변 세그먼트 기반 문맥 분석
    - Conversation Flow Analysis: 진료 단계별 화자 패턴 분석
    - 다중 신호 가중 합산 신뢰도 스코어링
    """

    # 신호별 가중치 (다중 신호 가중 합산)
    WEIGHT_HONORIFIC = 1.0      # 전략 A: 호칭 (가장 확실한 신호)
    WEIGHT_CONTENT = 0.8        # 전략 B: 발화 내용
    WEIGHT_CONTEXT = 0.5        # 문맥 창 분석
    WEIGHT_FLOW = 0.3           # 대화 흐름 분석

    def __init__(
        self,
        openai_api_key: str | None = None,
        use_gpt: bool = True,
        gpt_model: str = "gpt-4o-mini",
        correction_threshold: float = CORRECTION_THRESHOLD,
    ):
        """초기화.

        Args:
            openai_api_key: OpenAI API 키
            use_gpt: GPT 2단계 사용 여부
            gpt_model: 사용할 GPT 모델명
            correction_threshold: 교정 적용 최소 신뢰도 임계값 (기본 0.3)
        """
        self.openai_api_key = openai_api_key
        self.use_gpt = use_gpt and (openai_api_key is not None)
        self.gpt_model = gpt_model
        self.correction_threshold = correction_threshold
        self._client = None

        # 분석 결과 캐시 (동일 세그먼트 재분석 방지)
        self._ab_cache: dict[str, SpeakerSignal] = {}
        self._phase_cache: list[ConversationPhase | None] | None = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화."""
        if self._client is None and self.openai_api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.openai_api_key)
        return self._client

    def _clear_cache(self) -> None:
        """분석 캐시 초기화. 새로운 대화 분석 시작 시 호출."""
        self._ab_cache.clear()
        self._phase_cache = None

    def _get_content_key(self, content: str) -> str:
        """캐시 키 생성. 내용 기반."""
        # 짧은 내용은 그대로, 긴 내용은 해시 사용
        if len(content) <= 100:
            return content
        return f"{content[:50]}...{hash(content)}"

    # ── AB 1단계 (호칭 + 내용) ──

    def _ab_analyze_single(self, content: str) -> SpeakerSignal:
        """단일 세그먼트의 AB 패턴 분석 (캐시 활용)."""
        cache_key = self._get_content_key(content)
        if cache_key in self._ab_cache:
            # 캐시된 결과 복사 (원본 보호)
            cached = self._ab_cache[cache_key]
            result = SpeakerSignal(
                doctor_score=cached.doctor_score,
                patient_score=cached.patient_score,
                signals=list(cached.signals),
            )
            return result

        # 전략 A: 호칭 분석
        sig_a = strategy_a_honorific(content)
        # 전략 B: 내용 분석
        sig_b = strategy_b_content(content)

        # 합산
        result = SpeakerSignal(
            doctor_score=sig_a.doctor_score + sig_b.doctor_score,
            patient_score=sig_a.patient_score + sig_b.patient_score,
            signals=sig_a.signals + sig_b.signals,
        )

        # 캐시 저장
        self._ab_cache[cache_key] = SpeakerSignal(
            doctor_score=result.doctor_score,
            patient_score=result.patient_score,
            signals=list(result.signals),
        )

        return result

    def _ab_analyze(self, segments: list[dict]) -> list[SpeakerSignal]:
        """AB 패턴 분석 (신호 수집만, 교정 안 함). 캐시 활용."""
        signals = []
        for seg in segments:
            content = seg.get("content", "")
            sig = self._ab_analyze_single(content)
            signals.append(sig)
        return signals

    def _full_analyze(self, segments: list[dict]) -> list[SpeakerSignal]:
        """모든 분석 전략을 통합하여 화자 신호를 생성.

        가중치 합산:
        - 전략 A (호칭): weight=1.0
        - 전략 B (내용): weight=0.8
        - 문맥 창 분석: weight=0.5
        - 대화 흐름 분석: weight=0.3

        실제로는 A와 B가 이미 _ab_analyze_single에서 합산되어 있으므로,
        여기서는 문맥과 흐름 신호를 추가로 병합.
        """
        n = len(segments)
        if n == 0:
            return []

        # 1) AB 기본 분석 (호칭 + 내용)
        ab_signals = self._ab_analyze(segments)

        # 2) 진료 단계 매핑 (전체 대화 한 번에)
        if self._phase_cache is None or len(self._phase_cache) != n:
            self._phase_cache = _map_conversation_phases(segments)
        phases = self._phase_cache

        # 3) 각 세그먼트에 문맥 + 흐름 신호 추가
        full_signals = []
        for i in range(n):
            # AB 신호 복사 (원본 보존)
            sig = SpeakerSignal(
                doctor_score=ab_signals[i].doctor_score,
                patient_score=ab_signals[i].patient_score,
                signals=list(ab_signals[i].signals),
            )

            # 문맥 창 분석 (주변 세그먼트 참고)
            ctx_sig = _analyze_context(segments, i, ab_signals, window=3)
            sig.merge(ctx_sig, weight=self.WEIGHT_CONTEXT)

            # 대화 흐름 분석 (진료 단계)
            flow_sig = _analyze_conversation_flow(segments, phases, i)
            sig.merge(flow_sig, weight=self.WEIGHT_FLOW)

            full_signals.append(sig)

        return full_signals

    def _ab_correct(
        self, segments: list[dict], signals: list[SpeakerSignal]
    ) -> list[CorrectionResult]:
        """신호 기반 교정. 신뢰도 임계값 이상인 경우만 교정 적용."""
        results = []
        for seg, sig in zip(segments, signals):
            orig_role = seg.get("role", "")
            predicted = sig.predicted_role
            confidence = sig.confidence

            # 교정 조건: 예측 역할이 있고, 원래 역할과 다르고, 신뢰도가 임계값 이상
            if (predicted is not None
                    and predicted.value != orig_role
                    and confidence >= self.correction_threshold):
                results.append(CorrectionResult(
                    original_role=orig_role,
                    corrected_role=predicted.value,
                    strategy=f"AB(conf={confidence:.2f}, level={sig.confidence_level.value})",
                    changed=True,
                    confidence=confidence,
                    signals=sig.signals,
                ))
            else:
                results.append(CorrectionResult(
                    original_role=orig_role,
                    corrected_role=orig_role,
                    strategy="AB-유지",
                    changed=False,
                    confidence=confidence if sig.has_signal else 0.0,
                    signals=sig.signals,
                ))
        return results

    # ── GPT 2단계 ──

    def _gpt_correct_batch(
        self,
        segments: list[dict],
        target_indices: list[int],
    ) -> dict[int, str]:
        """GPT-4o-mini로 화자 교정 (배치 처리).

        세그먼트가 GPT_BATCH_SIZE를 초과하면 분할하여 호출.
        반환: {index: corrected_role}
        """
        client = self._get_client()
        if client is None:
            return {}

        all_changes: dict[int, str] = {}
        chunks = _chunk_indices(target_indices, GPT_BATCH_SIZE)

        for chunk_indices in chunks:
            changes = self._gpt_call(segments, chunk_indices)
            all_changes.update(changes)

        return all_changes

    def _gpt_call(
        self,
        segments: list[dict],
        target_indices: list[int],
    ) -> dict[int, str]:
        """단일 GPT API 호출. 반환: {index: corrected_role}."""
        client = self._get_client()
        if client is None:
            return {}

        # 전송할 세그먼트 준비
        formatted = []
        for i in target_indices:
            if i < len(segments):
                seg = segments[i]
                formatted.append({
                    "index": seg.get("index", i),
                    "role": seg.get("role", ""),
                    "content": seg.get("content", "")[:300],  # 토큰 절약
                })

        if not formatted:
            return {}

        user_msg = json.dumps(formatted, ensure_ascii=False)

        try:
            response = client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()

            # JSON 추출 (마크다운 코드블록 처리)
            if "```" in raw:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]

            changes = json.loads(raw)
            return {c["index"]: c["role"] for c in changes if "index" in c and "role" in c}

        except json.JSONDecodeError as e:
            logger.warning(f"GPT 응답 JSON 파싱 실패: {e}")
            return {}
        except Exception as e:
            logger.warning(f"GPT 화자교정 실패: {e}")
            return {}

    def _gpt_correct(
        self,
        segments: list[dict],
        target_indices: list[int] | None = None,
    ) -> dict[int, str]:
        """GPT-4o-mini로 화자 교정. 반환: {index: corrected_role}.

        target_indices가 None이면 전체 세그먼트 대상.
        GPT_BATCH_SIZE 초과 시 배치 분할 호출.
        """
        if target_indices is None:
            target_indices = list(range(len(segments)))

        if not target_indices:
            return {}

        return self._gpt_correct_batch(segments, target_indices)

    # ── 통합 파이프라인 ──

    def correct(self, segments: list[dict]) -> list[CorrectionResult]:
        """2단계 파이프라인: AB+문맥+흐름 → GPT 검증.

        Args:
            segments: [{"role": str, "index": int, "content": str}, ...]

        Returns:
            교정 결과 리스트
        """
        if not segments:
            return []

        # 캐시 초기화 (새로운 대화)
        self._clear_cache()

        # 1단계: 통합 분석 (AB + 문맥 + 흐름)
        signals = self._full_analyze(segments)
        ab_results = self._ab_correct(segments, signals)

        # AB 결과 적용한 중간 세그먼트
        mid_segments = []
        for seg, res in zip(segments, ab_results):
            mid = dict(seg)
            mid["role"] = res.corrected_role
            mid_segments.append(mid)

        # GPT 사용 안 하면 AB 결과만 반환
        if not self.use_gpt:
            return ab_results

        # 2단계: GPT 필요성 판단 (AB 적용 후 상태 기준)
        mid_signals = self._ab_analyze(mid_segments)
        decision = assess_gpt_need(mid_segments, mid_signals)

        if not decision.needs_review:
            logger.info("GPT 검증 불필요: AB+문맥+흐름 교정만으로 충분")
            return ab_results

        logger.info(
            f"GPT 검증 실행: scope={decision.scope}, "
            f"reasons={decision.reasons}"
        )

        # GPT 호출 (배치 처리)
        gpt_changes = self._gpt_correct(
            mid_segments,
            target_indices=decision.target_indices if decision.scope == "partial" else None,
        )

        # 3단계: AB + GPT 결과 병합
        final_results = []
        for i, (seg, ab_res) in enumerate(zip(segments, ab_results)):
            orig_index = seg.get("index", i)

            if orig_index in gpt_changes:
                gpt_role = gpt_changes[orig_index]

                if ab_res.changed and ab_res.corrected_role == gpt_role:
                    # AB와 GPT 일치 → 높은 신뢰도
                    final_results.append(CorrectionResult(
                        original_role=ab_res.original_role,
                        corrected_role=gpt_role,
                        changed=(ab_res.original_role != gpt_role),
                        strategy="AB+GPT 일치",
                        confidence=1.0,
                        signals=ab_res.signals + ["GPT 확인"],
                    ))
                elif ab_res.changed and ab_res.corrected_role != gpt_role:
                    # AB와 GPT 불일치 → GPT 우선 (문맥 이해 우수)
                    final_results.append(CorrectionResult(
                        original_role=ab_res.original_role,
                        corrected_role=gpt_role,
                        changed=(ab_res.original_role != gpt_role),
                        strategy="GPT 우선(AB 불일치)",
                        confidence=0.8,
                        signals=ab_res.signals + [f"GPT→{gpt_role} (AB→{ab_res.corrected_role})"],
                    ))
                else:
                    # AB는 유지했지만 GPT가 변경 → GPT 적용
                    final_results.append(CorrectionResult(
                        original_role=ab_res.original_role,
                        corrected_role=gpt_role,
                        changed=(ab_res.original_role != gpt_role),
                        strategy="GPT 추가교정",
                        confidence=0.85,
                        signals=ab_res.signals + [f"GPT→{gpt_role}"],
                    ))
            else:
                # GPT가 언급 안 함 → AB 결과 유지
                final_results.append(ab_res)

        return final_results

    def correct_ab_only(self, segments: list[dict]) -> list[CorrectionResult]:
        """AB+문맥+흐름 패턴만으로 교정 (GPT 미사용)."""
        self._clear_cache()
        signals = self._full_analyze(segments)
        return self._ab_correct(segments, signals)

    def apply(self, segments: list[dict]) -> list[dict]:
        """교정 적용한 새 세그먼트 리스트 반환."""
        results = self.correct(segments)
        corrected = []
        for seg, res in zip(segments, results):
            new_seg = dict(seg)
            new_seg["role"] = res.corrected_role
            if res.changed:
                new_seg["_speaker_corrected"] = True
                new_seg["_correction_strategy"] = res.strategy
                new_seg["_original_role"] = res.original_role
                new_seg["_confidence"] = res.confidence
                new_seg["_confidence_level"] = res.confidence_level.value
            corrected.append(new_seg)
        return corrected

    def analyze_signals(self, segments: list[dict]) -> list[dict]:
        """디버깅/분석용: 각 세그먼트의 모든 신호를 상세히 반환.

        교정은 적용하지 않고, 분석 결과만 반환.

        Returns:
            [{"index": int, "content": str, "role": str,
              "doctor_score": float, "patient_score": float,
              "predicted_role": str|None, "confidence": float,
              "confidence_level": str, "signals": list[str],
              "phase": str|None}, ...]
        """
        self._clear_cache()
        signals = self._full_analyze(segments)
        phases = self._phase_cache or _map_conversation_phases(segments)

        analysis = []
        for i, (seg, sig) in enumerate(zip(segments, signals)):
            phase = phases[i] if i < len(phases) else None
            analysis.append({
                "index": seg.get("index", i),
                "content": seg.get("content", "")[:100],
                "role": seg.get("role", ""),
                "doctor_score": round(sig.doctor_score, 3),
                "patient_score": round(sig.patient_score, 3),
                "predicted_role": sig.predicted_role.value if sig.predicted_role else None,
                "confidence": round(sig.confidence, 3),
                "confidence_level": sig.confidence_level.value,
                "signals": sig.signals,
                "phase": phase.value if phase else None,
            })

        return analysis
