"""의료 용어 교정 데이터 모델."""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class MatchStrategy(str, Enum):
    EXACT = "exact"
    REGEX = "regex"
    PHONETIC = "phonetic"


class CorrectionTier(str, Enum):
    """교정이 수행된 계층 (Tier)."""
    TIER1_DICT = "tier1_dict"          # 사전 기반 교정
    TIER2_AUTO = "tier2_auto"          # KOSTOM 자동 탐지 교정
    TIER2_REVIEW = "tier2_review"      # KOSTOM 검증 대기
    HALLUCINATION = "hallucination"    # 환각 제거


class DictEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    wrong: str
    correct: str
    category: str = "일반"
    strategy: MatchStrategy = MatchStrategy.EXACT
    pattern: str | None = None
    context_hint: list[str] = Field(default_factory=list)
    priority: int = 50
    confidence: float = 1.0
    enabled: bool = True
    notes: str = ""


class DictEntryCreate(BaseModel):
    wrong: str
    correct: str
    category: str = "일반"
    strategy: MatchStrategy = MatchStrategy.EXACT
    pattern: str | None = None
    context_hint: list[str] = Field(default_factory=list)
    priority: int = 50
    confidence: float = 1.0
    enabled: bool = True
    notes: str = ""


class DictEntryUpdate(BaseModel):
    wrong: str | None = None
    correct: str | None = None
    category: str | None = None
    strategy: MatchStrategy | None = None
    pattern: str | None = None
    context_hint: list[str] | None = None
    priority: int | None = None
    confidence: float | None = None
    enabled: bool | None = None
    notes: str | None = None


class MedicalDict(BaseModel):
    version: str = "1.0"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_terms: list[str] = Field(default_factory=list)
    entries: list[DictEntry] = Field(default_factory=list)


class CorrectionLog(BaseModel):
    original: str
    corrected: str
    entry_id: str
    strategy: MatchStrategy
    tier: CorrectionTier = CorrectionTier.TIER1_DICT  # 기본값: Tier 1 (하위 호환)


class CorrectionResult(BaseModel):
    text: str
    logs: list[CorrectionLog] = Field(default_factory=list)


class FullCorrectionResult(BaseModel):
    """correct_full()의 반환 모델 — Tier 정보 포함."""
    text: str
    original_text: str
    logs: list[CorrectionLog] = Field(default_factory=list)
    tier1_count: int = 0       # Tier 1 교정 횟수
    tier2_count: int = 0       # Tier 2 교정 횟수
    pending_count: int = 0     # 검증 대기 항목 수
    context_hint: str | None = None  # 사용된 문맥 힌트


class DictStats(BaseModel):
    total_entries: int = 0
    enabled_entries: int = 0
    disabled_entries: int = 0
    categories: dict[str, int] = Field(default_factory=dict)
    strategies: dict[str, int] = Field(default_factory=dict)


class EngineStats(BaseModel):
    """엔진 전체 통계 모델."""
    # Tier 1 사전 통계
    dict_total: int = 0
    dict_enabled: int = 0
    dict_exact: int = 0
    dict_regex: int = 0
    dict_phonetic: int = 0
    # Tier 2 자동 탐지 통계
    tier2_enabled: bool = False
    tier2_total_detections: int = 0
    tier2_auto_corrections: int = 0
    tier2_pending_reviews: int = 0
    tier2_cache_size: int = 0
    tier2_cache_hits: int = 0
    # 참조 DB 통계
    ref_db_loaded: bool = False
    ref_db_specialties: int = 0
    ref_db_total_terms: int = 0
    ref_db_cache_size: int = 0
    # 학습 통계
    learning_auto_learned: int = 0
    learning_llm_verified: int = 0
    learning_manual: int = 0
    learning_pending: int = 0
    learning_approved: int = 0
    learning_rejected: int = 0
    # 교정 세션 통계
    total_corrections_made: int = 0
    total_texts_processed: int = 0


class TestRequest(BaseModel):
    text: str


class ImportRequest(BaseModel):
    entries: list[DictEntryCreate]
