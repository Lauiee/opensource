"""3-Tier 의료 용어 교정 엔진: 사전(Tier1) → 자동탐지(Tier2) → GPT검증(Tier3)."""

import logging
import re
from pathlib import Path

from app.medterm.models import (
    CorrectionLog,
    CorrectionResult,
    CorrectionTier,
    DictEntry,
    EngineStats,
    FullCorrectionResult,
    MatchStrategy,
)
from app.medterm.phonetic import jamo_similarity
from app.medterm.store import DictionaryStore

logger = logging.getLogger(__name__)

# Tier 2 모듈 (옵셔널 — 참조 DB 없으면 Tier 1만 동작)
_auto_detector = None
_learning_manager = None
_ref_db = None

# 환각 패턴 (기존 postprocessing.py에서 이전)
_HALLUCINATION_PATTERNS = [
    (r"(.{10,}?)\1{1,}", r"\1"),
    (r"(감사합니다\.?\s*){3,}", ""),
    (r"(네\.?\s*){5,}", ""),
    (r"(MBC 뉴스.?\s*){2,}", ""),
    (r"(KBS 뉴스.?\s*){2,}", ""),
    (r"(시청해 주셔서 감사합니다.?\s*){2,}", ""),
    (r"(구독과 좋아요.?\s*){2,}", ""),
]


class MedicalCorrectionEngine:
    """의료 용어 교정 엔진 — Tier 1(사전) + Tier 2(자동탐지) 통합."""

    def __init__(self, store: DictionaryStore):
        self._store = store
        self._exact_entries: list[DictEntry] = []
        self._regex_entries: list[DictEntry] = []
        self._phonetic_entries: list[DictEntry] = []
        self._compile()

        # 교정 세션 통계
        self._total_corrections = 0
        self._total_texts_processed = 0

    def _compile(self) -> None:
        """사전 항목을 전략별로 분류하고 정렬."""
        entries = self._store.get_entries(enabled_only=True)
        self._exact_entries = sorted(
            [e for e in entries if e.strategy == MatchStrategy.EXACT],
            key=lambda e: (-e.priority, -len(e.wrong)),
        )
        self._regex_entries = [e for e in entries if e.strategy == MatchStrategy.REGEX]
        self._phonetic_entries = [e for e in entries if e.strategy == MatchStrategy.PHONETIC]
        logger.info(
            "엔진 컴파일: exact=%d, regex=%d, phonetic=%d",
            len(self._exact_entries),
            len(self._regex_entries),
            len(self._phonetic_entries),
        )

    def reload(self) -> int:
        """사전 핫 리로드."""
        count = self._store.reload()
        self._compile()
        return count

    # ──────────────────────────────────────────────
    # 기존 correct() — 하위 호환 유지
    # ──────────────────────────────────────────────
    def correct(self, text: str) -> CorrectionResult:
        """텍스트 교정 실행 (Tier 1 + Tier 2).

        하위 호환을 위해 CorrectionResult 반환.
        Tier 정보가 필요하면 correct_full()을 사용할 것.
        """
        full_result = self.correct_full(text)
        return CorrectionResult(text=full_result.text, logs=full_result.logs)

    # ──────────────────────────────────────────────
    # 신규 correct_full() — Tier 정보 포함 전체 파이프라인
    # ──────────────────────────────────────────────
    def correct_full(
        self,
        text: str,
        context_hint: str | None = None,
    ) -> FullCorrectionResult:
        """텍스트 교정 전체 파이프라인 (Tier 1 + Tier 2).

        Args:
            text: 교정할 텍스트
            context_hint: 진료과 힌트 (예: "정형외과", "내과")
                         Tier 2 자동 탐지 시 검색 범위를 좁히는 데 사용

        Returns:
            FullCorrectionResult: Tier 정보 포함 교정 결과
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return FullCorrectionResult(
                text=text or "",
                original_text=text or "",
                context_hint=context_hint,
            )

        original_text = text
        logs: list[CorrectionLog] = []
        tier1_count = 0
        tier2_count = 0
        pending_count = 0

        # === Tier 1: 사전 기반 교정 (μs) ===
        # Phase 1: 환각 제거
        text_before = text
        text = self._remove_hallucinations(text)
        if text != text_before:
            logs.append(CorrectionLog(
                original=text_before,
                corrected=text,
                entry_id="hallucination_filter",
                strategy=MatchStrategy.EXACT,
                tier=CorrectionTier.HALLUCINATION,
            ))

        # Phase 2: Regex 패턴 교정
        tier1_before = len(logs)
        text = self._apply_regex(text, logs)
        # Tier 1 태그 설정
        for log in logs[tier1_before:]:
            log.tier = CorrectionTier.TIER1_DICT

        # Phase 3: Exact 매칭 교정
        tier1_before = len(logs)
        text = self._apply_exact(text, logs)
        for log in logs[tier1_before:]:
            log.tier = CorrectionTier.TIER1_DICT

        # Phase 4: Phonetic 유사도 교정
        tier1_before = len(logs)
        text = self._apply_phonetic(text, logs)
        for log in logs[tier1_before:]:
            log.tier = CorrectionTier.TIER1_DICT

        # Tier 1 교정 횟수 집계 (환각 제거 제외)
        tier1_count = sum(
            1 for log in logs if log.tier == CorrectionTier.TIER1_DICT
        )

        # === Tier 2: 미등록 오류 자동 탐지 (ms) ===
        tier2_before = len(logs)
        text, _pending = self._apply_auto_detection(text, logs, context_hint)
        tier2_count = sum(
            1 for log in logs[tier2_before:] if log.tier == CorrectionTier.TIER2_AUTO
        )
        pending_count = _pending

        # 통계 업데이트
        total_new = tier1_count + tier2_count
        self._total_corrections += total_new
        self._total_texts_processed += 1

        return FullCorrectionResult(
            text=text,
            original_text=original_text,
            logs=logs,
            tier1_count=tier1_count,
            tier2_count=tier2_count,
            pending_count=pending_count,
            context_hint=context_hint,
        )

    def _apply_auto_detection(
        self,
        text: str,
        logs: list[CorrectionLog],
        context_hint: str | None = None,
    ) -> tuple[str, int]:
        """Tier 2: KOSTOM 참조 DB 기반 미등록 오류 탐지 + 자동 학습.

        Returns:
            (교정된 텍스트, pending 수)
        """
        global _auto_detector, _learning_manager
        pending_count = 0

        if _auto_detector is None or _learning_manager is None:
            return text, 0

        try:
            detections = _auto_detector.detect(text, context_hint=context_hint)
            for det in detections:
                best = det.best_candidate()
                if not best:
                    continue

                if det.action == "auto_correct":
                    # 고확신: 즉시 교정 + 사전 학습
                    new_text = text.replace(det.word, best["term"])
                    if new_text != text:
                        logs.append(CorrectionLog(
                            original=det.word,
                            corrected=best["term"],
                            entry_id=f"auto_{best['similarity']:.2f}",
                            strategy=MatchStrategy.EXACT,
                            tier=CorrectionTier.TIER2_AUTO,
                        ))
                        text = new_text
                        _learning_manager.auto_learn(det, context=text)
                        # 엔진 리컴파일 (새 항목 반영)
                        self._compile()
                        # 캐시 무효화 (사전 변경됨)
                        _auto_detector.invalidate_cache()

                elif det.action == "needs_review":
                    # 불확실: 검증 대기열로
                    _learning_manager.add_pending_review(det, context=text)
                    pending_count += 1
                    # 검증 대기 로그도 기록 (교정은 안 함)
                    logs.append(CorrectionLog(
                        original=det.word,
                        corrected=best["term"],
                        entry_id=f"review_{best['similarity']:.2f}",
                        strategy=MatchStrategy.EXACT,
                        tier=CorrectionTier.TIER2_REVIEW,
                    ))

        except Exception:
            logger.exception("Tier 2 자동 탐지 오류")

        return text, pending_count

    def _remove_hallucinations(self, text: str) -> str:
        """환각 패턴 제거."""
        for pattern, repl in _HALLUCINATION_PATTERNS:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        # 연속 동일 문장 제거
        sentences = re.split(r"(?<=[.?!])\s+", text)
        if len(sentences) > 1:
            deduped = [sentences[0]]
            for s in sentences[1:]:
                if s != deduped[-1]:
                    deduped.append(s)
            text = " ".join(deduped)
        return text.strip()

    def _check_context(self, text: str, position: int, hints: list[str], window: int = 30) -> bool:
        """문맥 힌트 확인: position 주변 window 범위에 힌트 단어가 있는지."""
        if not hints:
            return True  # 힌트 없으면 무조건 통과
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()
        return any(h.lower() in context for h in hints)

    def _apply_regex(self, text: str, logs: list[CorrectionLog]) -> str:
        """Regex 패턴 교정."""
        for entry in self._regex_entries:
            if not entry.pattern:
                continue
            try:
                new_text = re.sub(entry.pattern, entry.correct, text)
                if new_text != text:
                    logs.append(CorrectionLog(
                        original=text,
                        corrected=new_text,
                        entry_id=entry.id,
                        strategy=MatchStrategy.REGEX,
                    ))
                    text = new_text
            except re.error:
                logger.warning("잘못된 regex 패턴: %s (id=%s)", entry.pattern, entry.id)
        return text

    def _apply_exact(self, text: str, logs: list[CorrectionLog]) -> str:
        """Exact 문자열 교정 (긴 패턴 우선, priority 순)."""
        for entry in self._exact_entries:
            if entry.wrong not in text:
                continue
            # 문맥 힌트 확인
            pos = text.find(entry.wrong)
            if not self._check_context(text, pos, entry.context_hint):
                continue
            new_text = text.replace(entry.wrong, entry.correct)
            if new_text != text:
                logs.append(CorrectionLog(
                    original=entry.wrong,
                    corrected=entry.correct,
                    entry_id=entry.id,
                    strategy=MatchStrategy.EXACT,
                ))
                text = new_text
        return text

    def _apply_phonetic(self, text: str, logs: list[CorrectionLog]) -> str:
        """Phonetic 자모 유사도 교정."""
        for entry in self._phonetic_entries:
            if not entry.context_hint:
                continue  # phonetic은 context_hint 필수
            # 텍스트에서 entry.wrong 길이만큼 슬라이딩 윈도우
            wrong_len = len(entry.wrong)
            if wrong_len == 0:
                continue
            i = 0
            while i <= len(text) - wrong_len:
                candidate = text[i:i + wrong_len]
                sim = jamo_similarity(candidate, entry.wrong)
                if sim >= entry.confidence and candidate != entry.correct:
                    if self._check_context(text, i, entry.context_hint):
                        logs.append(CorrectionLog(
                            original=candidate,
                            corrected=entry.correct,
                            entry_id=entry.id,
                            strategy=MatchStrategy.PHONETIC,
                        ))
                        text = text[:i] + entry.correct + text[i + wrong_len:]
                        i += len(entry.correct)
                        continue
                i += 1
        return text

    # ──────────────────────────────────────────────
    # 통계 API
    # ──────────────────────────────────────────────
    def get_stats(self) -> EngineStats:
        """엔진 전체 통계 반환."""
        global _auto_detector, _learning_manager, _ref_db

        # Tier 1 사전 통계
        all_entries = self._store.get_entries()
        enabled_entries = self._store.get_entries(enabled_only=True)

        stats = EngineStats(
            dict_total=len(all_entries),
            dict_enabled=len(enabled_entries),
            dict_exact=len(self._exact_entries),
            dict_regex=len(self._regex_entries),
            dict_phonetic=len(self._phonetic_entries),
            total_corrections_made=self._total_corrections,
            total_texts_processed=self._total_texts_processed,
        )

        # Tier 2 자동 탐지 통계
        if _auto_detector is not None:
            stats.tier2_enabled = True
            det_stats = _auto_detector.get_stats()
            stats.tier2_total_detections = det_stats.get("total_detections", 0)
            stats.tier2_auto_corrections = det_stats.get("auto_corrections", 0)
            stats.tier2_pending_reviews = det_stats.get("needs_review", 0)
            stats.tier2_cache_size = det_stats.get("cache_size", 0)
            stats.tier2_cache_hits = det_stats.get("cache_hits", 0)

        # 참조 DB 통계
        if _ref_db is not None:
            stats.ref_db_loaded = True
            ref_stats = _ref_db.get_stats()
            stats.ref_db_specialties = ref_stats.get("total_specialties", 0)
            stats.ref_db_total_terms = ref_stats.get("total_terms", 0)
            stats.ref_db_cache_size = ref_stats.get("search_cache_size", 0)

        # 학습 통계
        if _learning_manager is not None:
            learn_stats = _learning_manager.get_stats()
            stats.learning_auto_learned = learn_stats.get("auto_learned", 0)
            stats.learning_llm_verified = learn_stats.get("llm_verified", 0)
            stats.learning_manual = learn_stats.get("manual", 0)
            stats.learning_pending = learn_stats.get("pending_reviews", 0)
            stats.learning_approved = learn_stats.get("approved_reviews", 0)
            stats.learning_rejected = learn_stats.get("rejected_reviews", 0)

        return stats


# --- 싱글톤 인스턴스 관리 ---

_engine: MedicalCorrectionEngine | None = None
_speaker_corrector = None


def init_engine(
    dict_path: Path,
    ref_db_path: Path | None = None,
    openai_api_key: str | None = None,
) -> MedicalCorrectionEngine:
    """엔진 초기화 (Tier 1 + 옵셔널 Tier 2 + 화자교정)."""
    global _engine, _auto_detector, _learning_manager, _ref_db, _speaker_corrector
    store = DictionaryStore(dict_path)
    _engine = MedicalCorrectionEngine(store)

    # Tier 2 초기화 (참조 DB가 있으면)
    if ref_db_path is None:
        ref_db_path = dict_path.parent / "kostom_reference.json"
    if ref_db_path.exists():
        try:
            from app.medterm.reference_db import ReferenceDB
            from app.medterm.auto_detector import AutoDetector
            from app.medterm.learning import LearningManager

            _ref_db = ReferenceDB(ref_db_path)
            _auto_detector = AutoDetector(_ref_db, store)
            _learning_manager = LearningManager(
                store, pending_path=dict_path.parent / "pending_reviews.json"
            )
            logger.info("Tier 2 자동 탐지 활성화: %s", ref_db_path)
        except Exception:
            logger.exception("Tier 2 초기화 실패 (Tier 1만 동작)")
    else:
        logger.info("참조 DB 없음, Tier 1만 동작: %s", ref_db_path)

    # 화자교정 모듈 초기화
    try:
        from app.medterm.speaker_corrector import SpeakerCorrector
        _speaker_corrector = SpeakerCorrector(
            openai_api_key=openai_api_key,
            use_gpt=(openai_api_key is not None),
        )
        gpt_status = "GPT 활성" if openai_api_key else "AB only"
        logger.info("화자교정 모듈 활성화 (%s)", gpt_status)
    except Exception:
        logger.exception("화자교정 모듈 초기화 실패")

    return _engine


def get_engine() -> MedicalCorrectionEngine | None:
    """현재 엔진 인스턴스 반환."""
    return _engine


def get_store() -> DictionaryStore | None:
    """현재 저장소 인스턴스 반환."""
    if _engine is None:
        return None
    return _engine._store


def get_speaker_corrector():
    """현재 화자교정기 반환."""
    return _speaker_corrector


def get_learning_manager():
    """현재 학습 매니저 반환."""
    return _learning_manager


def get_ref_db():
    """현재 참조 DB 반환."""
    return _ref_db
