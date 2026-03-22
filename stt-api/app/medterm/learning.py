"""학습 루프: 확정된 교정을 Tier 1 사전에 자동 추가."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.medterm.auto_detector import DetectionResult
from app.medterm.models import DictEntry, DictEntryCreate, MatchStrategy
from app.medterm.store import DictionaryStore

logger = logging.getLogger(__name__)


class PendingReview:
    """GPT 검증 대기 항목."""

    def __init__(
        self,
        original_word: str,
        candidate_term: str,
        similarity: float,
        context_sentence: str,
        specialty: str,
    ):
        self.id = uuid4().hex[:12]
        self.original_word = original_word
        self.candidate_term = candidate_term
        self.similarity = similarity
        self.context_sentence = context_sentence
        self.specialty = specialty
        self.status = "pending"  # "pending" | "approved" | "rejected"
        self.llm_result: dict | None = None
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_word": self.original_word,
            "candidate_term": self.candidate_term,
            "similarity": self.similarity,
            "context_sentence": self.context_sentence,
            "specialty": self.specialty,
            "status": self.status,
            "llm_result": self.llm_result,
            "created_at": self.created_at,
        }


class LearningManager:
    """학습 루프 관리."""

    def __init__(self, store: DictionaryStore, pending_path: Path):
        self._store = store
        self._pending_path = pending_path
        self._pending: list[PendingReview] = []
        self._load_pending()

    def _load_pending(self) -> None:
        if self._pending_path.exists():
            try:
                data = json.loads(self._pending_path.read_text(encoding="utf-8"))
                self._pending = []
                for item in data:
                    pr = PendingReview(
                        original_word=item["original_word"],
                        candidate_term=item["candidate_term"],
                        similarity=item["similarity"],
                        context_sentence=item["context_sentence"],
                        specialty=item["specialty"],
                    )
                    pr.id = item["id"]
                    pr.status = item["status"]
                    pr.llm_result = item.get("llm_result")
                    pr.created_at = item["created_at"]
                    self._pending.append(pr)
            except Exception:
                logger.exception("pending_reviews 로드 실패")

    def _save_pending(self) -> None:
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        data = [pr.to_dict() for pr in self._pending]
        self._pending_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def auto_learn(self, detection: DetectionResult, context: str = "") -> DictEntry | None:
        """Tier 2 고확신 결과를 사전에 즉시 추가.

        보호 검증:
        - 보호 단어(Safe Words)이면 학습 거부
        - 조사만 다른 경우 학습 거부
        - 원본이 후보의 부분 문자열이면 학습 거부 (정상 단어 가능성 높음)
        """
        from app.medterm.auto_detector import SAFE_WORDS, _strip_particle, _is_particle_only_diff

        best = detection.best_candidate()
        if not best:
            return None

        # ★ 보호 단어 체크 — 일반 한국어 단어를 사전에 추가하지 않음
        if detection.word in SAFE_WORDS:
            logger.debug("자동 학습 거부 (보호 단어): '%s'", detection.word)
            return None

        # ★ 조사만 다른 경우 학습 거부
        if _is_particle_only_diff(detection.word, best["term"]):
            logger.debug("자동 학습 거부 (조사 차이): '%s' → '%s'", detection.word, best["term"])
            return None

        # ★ 원본이 후보에 포함되거나, 후보가 원본에 포함되면 거부
        #    (예: "비타민" → "비타민D" — 그냥 접미사 추가일 뿐)
        if detection.word in best["term"] or best["term"] in detection.word:
            if abs(len(detection.word) - len(best["term"])) <= 2:
                logger.debug("자동 학습 거부 (부분 문자열): '%s' → '%s'", detection.word, best["term"])
                return None

        # 이미 등록된 wrong인지 확인
        existing = self._store.get_entries(search=detection.word)
        if any(e.wrong == detection.word for e in existing):
            return None

        entry_data = DictEntryCreate(
            wrong=detection.word,
            correct=best["term"],
            category=best.get("specialty", "일반"),
            strategy=MatchStrategy.EXACT,
            priority=50 + len(detection.word),
            enabled=True,
            notes=f"자동 학습 (유사도: {best['similarity']:.2f}, source=auto)",
        )
        entry = self._store.add_entry(entry_data)
        logger.info("자동 학습: '%s' → '%s' (%.2f)", detection.word, best["term"], best["similarity"])
        return entry

    def add_pending_review(self, detection: DetectionResult, context: str = "") -> PendingReview:
        """불확실한 후보를 검증 대기열에 추가."""
        best = detection.best_candidate()
        if not best:
            raise ValueError("후보 없음")

        pr = PendingReview(
            original_word=detection.word,
            candidate_term=best["term"],
            similarity=best["similarity"],
            context_sentence=context[:200],
            specialty=best.get("specialty", "일반"),
        )
        self._pending.append(pr)
        self._save_pending()
        logger.info("검증 대기: '%s' → '%s' (%.2f)", detection.word, best["term"], best["similarity"])
        return pr

    def approve_review(self, review_id: str, verified_by: str = "human") -> DictEntry | None:
        """검증 대기 항목 승인 → 사전 추가."""
        for pr in self._pending:
            if pr.id == review_id and pr.status == "pending":
                pr.status = "approved"
                entry_data = DictEntryCreate(
                    wrong=pr.original_word,
                    correct=pr.candidate_term,
                    category=pr.specialty,
                    strategy=MatchStrategy.EXACT,
                    priority=50 + len(pr.original_word),
                    enabled=True,
                    notes=f"검증 승인 (by={verified_by}, 유사도: {pr.similarity:.2f})",
                )
                entry = self._store.add_entry(entry_data)
                self._save_pending()
                logger.info("검증 승인: '%s' → '%s'", pr.original_word, pr.candidate_term)
                return entry
        return None

    def reject_review(self, review_id: str) -> bool:
        """검증 대기 항목 기각."""
        for pr in self._pending:
            if pr.id == review_id and pr.status == "pending":
                pr.status = "rejected"
                self._save_pending()
                logger.info("검증 기각: '%s'", pr.original_word)
                return True
        return False

    def get_pending_reviews(self, status: str | None = None) -> list[dict]:
        """검증 대기 목록."""
        reviews = self._pending
        if status:
            reviews = [pr for pr in reviews if pr.status == status]
        return [pr.to_dict() for pr in reviews]

    def get_stats(self) -> dict:
        """학습 통계."""
        entries = self._store.get_entries()
        auto_count = sum(1 for e in entries if "source=auto" in e.notes)
        llm_count = sum(1 for e in entries if "llm" in e.notes.lower())
        manual_count = len(entries) - auto_count - llm_count
        return {
            "total_entries": len(entries),
            "auto_learned": auto_count,
            "llm_verified": llm_count,
            "manual": manual_count,
            "pending_reviews": len([pr for pr in self._pending if pr.status == "pending"]),
            "approved_reviews": len([pr for pr in self._pending if pr.status == "approved"]),
            "rejected_reviews": len([pr for pr in self._pending if pr.status == "rejected"]),
        }
