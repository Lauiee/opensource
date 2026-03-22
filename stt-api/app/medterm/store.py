"""JSON 파일 기반 의료 사전 저장소."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.medterm.models import DictEntry, DictEntryCreate, DictEntryUpdate, MedicalDict

logger = logging.getLogger(__name__)


class DictionaryStore:
    """JSON 파일 하나로 범용 의료 사전을 관리."""

    def __init__(self, path: Path):
        self._path = path
        self._dict: MedicalDict = MedicalDict()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._dict = MedicalDict(**raw)
                logger.info("의료 사전 로드: %d개 항목 (%s)", len(self._dict.entries), self._path)
            except Exception:
                logger.exception("의료 사전 로드 실패, 빈 사전으로 시작")
                self._dict = MedicalDict()
        else:
            logger.info("의료 사전 파일 없음, 빈 사전 생성: %s", self._path)
            self._save()

    def _save(self) -> None:
        self._dict.updated_at = datetime.now(timezone.utc)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self._dict.model_dump_json())
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reload(self) -> int:
        """파일에서 다시 로드. 항목 수 반환."""
        self._load()
        return len(self._dict.entries)

    # --- 조회 ---

    def get_all(self) -> MedicalDict:
        return self._dict

    def get_entries(
        self,
        category: str | None = None,
        search: str | None = None,
        enabled_only: bool = False,
    ) -> list[DictEntry]:
        entries = self._dict.entries
        if category:
            entries = [e for e in entries if e.category == category]
        if enabled_only:
            entries = [e for e in entries if e.enabled]
        if search:
            q = search.lower()
            entries = [
                e for e in entries
                if q in e.wrong.lower() or q in e.correct.lower() or q in e.notes.lower()
            ]
        return entries

    def get_entry(self, entry_id: str) -> DictEntry | None:
        for e in self._dict.entries:
            if e.id == entry_id:
                return e
        return None

    def get_categories(self) -> list[str]:
        return sorted({e.category for e in self._dict.entries})

    def get_prompt_terms(self) -> list[str]:
        return self._dict.prompt_terms

    # --- 생성/수정/삭제 ---

    def add_entry(self, data: DictEntryCreate) -> DictEntry:
        entry = DictEntry(**data.model_dump())
        self._dict.entries.append(entry)
        self._save()
        return entry

    def update_entry(self, entry_id: str, data: DictEntryUpdate) -> DictEntry | None:
        for i, e in enumerate(self._dict.entries):
            if e.id == entry_id:
                updates = data.model_dump(exclude_none=True)
                updated = e.model_copy(update=updates)
                self._dict.entries[i] = updated
                self._save()
                return updated
        return None

    def delete_entry(self, entry_id: str) -> bool:
        before = len(self._dict.entries)
        self._dict.entries = [e for e in self._dict.entries if e.id != entry_id]
        if len(self._dict.entries) < before:
            self._save()
            return True
        return False

    # --- 대량 가져오기/내보내기 ---

    def import_entries(self, entries: list[DictEntryCreate]) -> int:
        count = 0
        for data in entries:
            entry = DictEntry(**data.model_dump())
            self._dict.entries.append(entry)
            count += 1
        if count > 0:
            self._save()
        return count

    def export_entries(self) -> list[dict]:
        return [e.model_dump() for e in self._dict.entries]

    # --- Prompt terms 관리 ---

    def set_prompt_terms(self, terms: list[str]) -> None:
        self._dict.prompt_terms = terms
        self._save()
