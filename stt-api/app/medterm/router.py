"""의료 사전 CRUD API 라우터."""

import csv
import io
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.medterm.engine import get_engine, get_store
from app.medterm.models import (
    CorrectionResult,
    DictEntry,
    DictEntryCreate,
    DictEntryUpdate,
    DictStats,
    ImportRequest,
    MatchStrategy,
    TestRequest,
)
from app.medterm.prompt_builder import build_initial_prompt

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_store():
    store = get_store()
    if store is None:
        raise HTTPException(503, "의료 사전 엔진이 초기화되지 않았습니다")
    return store


def _require_engine():
    engine = get_engine()
    if engine is None:
        raise HTTPException(503, "의료 사전 엔진이 초기화되지 않았습니다")
    return engine


# --- 용어 CRUD ---

@router.get("/entries", response_model=list[DictEntry])
def list_entries(
    category: str | None = None,
    search: str | None = None,
    enabled_only: bool = False,
):
    store = _require_store()
    return store.get_entries(category=category, search=search, enabled_only=enabled_only)


@router.post("/entries", response_model=DictEntry, status_code=201)
def create_entry(data: DictEntryCreate):
    store = _require_store()
    entry = store.add_entry(data)
    engine = get_engine()
    if engine:
        engine.reload()
    return entry


@router.put("/entries/{entry_id}", response_model=DictEntry)
def update_entry(entry_id: str, data: DictEntryUpdate):
    store = _require_store()
    entry = store.update_entry(entry_id, data)
    if entry is None:
        raise HTTPException(404, f"항목을 찾을 수 없습니다: {entry_id}")
    engine = get_engine()
    if engine:
        engine.reload()
    return entry


@router.delete("/entries/{entry_id}")
def delete_entry(entry_id: str):
    store = _require_store()
    if not store.delete_entry(entry_id):
        raise HTTPException(404, f"항목을 찾을 수 없습니다: {entry_id}")
    engine = get_engine()
    if engine:
        engine.reload()
    return {"ok": True}


# --- 카테고리 ---

@router.get("/categories", response_model=list[str])
def list_categories():
    store = _require_store()
    return store.get_categories()


# --- 통계 ---

@router.get("/stats", response_model=DictStats)
def get_stats():
    store = _require_store()
    entries = store.get_entries()
    categories: dict[str, int] = {}
    strategies: dict[str, int] = {}
    enabled = 0
    for e in entries:
        categories[e.category] = categories.get(e.category, 0) + 1
        strategies[e.strategy.value] = strategies.get(e.strategy.value, 0) + 1
        if e.enabled:
            enabled += 1
    return DictStats(
        total_entries=len(entries),
        enabled_entries=enabled,
        disabled_entries=len(entries) - enabled,
        categories=categories,
        strategies=strategies,
    )


# --- 핫 리로드 ---

@router.post("/reload")
def reload_dict():
    engine = _require_engine()
    count = engine.reload()
    return {"ok": True, "entries_loaded": count}


# --- Whisper Prompt ---

@router.get("/prompt")
def get_prompt():
    store = _require_store()
    prompt = build_initial_prompt(store)
    return {"prompt": prompt, "length": len(prompt)}


# --- 텍스트 교정 테스트 ---

@router.post("/test", response_model=CorrectionResult)
def test_correction(req: TestRequest):
    engine = _require_engine()
    return engine.correct(req.text)


# --- 가져오기/내보내기 ---

@router.post("/import")
def import_entries(req: ImportRequest):
    store = _require_store()
    count = store.import_entries(req.entries)
    engine = get_engine()
    if engine:
        engine.reload()
    return {"ok": True, "imported": count}


@router.post("/import/csv")
async def import_csv(file: UploadFile = File(...)):
    store = _require_store()
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    entries: list[DictEntryCreate] = []
    for row in reader:
        entries.append(DictEntryCreate(
            wrong=row.get("wrong", ""),
            correct=row.get("correct", ""),
            category=row.get("category", "일반"),
            strategy=MatchStrategy(row.get("strategy", "exact")),
            priority=int(row.get("priority", 50)),
            enabled=row.get("enabled", "true").lower() == "true",
            notes=row.get("notes", ""),
        ))
    count = store.import_entries(entries)
    engine = get_engine()
    if engine:
        engine.reload()
    return {"ok": True, "imported": count}


@router.get("/export")
def export_json():
    store = _require_store()
    return store.export_entries()


@router.get("/export/csv")
def export_csv():
    store = _require_store()
    entries = store.export_entries()
    output = io.StringIO()
    if entries:
        writer = csv.DictWriter(output, fieldnames=entries[0].keys())
        writer.writeheader()
        for e in entries:
            # context_hint는 리스트이므로 문자열로 변환
            row = {**e, "context_hint": "|".join(e.get("context_hint", []))}
            writer.writerow(row)
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=medical_dict.csv"},
    )
