"""Whisper initial_prompt 동적 생성기."""

from app.medterm.store import DictionaryStore


def build_initial_prompt(store: DictionaryStore | None = None) -> str:
    """사전의 prompt_terms + correct 용어로 Whisper initial_prompt 생성."""
    base = "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다. "

    if store is None:
        return base

    terms: list[str] = []

    # prompt_terms 먼저
    terms.extend(store.get_prompt_terms())

    # 활성 항목의 correct 값 추가
    for entry in store.get_entries(enabled_only=True):
        if entry.correct and entry.correct not in terms:
            terms.append(entry.correct)

    # 중복 제거, 80개 제한 (Whisper 토큰 한계)
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) >= 80:
            break

    return base + ", ".join(unique)
