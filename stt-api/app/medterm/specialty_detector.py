"""전사 텍스트에서 진료과를 자동 추정."""

from app.medterm.reference_db import ReferenceDB


def detect_specialty(text: str, ref_db: ReferenceDB) -> list[tuple[str, float]]:
    """텍스트 내 키워드 빈도 기반 진료과 추정.
    Returns: [(진료과, 점수)] 점수 내림차순, 0.0 이상만.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for specialty in ref_db.get_specialties():
        keywords = ref_db.get_keywords(specialty)
        if not keywords:
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[specialty] = count / len(keywords)

    if not scores:
        return []

    # 정규화
    max_score = max(scores.values())
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    result = sorted(scores.items(), key=lambda x: -x[1])
    return [(name, score) for name, score in result if score > 0.0]
