"""KOSTOM 기반 표준 의학 용어 참조 DB — 진료과별 검색 + 캐시 지원."""

import json
import logging
from collections import OrderedDict
from pathlib import Path

from app.medterm.phonetic import jamo_similarity

logger = logging.getLogger(__name__)

# 기본 캐시 크기 설정
_DEFAULT_CACHE_SIZE = 1024


class _LRUCache:
    """간단한 LRU 캐시 구현 (OrderedDict 기반)."""

    def __init__(self, maxsize: int = _DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key):
        """캐시에서 값 조회. 없으면 None 반환."""
        if key in self._cache:
            self._hits += 1
            # LRU: 최근 접근한 항목을 뒤로 이동
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key, value):
        """캐시에 값 저장."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self._maxsize:
                # 가장 오래된 항목 제거
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self):
        """캐시 초기화."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


class ReferenceDB:
    """진료과별 표준 의학 용어 인덱스 — 캐시 및 진료과별 검색 지원."""

    def __init__(self, path: Path, cache_size: int = _DEFAULT_CACHE_SIZE):
        self._path = path
        self._specialties: dict[str, dict] = {}
        self._all_terms: set[str] = set()
        # 진료과별 용어 인덱스 (빠른 접근용)
        self._specialty_terms: dict[str, list[str]] = {}
        # 검색 결과 캐시
        self._search_cache = _LRUCache(maxsize=cache_size)
        # has_exact 캐시 (자주 조회되는 용어)
        self._exact_cache = _LRUCache(maxsize=cache_size)
        self._load()

    def _load(self) -> None:
        """참조 DB 파일 로드 및 인덱스 구축."""
        if not self._path.exists():
            logger.warning("참조 DB 파일 없음: %s", self._path)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._specialties = raw.get("specialties", {})
            self._all_terms = set()
            self._specialty_terms = {}

            for spec_name, spec_data in self._specialties.items():
                terms = spec_data.get("terms", [])
                self._specialty_terms[spec_name] = terms
                for term in terms:
                    self._all_terms.add(term)

            logger.info(
                "참조 DB 로드: %d개 진료과, %d개 용어",
                len(self._specialties),
                len(self._all_terms),
            )
        except Exception:
            logger.exception("참조 DB 로드 실패")

    def reload(self) -> None:
        """참조 DB 재로드 및 캐시 초기화."""
        self._search_cache.clear()
        self._exact_cache.clear()
        self._load()
        logger.info("참조 DB 재로드 완료")

    def get_specialties(self) -> list[str]:
        """전체 진료과 목록 반환."""
        return list(self._specialties.keys())

    def get_keywords(self, specialty: str) -> list[str]:
        """특정 진료과의 키워드 목록 반환."""
        spec = self._specialties.get(specialty, {})
        return spec.get("keywords", [])

    def get_terms(self, specialty: str) -> list[str]:
        """특정 진료과의 용어 목록 반환."""
        return self._specialty_terms.get(specialty, [])

    def get_all_terms(self) -> set[str]:
        """전체 용어 집합 반환."""
        return self._all_terms.copy()

    def has_exact(self, word: str) -> bool:
        """표준 용어에 정확히 존재하는지 확인 (캐시 지원)."""
        # 캐시 확인
        cached = self._exact_cache.get(word)
        if cached is not None:
            return cached

        result = word in self._all_terms
        self._exact_cache.put(word, result)
        return result

    def search(
        self,
        word: str,
        specialty: str | None = None,
        top_n: int = 3,
    ) -> list[dict]:
        """자모 유사도 기반 후보 검색 (캐시 지원).

        Args:
            word: 검색할 단어
            specialty: 특정 진료과로 범위 한정 (None이면 전체 검색)
            top_n: 반환할 최대 후보 수

        Returns:
            [{"term": str, "similarity": float, "specialty": str}]
        """
        if len(word) < 2:
            return []

        # 캐시 키 생성
        cache_key = f"{word}|{specialty or 'ALL'}|{top_n}"
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        # 검색 대상 용어 수집
        terms_to_search: list[tuple[str, str]] = []
        if specialty and specialty in self._specialty_terms:
            for t in self._specialty_terms[specialty]:
                terms_to_search.append((t, specialty))
        else:
            for spec_name, terms in self._specialty_terms.items():
                for t in terms:
                    terms_to_search.append((t, spec_name))

        candidates: list[dict] = []
        for term, spec_name in terms_to_search:
            # 길이 차이가 너무 크면 스킵 (속도 최적화)
            if abs(len(term) - len(word)) > max(len(word) // 2, 2):
                continue
            sim = jamo_similarity(word, term)
            if sim >= 0.60:
                candidates.append({
                    "term": term,
                    "similarity": sim,
                    "specialty": spec_name,
                })

        candidates.sort(key=lambda c: -c["similarity"])
        result = candidates[:top_n]

        # 캐시에 저장
        self._search_cache.put(cache_key, result)
        return result

    def search_by_specialty(
        self,
        word: str,
        specialties: list[str],
        top_n: int = 3,
    ) -> list[dict]:
        """여러 진료과에서 우선순위 기반 검색.

        지정된 진료과 순서대로 검색하여,
        상위 진료과의 결과가 우선적으로 반환됨.

        Args:
            word: 검색할 단어
            specialties: 진료과 목록 (우선순위 순)
            top_n: 반환할 최대 후보 수

        Returns:
            [{"term": str, "similarity": float, "specialty": str}]
        """
        if len(word) < 2 or not specialties:
            return []

        all_candidates: list[dict] = []
        for spec in specialties:
            spec_results = self.search(word, specialty=spec, top_n=top_n)
            # 진료과 우선순위 보너스 (첫 번째 진료과에 약간의 가산점)
            priority_bonus = 0.0
            if spec == specialties[0]:
                priority_bonus = 0.02  # 주요 진료과에 2% 보너스
            for r in spec_results:
                r["adjusted_similarity"] = r["similarity"] + priority_bonus
                all_candidates.append(r)

        # 조정된 유사도로 정렬
        all_candidates.sort(key=lambda c: -c.get("adjusted_similarity", c["similarity"]))

        # adjusted_similarity 필드 제거 (외부에 노출 불필요)
        result = []
        for c in all_candidates[:top_n]:
            c.pop("adjusted_similarity", None)
            result.append(c)

        return result

    def get_stats(self) -> dict:
        """참조 DB 통계 반환."""
        specialty_counts = {}
        for spec_name, terms in self._specialty_terms.items():
            specialty_counts[spec_name] = len(terms)

        return {
            "loaded": bool(self._all_terms),
            "total_specialties": len(self._specialties),
            "total_terms": len(self._all_terms),
            "specialty_term_counts": specialty_counts,
            "search_cache_size": self._search_cache.size,
            "search_cache_hits": self._search_cache.hits,
            "search_cache_misses": self._search_cache.misses,
            "exact_cache_size": self._exact_cache.size,
            "exact_cache_hits": self._exact_cache.hits,
            "exact_cache_misses": self._exact_cache.misses,
        }
