"""기존 STT 결과 vs 교정 후 결과 비교 테스트."""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.engine import init_engine

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")


def load_stt_texts(type_num: int) -> dict[str, list[str]]:
    """donkey/dalpha STT 결과를 로드."""
    result = {}
    for prefix in ("donkey", "dalpha"):
        path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            texts = [item["content"] for item in data if "content" in item]
            result[prefix] = texts
        except Exception:
            pass
    return result


def main():
    # 엔진 초기화
    dict_path = Path(__file__).parent / "data" / "medical_dict.json"
    engine = init_engine(dict_path)

    print("=" * 80)
    print("의료 용어 교정 테스트 — 기존 STT 결과 vs 교정 후")
    print("=" * 80)

    for type_num in range(1, 11):
        texts = load_stt_texts(type_num)
        if "donkey" not in texts:
            continue

        print(f"\n{'─' * 80}")
        print(f"  TYPE {type_num} (Donkey STT)")
        print(f"{'─' * 80}")

        corrections_found = 0
        for i, text in enumerate(texts["donkey"]):
            result = engine.correct(text)
            if result.logs:
                corrections_found += 1
                print(f"\n  [{i}] 원본: {text}")
                print(f"       교정: {result.text}")
                for log in result.logs:
                    print(f"       ├─ '{log.original}' → '{log.corrected}' ({log.strategy})")

        if corrections_found == 0:
            print("  (교정 사항 없음)")
        else:
            print(f"\n  → {corrections_found}개 세그먼트에서 교정 발생")

    # D-Alpha 결과도 비교
    print(f"\n{'=' * 80}")
    print("D-Alpha STT에도 교정 적용")
    print(f"{'=' * 80}")

    for type_num in range(1, 11):
        texts = load_stt_texts(type_num)
        if "dalpha" not in texts:
            continue

        corrections_found = 0
        for i, text in enumerate(texts["dalpha"]):
            result = engine.correct(text)
            if result.logs:
                corrections_found += 1

        if corrections_found > 0:
            print(f"\n  TYPE {type_num} (D-Alpha): {corrections_found}개 세그먼트 교정")
            for i, text in enumerate(texts["dalpha"]):
                result = engine.correct(text)
                if result.logs:
                    print(f"    [{i}] 원본: {text[:80]}...")
                    print(f"         교정: {result.text[:80]}...")
                    for log in result.logs:
                        print(f"         ├─ '{log.original}' → '{log.corrected}'")


if __name__ == "__main__":
    main()
