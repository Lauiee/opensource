"""STT 교정 전/후 처리 시간 벤치마크."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.store import DictionaryStore
from app.medterm.engine import MedicalCorrectionEngine

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")


def load_all_segments() -> list[dict]:
    """모든 타입의 STT 세그먼트를 로드."""
    segments = []
    for type_num in range(1, 11):
        for prefix in ("donkey", "dalpha"):
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
                bracket_end = text.rfind(']')
                if bracket_end >= 0:
                    text = text[:bracket_end + 1]
                data = json.loads(text)
                for item in data:
                    content = item.get("content", "")
                    if content:
                        segments.append({
                            "type": type_num,
                            "prefix": prefix,
                            "role": item.get("role", "?"),
                            "content": content,
                        })
            except Exception:
                pass
    return segments


def main():
    print("=" * 70)
    print("  STT 처리 시간 벤치마크")
    print("=" * 70)

    # === 1단계: STT 결과 로드 (교정 없이 읽기만) ===
    print("\n[1] STT 결과 로드 (교정 적용 없음)...")
    t0 = time.perf_counter()
    segments = load_all_segments()
    t_load = time.perf_counter() - t0
    print(f"    세그먼트 수: {len(segments)}개")
    print(f"    총 글자 수: {sum(len(s['content']) for s in segments):,}자")
    print(f"    소요 시간: {t_load*1000:.1f}ms")

    # === 2단계: 엔진 초기화 시간 ===
    print("\n[2] 교정 엔진 초기화...")
    dict_path = Path(__file__).parent / "data" / "medical_dict.json"

    t0 = time.perf_counter()
    store = DictionaryStore(dict_path)
    engine = MedicalCorrectionEngine(store)
    t_init = time.perf_counter() - t0
    print(f"    사전 항목: {len(store.get_entries())}개")
    print(f"    초기화 시간: {t_init*1000:.1f}ms")

    # === 3단계: 교정 적용 시간 (Tier 1만) ===
    print("\n[3] 전체 세그먼트 교정 (Tier 1 사전 교정)...")
    corrections = 0
    correction_times = []

    t_total_start = time.perf_counter()
    for seg in segments:
        t_seg = time.perf_counter()
        result = engine.correct(seg["content"])
        elapsed = time.perf_counter() - t_seg
        correction_times.append(elapsed)
        if result.logs:
            corrections += 1
    t_total_correct = time.perf_counter() - t_total_start

    avg_per_seg = (t_total_correct / len(segments)) * 1000  # ms
    max_time = max(correction_times) * 1000
    min_time = min(correction_times) * 1000

    print(f"    교정된 세그먼트: {corrections}개 / {len(segments)}개")
    print(f"    총 교정 시간: {t_total_correct*1000:.1f}ms")
    print(f"    세그먼트당 평균: {avg_per_seg:.3f}ms")
    print(f"    세그먼트당 최소: {min_time:.3f}ms")
    print(f"    세그먼트당 최대: {max_time:.3f}ms")

    # === 4단계: 비교 요약 ===
    print("\n" + "=" * 70)
    print("  비교 요약")
    print("=" * 70)
    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  처리 단계              │  소요 시간                 │
  ├──────────────────────────────────────────────────────┤
  │  STT 결과 로드 (파일)   │  {t_load*1000:>8.1f}ms               │
  │  교정 엔진 초기화       │  {t_init*1000:>8.1f}ms               │
  │  전체 교정 처리         │  {t_total_correct*1000:>8.1f}ms               │
  ├──────────────────────────────────────────────────────┤
  │  교정 없이 총 시간      │  {t_load*1000:>8.1f}ms               │
  │  교정 포함 총 시간      │  {(t_load+t_init+t_total_correct)*1000:>8.1f}ms               │
  │  추가 오버헤드          │  {(t_init+t_total_correct)*1000:>8.1f}ms (+{(t_init+t_total_correct)/max(t_load,0.0001)*100:.0f}%)  │
  └──────────────────────────────────────────────────────┘

  세그먼트당 교정 시간: {avg_per_seg:.3f}ms (= {avg_per_seg*1000:.1f}μs)
  → 실시간 STT 후처리에 전혀 부담 없는 수준
""")

    # === 5단계: 타입별 교정 시간 ===
    print("-" * 70)
    print("  타입별 세그먼트 교정 시간")
    print("-" * 70)

    type_stats = {}
    idx = 0
    for seg in segments:
        key = f"type{seg['type']}_{seg['prefix']}"
        if key not in type_stats:
            type_stats[key] = {"count": 0, "total_time": 0, "corrections": 0}
        type_stats[key]["count"] += 1
        type_stats[key]["total_time"] += correction_times[idx]
        result = engine.correct(seg["content"])
        if result.logs:
            type_stats[key]["corrections"] += 1
        idx += 1

    for key in sorted(type_stats.keys()):
        s = type_stats[key]
        avg = (s["total_time"] / s["count"]) * 1000
        print(f"  {key:20s}  세그먼트 {s['count']:>4d}개  평균 {avg:.3f}ms  교정 {s['corrections']}개")

    print()


if __name__ == "__main__":
    main()
