"""AB가 놓친 케이스 분석 → GPT 2차 검증 최적 기준 도출.

GPT가 잡고 AB가 놓친 77건의 공통 특성을 분석하여
GPT로 보낼 기준(threshold)을 결정합니다.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.speaker_corrector import (
    SpeakerCorrector, strategy_a_honorific, strategy_b_content,
    SpeakerSignal
)

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")


def load_segments(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        bracket_end = text.rfind(']')
        if bracket_end >= 0:
            text = text[:bracket_end + 1]
        return json.loads(text)
    except Exception:
        return None


def get_signal_detail(content: str) -> dict:
    """세그먼트의 AB 신호 상세 분석."""
    sig_a = strategy_a_honorific(content)
    sig_b = strategy_b_content(content)

    total_doc = sig_a.doctor_score + sig_b.doctor_score
    total_pat = sig_a.patient_score + sig_b.patient_score
    total = total_doc + total_pat

    return {
        "a_doc": sig_a.doctor_score,
        "a_pat": sig_a.patient_score,
        "b_doc": sig_b.doctor_score,
        "b_pat": sig_b.patient_score,
        "total_doc": total_doc,
        "total_pat": total_pat,
        "confidence": abs(total_doc - total_pat) / total if total > 0 else 0,
        "has_signal": total > 0,
        "signals": sig_a.signals + sig_b.signals,
        "content_len": len(content),
    }


def count_consecutive_same_role(segments: list[dict], idx: int) -> int:
    """현재 인덱스에서 동일 역할이 연속된 횟수."""
    role = segments[idx].get("role", "")
    count = 1
    # 앞으로
    i = idx - 1
    while i >= 0 and segments[i].get("role") == role:
        count += 1
        i -= 1
    # 뒤로
    i = idx + 1
    while i < len(segments) and segments[i].get("role") == role:
        count += 1
        i += 1
    return count


def calc_role_ratio(segments: list[dict]) -> float:
    """의사 세그먼트 비율 (0~1)."""
    doc_count = sum(1 for s in segments if s.get("role") == "원장님")
    return doc_count / max(len(segments), 1)


def main():
    print("=" * 80)
    print("  GPT 2차 검증 기준 분석")
    print("=" * 80)

    corrector = SpeakerCorrector(strategy="ab")

    # 모든 세그먼트 분석
    all_segments_info = []  # (name, seg, ab_changed, signal_detail, consecutive, ratio)

    for type_num in range(1, 11):
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            segments = load_segments(path)
            if not segments:
                continue

            test_segs = segments[:30] if len(segments) > 30 else segments
            results = corrector.correct(test_segs)
            ratio = calc_role_ratio(test_segs)

            for i, (seg, res) in enumerate(zip(test_segs, results)):
                content = seg.get("content", "")
                detail = get_signal_detail(content)
                consec = count_consecutive_same_role(test_segs, i)

                all_segments_info.append({
                    "name": f"Type{type_num}-{label}",
                    "index": seg.get("index", i),
                    "role": seg.get("role", ""),
                    "content": content[:60],
                    "ab_changed": res.changed,
                    "detail": detail,
                    "consecutive": consec,
                    "role_ratio": ratio,
                    "seg_count": len(test_segs),
                })

    # AB가 변경한 것 / 안 한 것 분류
    changed = [s for s in all_segments_info if s["ab_changed"]]
    unchanged = [s for s in all_segments_info if not s["ab_changed"]]

    print(f"\n  전체 세그먼트: {len(all_segments_info)}개")
    print(f"  AB 변경: {len(changed)}개")
    print(f"  AB 미변경: {len(unchanged)}개")

    # ─── 분석 1: 신호 유무 ───
    print("\n" + "─" * 60)
    print("  분석 1: AB 신호(signal) 유무별 분포")
    print("─" * 60)

    unchanged_no_signal = [s for s in unchanged if not s["detail"]["has_signal"]]
    unchanged_has_signal = [s for s in unchanged if s["detail"]["has_signal"]]

    print(f"  AB 미변경 중 신호 없음: {len(unchanged_no_signal)}개 ({len(unchanged_no_signal)/len(unchanged)*100:.0f}%)")
    print(f"  AB 미변경 중 신호 있음: {len(unchanged_has_signal)}개")
    print(f"  → 신호 없는 세그먼트를 GPT로 보내면: {len(unchanged_no_signal)}건 API 호출")

    # ─── 분석 2: 연속 동일 화자 ───
    print("\n" + "─" * 60)
    print("  분석 2: 연속 동일 화자 횟수별 분포")
    print("─" * 60)

    consec_counter = Counter()
    for s in unchanged:
        consec_counter[s["consecutive"]] += 1

    for c in sorted(consec_counter.keys()):
        print(f"  연속 {c}회: {consec_counter[c]}개")

    # 3회 이상 연속
    consec_3plus = [s for s in unchanged if s["consecutive"] >= 3]
    print(f"\n  3회 이상 연속: {len(consec_3plus)}개")

    # ─── 분석 3: 역할 비율 ───
    print("\n" + "─" * 60)
    print("  분석 3: 파일별 역할 비율")
    print("─" * 60)

    seen = set()
    for s in all_segments_info:
        key = s["name"]
        if key in seen:
            continue
        seen.add(key)
        ratio = s["role_ratio"]
        count = s["seg_count"]
        balance = "균형" if 0.3 <= ratio <= 0.7 else "불균형"
        single = "단일화자" if ratio == 0.0 or ratio == 1.0 else ""
        marker = " ← 문제" if ratio < 0.2 or ratio > 0.8 else ""
        print(f"  {key:20s}  의사비율={ratio:.0%}  ({count}개)  {balance}{single}{marker}")

    # ─── 분석 4: 내용 길이별 ───
    print("\n" + "─" * 60)
    print("  분석 4: 내용 길이별 분포 (AB 미변경)")
    print("─" * 60)

    len_bins = [(0, 5, "매우짧음(~5자)"), (5, 15, "짧음(5~15자)"),
                (15, 40, "보통(15~40자)"), (40, 80, "긴편(40~80자)"),
                (80, 9999, "매우긴(80자+)")]

    for lo, hi, label in len_bins:
        count = sum(1 for s in unchanged if lo <= s["detail"]["content_len"] < hi)
        no_sig = sum(1 for s in unchanged
                     if lo <= s["detail"]["content_len"] < hi and not s["detail"]["has_signal"])
        print(f"  {label:15s}: {count:>4}개 (신호없음: {no_sig}개)")

    # ─── 후보 기준들 시뮬레이션 ───
    print("\n" + "=" * 80)
    print("  후보 기준별 시뮬레이션")
    print("=" * 80)

    criteria = [
        ("기준1: 신호 없는 세그먼트 전부",
         lambda s: not s["detail"]["has_signal"]),

        ("기준2: 연속 3회+ 동일화자",
         lambda s: s["consecutive"] >= 3),

        ("기준3: 연속 5회+ 동일화자",
         lambda s: s["consecutive"] >= 5),

        ("기준4: 역할비율 불균형(>80%) 파일의 전체",
         lambda s: s["role_ratio"] < 0.2 or s["role_ratio"] > 0.8),

        ("기준5: 신호없음 + 연속3회+",
         lambda s: not s["detail"]["has_signal"] and s["consecutive"] >= 3),

        ("기준6: 신호없음 OR 연속5회+",
         lambda s: not s["detail"]["has_signal"] or s["consecutive"] >= 5),

        ("기준7: 불균형파일 OR (신호없음+연속3회+)",
         lambda s: (s["role_ratio"] < 0.2 or s["role_ratio"] > 0.8)
                   or (not s["detail"]["has_signal"] and s["consecutive"] >= 3)),

        ("기준8: 파일 단위 - 불균형 파일만 GPT 전체 검증",
         lambda s: s["role_ratio"] < 0.25 or s["role_ratio"] > 0.75),

        ("기준9: 파일 단위 - 불균형 OR 단일화자",
         lambda s: s["role_ratio"] == 0.0 or s["role_ratio"] == 1.0
                   or s["role_ratio"] < 0.25 or s["role_ratio"] > 0.75),
    ]

    print(f"\n  {'기준':<45s} {'대상수':>6} {'전체대비':>8} {'예상GPT호출':>12}")
    print(f"  {'─'*45} {'─'*6} {'─'*8} {'─'*12}")

    for name, fn in criteria:
        targets = [s for s in unchanged if fn(s)]
        pct = len(targets) / max(len(unchanged), 1) * 100

        # 파일 단위 추정 (같은 파일 세그먼트는 1회 호출)
        files = set(s["name"] for s in targets)
        api_calls = len(files) if "파일 단위" in name else len(targets)

        print(f"  {name:<45s} {len(targets):>5}개 {pct:>6.1f}% {api_calls:>10}회")

    # ─── 최종 추천 ───
    print("\n" + "=" * 80)
    print("  최종 추천")
    print("=" * 80)

    print("""
  [추천 A] 파일 단위 전송 (속도+정확도 균형)
  ─────────────────────────────────────────────
  조건: 의사/환자 비율이 75% 이상 한쪽으로 치우친 파일
  방법: 해당 파일 전체를 GPT에 1회 전송
  장점: API 호출 최소화, GPT가 전체 문맥 파악 가능
  단점: 균형 파일 내 개별 오류는 놓칠 수 있음

  [추천 B] 세그먼트 단위 (정확도 최우선)
  ─────────────────────────────────────────────
  조건: AB 신호 없음 + 연속 3회 이상 동일 화자
  방법: 해당 세그먼트 + 앞뒤 2개씩 묶어서 GPT 전송
  장점: 정밀한 교정, 필요한 곳만 검증
  단점: API 호출 수 증가

  [추천 C] 하이브리드 (추천)
  ─────────────────────────────────────────────
  1단계: 불균형 파일(>75% 한쪽) → 파일 통째로 GPT 검증
  2단계: 균형 파일에서 연속 5회+ 동일화자 구간 → 해당 구간만 GPT 검증
  3단계: AB confidence > 0.5인 변경은 그대로 적용
  장점: 최소 API 호출로 최대 커버리지
""")


if __name__ == "__main__":
    main()
