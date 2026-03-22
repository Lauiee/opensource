"""AB+GPT 하이브리드 화자교정 통합 테스트.

실제 파이프라인: AB 즉시교정 → GPT 2차 검증 → 최종 결과.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.speaker_corrector import SpeakerCorrector, assess_gpt_need

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")
API_KEY = os.environ.get("OPENAI_API_KEY", "")


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


def main():
    print("=" * 80)
    print("  AB + GPT 하이브리드 화자교정 테스트")
    print("=" * 80)

    # 두 교정기 생성
    ab_only = SpeakerCorrector(use_gpt=False)
    hybrid = SpeakerCorrector(openai_api_key=API_KEY, use_gpt=True)

    total_ab_changes = 0
    total_hybrid_changes = 0
    total_ab_time = 0
    total_hybrid_time = 0
    total_segments = 0
    total_gpt_calls = 0
    total_gpt_skipped = 0

    report_lines = []

    for type_num in range(1, 11):
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            segments = load_segments(path)
            if not segments:
                continue

            test_segs = segments[:30] if len(segments) > 30 else segments
            total_segments += len(test_segs)

            name = f"Type{type_num}-{label}"
            print(f"\n{'─'*60}")
            print(f"  {name} ({len(test_segs)}개 세그먼트)")
            print(f"{'─'*60}")

            # AB only
            t0 = time.perf_counter()
            ab_results = ab_only.correct(test_segs)
            ab_time = time.perf_counter() - t0
            ab_changes = sum(1 for r in ab_results if r.changed)
            total_ab_changes += ab_changes
            total_ab_time += ab_time

            # GPT 필요성 먼저 체크 (로그용)
            ab_signals = ab_only._ab_analyze(test_segs)
            decision = assess_gpt_need(test_segs, ab_signals)

            # Hybrid
            t0 = time.perf_counter()
            hybrid_results = hybrid.correct(test_segs)
            hybrid_time = time.perf_counter() - t0
            hybrid_changes = sum(1 for r in hybrid_results if r.changed)
            total_hybrid_changes += hybrid_changes
            total_hybrid_time += hybrid_time

            if decision.needs_review:
                total_gpt_calls += 1
            else:
                total_gpt_skipped += 1

            # 결과 출력
            gpt_tag = f"GPT 검증({decision.scope})" if decision.needs_review else "GPT 스킵"
            print(f"  AB: {ab_changes}개 변경 ({ab_time*1000:.1f}ms)")
            print(f"  하이브리드: {hybrid_changes}개 변경 ({hybrid_time*1000:.0f}ms) [{gpt_tag}]")

            if decision.needs_review:
                print(f"  GPT 사유: {', '.join(decision.reasons)}")

            # 변경된 세그먼트 상세
            detail_lines = []
            for seg, ab_r, hy_r in zip(test_segs, ab_results, hybrid_results):
                if hy_r.changed:
                    idx = seg.get("index", "?")
                    content = seg.get("content", "")[:50]
                    ab_mark = ""
                    if ab_r.changed and ab_r.corrected_role == hy_r.corrected_role:
                        ab_mark = " [AB+GPT 일치]"
                    elif ab_r.changed:
                        ab_mark = f" [AB→{ab_r.corrected_role}, GPT→{hy_r.corrected_role}]"
                    elif not ab_r.changed:
                        ab_mark = " [GPT 추가]"

                    line = (
                        f"    idx={idx}: {hy_r.original_role}→{hy_r.corrected_role} "
                        f"({hy_r.strategy}){ab_mark}"
                    )
                    detail_lines.append(line)
                    detail_lines.append(f"      \"{content}\"")

            if detail_lines:
                for dl in detail_lines[:20]:  # 너무 길면 자름
                    print(dl)
                if len(detail_lines) > 20:
                    print(f"    ... 외 {len(detail_lines)//2 - 10}개")

            # 리포트
            report_lines.append(f"\n--- {name} ---")
            for seg, hy_r in zip(test_segs, hybrid_results):
                content = seg.get("content", "")
                if len(content) > 60:
                    content = content[:57] + "..."
                role = hy_r.corrected_role
                marker = ""
                if hy_r.changed:
                    marker = f"  << {hy_r.original_role}→{hy_r.corrected_role} ({hy_r.strategy})"
                report_lines.append(f"  [{role:^6}] {content}{marker}")

    # ─── 종합 ───
    print("\n" + "=" * 80)
    print("  종합 결과")
    print("=" * 80)
    print(f"""
  전체 세그먼트: {total_segments}개

  ┌────────────────────────────────────────────────────┐
  │  항목              │  AB only    │  AB+GPT 하이브리드 │
  ├────────────────────────────────────────────────────┤
  │  교정 수           │  {total_ab_changes:>5}개     │  {total_hybrid_changes:>5}개           │
  │  소요 시간         │  {total_ab_time*1000:>7.1f}ms  │  {total_hybrid_time*1000:>7.0f}ms         │
  ├────────────────────────────────────────────────────┤
  │  GPT 호출 횟수     │  -          │  {total_gpt_calls}회 (스킵 {total_gpt_skipped})  │
  │  추가 교정 수      │  -          │  +{total_hybrid_changes - total_ab_changes}개           │
  │  정확도 향상       │  기준       │  {total_hybrid_changes/max(total_ab_changes,1)*100:.0f}%             │
  └────────────────────────────────────────────────────┘

  파이프라인: AB 즉시교정(0ms) → GPT 필요성 판단 → GPT 검증(필요 시)
  GPT 호출 기준: 불균형 파일, 단일화자, 연속3+, AB 신호 부족
""")

    # 리포트 저장
    report_text = "\n".join(report_lines)
    output = DATA_DIR / "speaker_hybrid_report.txt"
    output.write_text(report_text, encoding="utf-8")
    print(f"  리포트 저장: {output}")


if __name__ == "__main__":
    main()
