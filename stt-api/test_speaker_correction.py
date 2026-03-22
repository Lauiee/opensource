"""화자분리 교정 전략 비교 벤치마크.

전략 A(호칭), B(내용), AB(호칭+내용), C(스왑), Combined(A+B+C+D)를
각 타입별로 적용하고 결과를 비교합니다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.speaker_corrector import SpeakerCorrector, strategy_a_honorific, strategy_b_content

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")

STRATEGIES = ["a", "b", "ab", "c", "combined"]


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


def analyze_problems(segments: list[dict]) -> list[str]:
    """수동으로 발견 가능한 문제점들."""
    problems = []
    roles = set(s.get("role") for s in segments)

    # 단일 화자
    if len(roles) == 1 and len(segments) > 3:
        problems.append(f"[단일화자] 전체 {len(segments)}개가 모두 '{list(roles)[0]}'")

    for seg in segments:
        content = seg.get("content", "")
        role = seg.get("role", "")
        idx = seg.get("index", "?")

        # 환자인데 의사 어투
        if role == "환자":
            if "환자분" in content:
                problems.append(f"  idx={idx}: 환자인데 '환자분' 사용 → 의사 발화 추정")
            if any(p in content for p in ["해 드리", "해드리", "놔 드리", "놔드리", "드릴게", "드리겠"]):
                problems.append(f"  idx={idx}: 환자인데 '~해드리다' 어투 → 의사 발화 추정")
            if "어머님" in content or "어머니" in content:
                problems.append(f"  idx={idx}: 환자인데 '어머님' 호칭 → 의사 발화 추정")

        # 의사인데 환자 어투
        if role == "원장님":
            if content.startswith("선생님") or "원장님" in content:
                problems.append(f"  idx={idx}: 원장님인데 '선생님/원장님' 호칭 → 환자 발화 추정")
            if any(p in content for p in ["살고 싶", "죽고 싶", "한심해서"]):
                problems.append(f"  idx={idx}: 원장님인데 감정 호소 → 환자 발화 추정")
            if "잘 부탁" in content or "잘부탁" in content:
                problems.append(f"  idx={idx}: 원장님인데 '잘 부탁' → 환자 발화 추정")
            if content.strip() in ["네", "네.", "예", "예.", "아.", "네, 감사합니다.", "감사합니다", "감사합니다."]:
                problems.append(f"  idx={idx}: 원장님인데 단순응답 '{content.strip()}' → 환자 가능성")

    return problems


def format_comparison(segments: list[dict], results, strategy_name: str) -> list[str]:
    """교정 전후 비교 포맷."""
    lines = []
    changed_count = sum(1 for r in results if r.changed)

    if changed_count == 0:
        lines.append(f"  [{strategy_name}] 변경 없음")
        return lines

    lines.append(f"  [{strategy_name}] {changed_count}개 변경:")
    for seg, res in zip(segments, results):
        if res.changed:
            idx = seg.get("index", "?")
            content = seg.get("content", "")[:50]
            lines.append(
                f"    idx={idx}: {res.original_role} → {res.corrected_role} "
                f"({res.strategy}, conf={res.confidence:.2f})"
            )
            if res.signals:
                lines.append(f"           근거: {', '.join(res.signals[:3])}")
    return lines


def evaluate_correction_quality(segments: list[dict], results) -> dict:
    """교정 품질 평가 (규칙 기반 자동 평가).

    교정 후 세그먼트에 대해 호칭/내용 분석 점수를 재계산하여
    교정이 올바른 방향인지 평가.
    """
    improved = 0
    worsened = 0
    neutral = 0
    total_changed = 0

    for seg, res in zip(segments, results):
        if not res.changed:
            continue
        total_changed += 1
        content = seg.get("content", "")

        # A+B 신호 합산
        sig_a = strategy_a_honorific(content)
        sig_b = strategy_b_content(content)
        doctor_score = sig_a.doctor_score + sig_b.doctor_score
        patient_score = sig_a.patient_score + sig_b.patient_score

        # 교정 후 역할이 신호와 일치하는지
        if res.corrected_role == "원장님":
            if doctor_score > patient_score:
                improved += 1
            elif patient_score > doctor_score:
                worsened += 1
            else:
                neutral += 1
        else:
            if patient_score > doctor_score:
                improved += 1
            elif doctor_score > patient_score:
                worsened += 1
            else:
                neutral += 1

    return {
        "total_changed": total_changed,
        "improved": improved,
        "worsened": worsened,
        "neutral": neutral,
        "accuracy": improved / max(total_changed, 1),
    }


def main():
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  화자분리 교정 전략 비교 리포트")
    report_lines.append("=" * 80)

    # 전략별 종합 점수
    strategy_scores = {s: {"improved": 0, "worsened": 0, "neutral": 0, "changed": 0}
                       for s in STRATEGIES}

    for type_num in range(1, 11):
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            segments = load_segments(path)
            if not segments:
                continue

            report_lines.append("")
            report_lines.append(f"{'─'*80}")
            report_lines.append(f"  Type {type_num} - {label} ({len(segments)}개 세그먼트)")
            report_lines.append(f"{'─'*80}")

            # 문제 분석
            problems = analyze_problems(segments)
            if problems:
                report_lines.append("  ※ 감지된 문제:")
                report_lines.extend(f"    {p}" for p in problems)
            else:
                report_lines.append("  ※ 명백한 문제 감지 안됨")

            # 각 전략 적용
            for strat_name in STRATEGIES:
                corrector = SpeakerCorrector(strategy=strat_name)
                results = corrector.correct(segments)
                eval_result = evaluate_correction_quality(segments, results)

                # 포맷
                comp_lines = format_comparison(segments, results, strat_name.upper())
                report_lines.extend(comp_lines)

                if eval_result["total_changed"] > 0:
                    report_lines.append(
                        f"    → 평가: 개선 {eval_result['improved']} / "
                        f"악화 {eval_result['worsened']} / "
                        f"중립 {eval_result['neutral']} "
                        f"(정확도 {eval_result['accuracy']:.0%})"
                    )

                # 종합 집계
                s = strategy_scores[strat_name]
                s["improved"] += eval_result["improved"]
                s["worsened"] += eval_result["worsened"]
                s["neutral"] += eval_result["neutral"]
                s["changed"] += eval_result["total_changed"]

    # ─────── 종합 결과 ─────────
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("  종합 비교 결과")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"  {'전략':<12} {'변경수':>6} {'개선':>6} {'악화':>6} {'중립':>6} {'정확도':>8} {'순이익':>8}")
    report_lines.append(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")

    best_score = -999
    best_strategy = ""

    for strat_name in STRATEGIES:
        s = strategy_scores[strat_name]
        acc = s["improved"] / max(s["changed"], 1)
        net = s["improved"] - s["worsened"]

        # 순이익 + 정확도 가중 점수
        score = net + (acc * 10)

        report_lines.append(
            f"  {strat_name:<12} {s['changed']:>6} {s['improved']:>6} "
            f"{s['worsened']:>6} {s['neutral']:>6} {acc:>7.0%} {net:>+8}"
        )

        if score > best_score:
            best_score = score
            best_strategy = strat_name

    report_lines.append("")
    report_lines.append(f"  >>> 최적 전략: {best_strategy.upper()} (점수: {best_score:.1f})")
    report_lines.append("")

    # 전략 설명
    desc = {
        "a": "호칭 기반 - '선생님', '환자분', '어머님' 등 호칭 패턴으로 판별",
        "b": "내용 분석 - 진단/설명 어투 vs 증상 호소/질문 패턴으로 판별",
        "ab": "호칭 + 내용 - A와 B의 점수를 합산하여 판별",
        "c": "글로벌 스왑 - 전체 역할이 뒤바뀌었는지 감지하여 일괄 교정",
        "combined": "통합 - A+B+C+D 모두 적용, 글로벌 스왑 후 개별 재판정",
    }
    report_lines.append("  전략 설명:")
    for k, v in desc.items():
        marker = " <<<" if k == best_strategy else ""
        report_lines.append(f"    {k.upper():>10}: {v}{marker}")

    report_lines.append("")

    # ─────── 대표 교정 샘플 (Combined) ─────────
    report_lines.append("=" * 80)
    report_lines.append("  대표 교정 샘플 (Combined 전략)")
    report_lines.append("=" * 80)

    corrector = SpeakerCorrector(strategy="combined")

    for type_num in [1, 2, 3]:  # 문제가 뚜렷한 타입만
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            segments = load_segments(path)
            if not segments:
                continue

            results = corrector.correct(segments)
            has_changes = any(r.changed for r in results)
            if not has_changes:
                continue

            report_lines.append(f"\n  --- Type {type_num} {label} ---")
            for seg, res in zip(segments, results):
                idx = seg.get("index", "?")
                content = seg.get("content", "")
                role_display = res.corrected_role
                marker = ""
                if res.changed:
                    marker = f"  ← 변경({res.original_role}→{res.corrected_role})"
                    role_display = f"*{res.corrected_role}*"

                # 긴 내용은 자름
                if len(content) > 60:
                    content = content[:57] + "..."

                report_lines.append(f"  [{role_display:^6}] {content}{marker}")

    report_lines.append("")

    # 출력 및 저장
    report_text = "\n".join(report_lines)
    print(report_text)

    output_path = DATA_DIR / "speaker_correction_report.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\n리포트 저장: {output_path}")


if __name__ == "__main__":
    main()
