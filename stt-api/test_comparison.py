"""STT 교정 전/후 비교 리포트 생성."""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.models import CorrectionLog, CorrectionResult, MatchStrategy
from app.medterm.store import DictionaryStore
from app.medterm.engine import MedicalCorrectionEngine

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")
OUTPUT_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")


def load_stt(type_num: int, prefix: str) -> list[dict] | None:
    path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        # 끝에 불필요한 텍스트 제거 (JSON 배열 이후)
        bracket_end = text.rfind(']')
        if bracket_end >= 0:
            text = text[:bracket_end + 1]
        return json.loads(text)
    except Exception as e:
        print(f"  [WARN] {path.name} 파싱 실패: {e}")
        return None


def main():
    dict_path = Path(__file__).parent / "data" / "medical_dict.json"
    # Tier 1 사전 교정만 사용 (Tier 2 KOSTOM 비교는 느리므로 제외)
    store = DictionaryStore(dict_path)
    engine = MedicalCorrectionEngine(store)

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("  STT 의료 용어 교정 전/후 비교 리포트")
    report_lines.append(f"  생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 90)

    total_corrections = 0
    total_segments = 0
    total_auto_learned = 0

    for type_num in range(1, 11):
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            data = load_stt(type_num, prefix)
            if not data:
                continue

            corrections_in_type = []
            for i, item in enumerate(data):
                text = item.get("content", "")
                if not text:
                    continue
                total_segments += 1
                result = engine.correct(text)
                if result.logs:
                    corrections_in_type.append({
                        "index": item.get("index", i),
                        "role": item.get("role", "?"),
                        "before": text,
                        "after": result.text,
                        "changes": [(log.original, log.corrected, log.strategy) for log in result.logs],
                    })

            if not corrections_in_type:
                continue

            total_corrections += len(corrections_in_type)
            report_lines.append("")
            report_lines.append("-" * 90)
            report_lines.append(f"  TYPE {type_num} ({label} STT) - {len(corrections_in_type)}개 교정")
            report_lines.append("-" * 90)

            for c in corrections_in_type:
                report_lines.append(f"")
                report_lines.append(f"  [{c['index']}] {c['role']}")
                report_lines.append(f"  BEFORE: {c['before']}")
                report_lines.append(f"  AFTER : {c['after']}")
                for orig, corr, strategy in c['changes']:
                    marker = "AUTO" if "auto" in str(strategy).lower() or "auto_" in str(corr) else "DICT"
                    report_lines.append(f"    -> [{marker}] '{orig}' => '{corr}'")

    # 요약
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("  요약")
    report_lines.append("=" * 90)
    report_lines.append(f"  총 세그먼트: {total_segments}개")
    report_lines.append(f"  교정된 세그먼트: {total_corrections}개")
    report_lines.append(f"  교정율: {total_corrections / max(total_segments, 1) * 100:.1f}%")

    # 사전 현황
    dict_data = json.loads(dict_path.read_text(encoding="utf-8"))
    entries = dict_data.get("entries", [])
    auto_entries = [e for e in entries if "source=auto" in e.get("notes", "")]
    report_lines.append(f"")
    report_lines.append(f"  사전 현황:")
    report_lines.append(f"    총 항목: {len(entries)}개")
    report_lines.append(f"    수동 등록: {len(entries) - len(auto_entries)}개")
    report_lines.append(f"    자동 학습: {len(auto_entries)}개")

    # KOSTOM 참조 DB 현황
    ref_path = dict_path.parent / "kostom_reference.json"
    if ref_path.exists():
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        specs = ref_data.get("specialties", {})
        total_terms = sum(len(v.get("terms", [])) for v in specs.values())
        report_lines.append(f"")
        report_lines.append(f"  KOSTOM 참조 DB:")
        report_lines.append(f"    진료과: {len(specs)}개")
        report_lines.append(f"    표준 용어: {total_terms}개")

    # pending reviews
    pending_path = dict_path.parent / "pending_reviews.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        pending_count = len([p for p in pending if p.get("status") == "pending"])
        report_lines.append(f"")
        report_lines.append(f"  검증 대기열: {pending_count}개 대기 중")

    report_lines.append("")
    report_lines.append("=" * 90)

    report_text = "\n".join(report_lines)

    # 파일로 저장
    output_path = OUTPUT_DIR / "correction_report.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"리포트 저장: {output_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
