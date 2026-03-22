"""기존 하드코딩 MEDICAL_DICT → JSON 마이그레이션 + 분석 기반 추가 항목."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# 기존 postprocessing.py의 MEDICAL_DICT (정형외과)
_LEGACY_ORTHOPEDIC = [
    ("전체 환수로", "전치환술 후"),
    ("전체 환술의", "전치환술의"),
    ("전체 환술을", "전치환술을"),
    ("전체 환술", "전치환술"),
    ("전체환술", "전치환술"),
    ("전체 환수", "전치환술"),
    ("전체환수", "전치환술"),
    ("무혈설 계세사", "무혈성 괴사"),
    ("무혈성 계세사", "무혈성 괴사"),
    ("무혈성계세사", "무혈성 괴사"),
    ("무혈성 계서", "무혈성 괴사"),
    ("계세사증", "괴사증"),
    ("계세사", "괴사"),
    ("관절념", "관절염"),
    ("관절륨", "관절염"),
    ("관질염", "관절염"),
    ("고관질", "고관절"),
    ("이용성증", "이형성증"),
    ("이영성증", "이형성증"),
    ("스테로지를", "스테로이드를"),
    ("스테로지", "스테로이드"),
    ("스테로에즈", "스테로이드"),
    ("릴리카", "리리카"),
    ("니리카", "리리카"),
    ("트리드로", "트리돌"),
    ("세레네스", "세레브렉스"),
    ("코피바", "본비바"),
    ("연고라 골절", "연골하 골절"),
    ("손통제", "진통제"),
    ("고기 고른 즙", "고름집"),
    ("구름집같이", "고름집같이"),
    ("구름집", "고름집"),
    ("액저리", "엑스레이"),
    ("이명옥", "임영욱"),
]

# 데이터 분석에서 발견된 추가 오류
_DISCOVERED_OPHTHALMOLOGY = [
    ("백래장", "백내장", "백내장이 정확한 의학 용어"),
    ("백내장 진에 억지한 안약", "백내장 진행 억제 안약", "안약 관련 표현 교정"),
    ("진연제", "점안제", "안과 점안제 교정"),
]

_DISCOVERED_GASTRO = [
    ("담집", "담즙", "담즙(Bile)의 오인식"),
    ("총담관 난종", "총담관 낭종", "낭종(cyst) 오인식"),
    ("총담관 난독", "총담관 낭종", "낭종 다른 오인식 패턴"),
    ("정당관 낭독", "총담관 낭종", "총담관 오인식"),
    ("담관 감석증", "담관 담석증", "담석증 오인식"),
    ("담관 암증", "담관암", "담관암 형태 교정"),
    ("유황화이 담관 공장문 앞줄", "루와이 담관 공장 문합술", "Roux-en-Y hepaticojejunostomy"),
    ("유황화의 낭독", "루와이 낭종", "수술법 오인식"),
]

_DISCOVERED_ENDOCRINE = [
    ("쿠킹 중이기", "쿠싱 증후군이기", "쿠싱 증후군(Cushing syndrome)"),
    ("쿠킹증후군", "쿠싱 증후군", "쿠싱 증후군 붙여쓰기"),
    ("쿠킹 증후군", "쿠싱 증후군", "쿠싱 증후군 띄어쓰기"),
]

_DISCOVERED_GENERAL = [
    ("공상공포증", "광장공포증", "agoraphobia 오인식"),
    ("양성 종변", "양성 병변", "병변(lesion) 오인식"),
    ("남성으로 확장돼", "낭성으로 확장돼", "낭성(cystic) 오인식"),
    ("기능해 떨어지고", "기능이 떨어지고", "조사 오인식"),
    ("뼈 기품은", "뼈 붙으면", "음성 유사 오인식"),
    ("사타리", "사타구니", "부위명 오인식"),
]

# Whisper initial_prompt용 핵심 용어
_PROMPT_TERMS = [
    "정형외과", "진료", "상담", "의사", "환자",
    "고관절", "무릎", "척추", "디스크", "인공관절", "수술", "재활",
    "X-ray", "MRI", "CT", "골절", "연골", "인대", "관절염",
    "퇴행성", "류마티스", "스테로이드", "주사", "물리치료",
    "통증", "저림", "부종", "염증", "감염", "항생제",
    "대퇴골", "경골", "비골", "슬개골", "반월상연골", "십자인대",
    "이형성증", "활액막", "사타구니",
    "고관절 치환술", "슬관절 치환술", "관절경",
    "진통제", "소염제", "리리카", "세레브렉스",
    "처방", "약", "입원", "퇴원", "외래",
    # 안과
    "백내장", "녹내장", "비문증", "안압", "시력", "안약", "점안제",
    # 소화기
    "담즙", "총담관", "낭종", "담석", "담관암", "간", "소장",
    # 내분비
    "쿠싱 증후군", "부신", "호르몬", "코르티솔",
    # 일반
    "병변", "진단", "검사", "치료", "경과", "합병증",
]


def _uid() -> str:
    return uuid4().hex[:12]


def build_migration_dict() -> dict:
    """마이그레이션 데이터 생성."""
    entries = []

    # 기존 정형외과 41개
    for i, (wrong, correct) in enumerate(_LEGACY_ORTHOPEDIC):
        entries.append({
            "id": _uid(),
            "wrong": wrong,
            "correct": correct,
            "category": "정형외과",
            "strategy": "exact",
            "pattern": None,
            "context_hint": [],
            "priority": 60 + len(wrong),  # 긴 패턴 우선
            "confidence": 1.0,
            "enabled": True,
            "notes": "기존 MEDICAL_DICT에서 마이그레이션",
        })

    # 안과
    for wrong, correct, note in _DISCOVERED_OPHTHALMOLOGY:
        entries.append({
            "id": _uid(),
            "wrong": wrong,
            "correct": correct,
            "category": "안과",
            "strategy": "exact",
            "pattern": None,
            "context_hint": ["눈", "시력", "안약", "안과"],
            "priority": 50 + len(wrong),
            "confidence": 1.0,
            "enabled": True,
            "notes": note,
        })

    # 소화기내과
    for wrong, correct, note in _DISCOVERED_GASTRO:
        entries.append({
            "id": _uid(),
            "wrong": wrong,
            "correct": correct,
            "category": "소화기내과",
            "strategy": "exact",
            "pattern": None,
            "context_hint": ["담", "간", "소장", "수술"],
            "priority": 50 + len(wrong),
            "confidence": 1.0,
            "enabled": True,
            "notes": note,
        })

    # 내분비내과
    for wrong, correct, note in _DISCOVERED_ENDOCRINE:
        entries.append({
            "id": _uid(),
            "wrong": wrong,
            "correct": correct,
            "category": "내분비내과",
            "strategy": "exact",
            "pattern": None,
            "context_hint": ["호르몬", "부신", "증후군"],
            "priority": 50 + len(wrong),
            "confidence": 1.0,
            "enabled": True,
            "notes": note,
        })

    # 일반
    for wrong, correct, note in _DISCOVERED_GENERAL:
        entries.append({
            "id": _uid(),
            "wrong": wrong,
            "correct": correct,
            "category": "일반",
            "strategy": "exact",
            "pattern": None,
            "context_hint": [],
            "priority": 50 + len(wrong),
            "confidence": 1.0,
            "enabled": True,
            "notes": note,
        })

    return {
        "version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_terms": _PROMPT_TERMS,
        "entries": entries,
    }


def run_migration(output_path: Path | None = None) -> Path:
    """마이그레이션 실행, JSON 파일 생성."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent.parent / "data" / "medical_dict.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_migration_dict()

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"마이그레이션 완료: {len(data['entries'])}개 항목 → {output_path}")
    return output_path


if __name__ == "__main__":
    run_migration()
