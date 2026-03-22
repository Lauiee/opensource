"""STT 결과 뷰어 API — 파일 목록, STT 결과 조회, 교정 적용, SOAP 생성, 통계."""

import json
import logging
import os
import re
import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.medterm.engine import (
    get_engine,
    get_learning_manager,
    get_ref_db,
    get_speaker_corrector,
)
from app.medterm.store import DictionaryStore
from app.medterm.engine import MedicalCorrectionEngine

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/test_set")


# ──────────────────────────────────────────────────────────────────────
# SOAPGenerator: 의료 대화에서 SOAP 노트를 생성하는 고급 분류기
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SOAPClassification:
    """개별 세그먼트의 SOAP 분류 결과"""
    category: str               # "S", "O", "A", "P", "greeting", "farewell", "unknown"
    confidence: float           # 0.0 ~ 1.0
    keywords_matched: list[str] = field(default_factory=list)
    role: str = ""              # 원본 화자 역할
    text: str = ""              # 원본 텍스트


class SOAPGenerator:
    """의료 대화에서 SOAP 노트를 생성하는 고급 분류기

    특징:
    - 다중 키워드 점수 체계 (카테고리별 가중 키워드)
    - 화자 역할 인식 (의사 발화 → O/A/P 우선, 환자 발화 → S 우선)
    - 문맥 연속성 (이전 세그먼트의 카테고리 참조)
    - 신뢰도 점수 산출
    - 인사/마무리 발화 별도 처리
    - 핵심 요약 생성
    """

    # SOAP 카테고리별 키워드 및 가중치
    # 가중치: 높을수록 해당 카테고리에 강하게 매핑됨
    KEYWORDS = {
        "S": {
            # 증상 관련
            "아프": 3, "아파": 3, "통증": 3, "쑤시": 2, "쑤셔": 2,
            "저리": 3, "저려": 3, "뻣뻣": 2, "시리": 2, "가려": 2,
            "붓": 2, "부어": 2, "부종": 3, "열이": 2, "두통": 3,
            "어지러": 3, "어지럼": 3, "구토": 3, "메스꺼": 2,
            "기침": 3, "가래": 2, "숨이": 2, "답답": 2,
            "피가": 2, "출혈": 3, "멍": 2,
            # 기간/시점
            "며칠": 2, "일주일": 2, "한달": 2, "한 달": 2, "개월": 2,
            "전부터": 2, "됐어": 2, "됐습니다": 2, "시작": 1,
            "갑자기": 2, "점점": 2, "계속": 1, "자꾸": 2,
            # 병력
            "전에도": 2, "예전에": 2, "가족력": 3, "알레르기": 3,
            "수술받": 2, "약을 먹": 2, "복용": 2,
            # 환자 표현 패턴
            "불편": 2, "힘들": 2, "못하겠": 2, "잠을 못": 2,
            "걱정": 1, "심해": 2, "심해졌": 3,
        },
        "O": {
            # 검사/진찰
            "검사": 3, "검진": 3, "진찰": 3, "촉진": 3,
            "청진": 3, "시진": 3, "타진": 3,
            # 영상 검사
            "MRI": 4, "CT": 4, "X-ray": 4, "엑스레이": 4,
            "초음파": 4, "내시경": 4, "심전도": 3,
            "촬영": 3, "영상": 2,
            # 수치/결과
            "수치": 3, "수준": 2, "결과": 2, "소견": 3,
            "확인": 1, "관찰": 2, "발견": 2,
            "혈압": 3, "맥박": 3, "체온": 3, "산소포화도": 4,
            "혈당": 3, "콜레스테롤": 3, "헤모글로빈": 3,
            "백혈구": 3, "적혈구": 3, "혈소판": 3,
            # 신체 소견
            "확장": 2, "좁아": 2, "비대": 3, "위축": 3,
            "압통": 3, "부종": 2, "발적": 3, "경직": 3,
            "운동범위": 3, "ROM": 4, "근력": 3,
            # 측정
            "밀리미터": 2, "센티미터": 2, "mm": 3, "cm": 3,
            "도": 1, "회": 1,
        },
        "A": {
            # 진단
            "진단": 4, "판단": 3, "의심": 3, "소견상": 3,
            "생각됩니다": 2, "보입니다": 2, "판단됩니다": 3,
            "가능성": 2, "추정": 3,
            # 질환명
            "증후군": 4, "질환": 4, "장애": 3,
            "염": 2, "암": 3, "종양": 4, "낭종": 3,
            "골절": 4, "탈구": 4, "인대": 3,
            "협착": 4, "탈출": 3, "디스크": 3,
            "관절염": 4, "류마티스": 4, "통풍": 3,
            "고혈압": 4, "당뇨": 4, "갑상선": 3,
            "폐렴": 4, "기관지": 3, "천식": 4,
            # 상태 평가
            "심각": 3, "경미": 2, "중등도": 3, "급성": 3, "만성": 3,
            "양성": 3, "악성": 4, "양호": 2,
            "호전": 2, "악화": 3, "진행": 2,
            # 원인 설명
            "때문": 2, "원인": 3, "기인": 3,
        },
        "P": {
            # 치료
            "치료": 3, "수술": 4, "시술": 4, "절제": 4,
            "봉합": 3, "고정": 2, "교정": 2,
            # 약물
            "처방": 4, "약": 2, "투약": 3, "복용": 2,
            "주사": 3, "수액": 3, "항생제": 4,
            "진통제": 3, "소염제": 3, "연고": 2,
            # 재활/관리
            "재활": 4, "물리치료": 4, "운동치료": 4,
            "스트레칭": 3, "마사지": 2,
            # 계획
            "계획": 3, "예정": 2, "필요": 1, "권고": 3,
            "추적": 3, "관찰": 2, "경과": 2,
            # 후속 조치
            "재진": 4, "다음에": 2, "주 후": 3, "개월 후": 3,
            "검진": 2, "예약": 3,
            "입원": 3, "퇴원": 3, "전원": 3,
            # 생활 지도
            "식이": 3, "식단": 2, "금연": 3, "금주": 3,
            "안정": 2, "휴식": 2, "주의": 2,
            # 의사 안내 패턴
            "드리": 1, "하겠습니다": 1, "하실": 1, "해주세요": 2,
            "오세요": 2, "오십시오": 2,
        },
    }

    # 인사/마무리 키워드
    GREETING_KEYWORDS = [
        "안녕하세요", "안녕하십니까", "반갑습니다",
        "오셨어요", "오셨습니까", "어서 오세요",
        "들어오세요", "앉으세요", "어떻게 오셨",
    ]

    FAREWELL_KEYWORDS = [
        "감사합니다", "수고하세요", "수고하셨", "고맙습니다",
        "다음에 뵙", "좋은 하루", "조심히 가세요",
        "안녕히 가세요", "안녕히 계세요", "건강하세요",
        "괜찮으실 거", "걱정 마세요", "나아지실",
    ]

    # 화자 역할별 카테고리 가중치 보정
    # 의사 발화는 O/A/P에 보정, 환자 발화는 S에 보정
    ROLE_WEIGHTS = {
        "doctor": {"S": 0.5, "O": 1.5, "A": 1.5, "P": 1.5},
        "patient": {"S": 1.5, "O": 0.7, "A": 0.5, "P": 0.5},
        "unknown": {"S": 1.0, "O": 1.0, "A": 1.0, "P": 1.0},
    }

    def __init__(self, include_summary: bool = True):
        """
        Args:
            include_summary: True면 핵심 요약을 SOAP 결과에 포함
        """
        self.include_summary = include_summary

    def _detect_role_type(self, role: str) -> str:
        """화자 역할을 doctor/patient/unknown으로 분류"""
        role_lower = role.lower() if role else ""
        if any(k in role_lower for k in ["원장", "의사", "doctor", "dr", "선생"]):
            return "doctor"
        if any(k in role_lower for k in ["환자", "patient", "내원", "보호자"]):
            return "patient"
        return "unknown"

    def _is_greeting(self, text: str) -> bool:
        """인사 발화 여부 판별"""
        # 짧은 텍스트 + 인사 키워드 포함
        if len(text) > 50:
            return False
        return any(kw in text for kw in self.GREETING_KEYWORDS)

    def _is_farewell(self, text: str) -> bool:
        """마무리 발화 여부 판별"""
        if len(text) > 60:
            return False
        return any(kw in text for kw in self.FAREWELL_KEYWORDS)

    def _score_segment(self, text: str, role_type: str) -> dict[str, float]:
        """세그먼트의 각 SOAP 카테고리별 점수 산출

        Returns:
            ({"S": 점수, "O": 점수, "A": 점수, "P": 점수}, {"S": [매칭 키워드], ...})
        """
        scores: dict[str, float] = {"S": 0.0, "O": 0.0, "A": 0.0, "P": 0.0}
        matched: dict[str, list[str]] = {"S": [], "O": [], "A": [], "P": []}

        role_weights = self.ROLE_WEIGHTS.get(role_type, self.ROLE_WEIGHTS["unknown"])

        for category, keywords in self.KEYWORDS.items():
            for keyword, weight in keywords.items():
                if keyword in text:
                    adjusted_weight = weight * role_weights[category]
                    scores[category] += adjusted_weight
                    matched[category].append(keyword)

        return scores, matched

    def classify_segment(
        self,
        text: str,
        role: str,
        prev_category: Optional[str] = None,
    ) -> SOAPClassification:
        """단일 세그먼트를 SOAP 카테고리로 분류

        Args:
            text: 세그먼트 텍스트
            role: 화자 역할 (예: "원장님", "환자")
            prev_category: 이전 세그먼트의 카테고리 (문맥 연속성)

        Returns:
            SOAPClassification 결과
        """
        if not text or not text.strip():
            return SOAPClassification(
                category="unknown", confidence=0.0,
                role=role, text=text,
            )

        # 인사/마무리 먼저 확인
        if self._is_greeting(text):
            return SOAPClassification(
                category="greeting", confidence=0.9,
                keywords_matched=[], role=role, text=text,
            )
        if self._is_farewell(text):
            return SOAPClassification(
                category="farewell", confidence=0.9,
                keywords_matched=[], role=role, text=text,
            )

        # SOAP 점수 산출
        role_type = self._detect_role_type(role)
        scores, matched = self._score_segment(text, role_type)

        # 문맥 연속성 보정: 이전 카테고리와 같은 카테고리에 소량 보정
        if prev_category and prev_category in scores:
            scores[prev_category] += 1.0

        # 최고 점수 카테고리 선택
        total = sum(scores.values())
        if total == 0:
            # 키워드가 하나도 없으면 화자 기반 기본 분류
            if role_type == "patient":
                return SOAPClassification(
                    category="S", confidence=0.3,
                    keywords_matched=[], role=role, text=text,
                )
            elif role_type == "doctor":
                # 의사 발화인데 키워드가 없으면 P(안내/계획)로 기본 분류
                return SOAPClassification(
                    category="P", confidence=0.3,
                    keywords_matched=[], role=role, text=text,
                )
            return SOAPClassification(
                category="unknown", confidence=0.1,
                keywords_matched=[], role=role, text=text,
            )

        best_category = max(scores, key=scores.get)
        confidence = scores[best_category] / total if total > 0 else 0.0

        # 신뢰도 보정: 매칭된 키워드가 많을수록 신뢰도 증가 (최대 1.0)
        keyword_count = len(matched[best_category])
        if keyword_count >= 3:
            confidence = min(confidence + 0.1, 1.0)

        return SOAPClassification(
            category=best_category,
            confidence=round(confidence, 3),
            keywords_matched=matched[best_category],
            role=role,
            text=text,
        )

    def generate(self, corrected_items: list[dict]) -> dict:
        """교정된 세그먼트 리스트에서 SOAP 노트 생성

        Args:
            corrected_items: [{"role": "원장님", "text": "..."}, ...]

        Returns:
            SOAP 구조화 딕셔너리
        """
        # 모든 세그먼트 분류
        classifications: list[SOAPClassification] = []
        prev_category = None

        for item in corrected_items:
            text = item.get("text", "")
            role = item.get("role", "?")

            classification = self.classify_segment(text, role, prev_category)
            classifications.append(classification)

            # 문맥 연속성을 위해 이전 카테고리 기록
            if classification.category in ("S", "O", "A", "P"):
                prev_category = classification.category

        # 카테고리별 세그먼트 수집
        soap_segments: dict[str, list[dict]] = {
            "S": [], "O": [], "A": [], "P": [],
            "greeting": [], "farewell": [],
        }

        for cls in classifications:
            entry = {
                "role": cls.role,
                "text": cls.text,
                "confidence": cls.confidence,
                "keywords": cls.keywords_matched,
            }
            if cls.category in soap_segments:
                soap_segments[cls.category].append(entry)

        # 분류 통계
        total_classified = len(classifications)
        category_counts = {}
        for cls in classifications:
            category_counts[cls.category] = category_counts.get(cls.category, 0) + 1

        avg_confidence = (
            sum(c.confidence for c in classifications) / total_classified
            if total_classified > 0 else 0.0
        )

        # SOAP 구조화 결과
        soap = {
            "S": {
                "title": "Subjective (주관적 소견)",
                "description": "환자가 호소하는 증상, 기간, 병력 등 주관적 표현",
                "segments": soap_segments["S"],
                "content": [seg["text"] for seg in soap_segments["S"]],
                "count": len(soap_segments["S"]),
            },
            "O": {
                "title": "Objective (객관적 소견)",
                "description": "의사의 검사 결과, 진찰 소견, 측정값 등 객관적 정보",
                "segments": soap_segments["O"],
                "content": [seg["text"] for seg in soap_segments["O"]],
                "count": len(soap_segments["O"]),
            },
            "A": {
                "title": "Assessment (평가)",
                "description": "의사의 진단, 질환명, 상태 평가",
                "segments": soap_segments["A"],
                "content": [seg["text"] for seg in soap_segments["A"]],
                "count": len(soap_segments["A"]),
            },
            "P": {
                "title": "Plan (계획)",
                "description": "치료 계획, 처방, 수술, 재활, 추적 관찰 등",
                "segments": soap_segments["P"],
                "content": [seg["text"] for seg in soap_segments["P"]],
                "count": len(soap_segments["P"]),
            },
            "greeting": {
                "title": "인사",
                "content": [seg["text"] for seg in soap_segments["greeting"]],
                "count": len(soap_segments["greeting"]),
            },
            "farewell": {
                "title": "마무리 인사",
                "content": [seg["text"] for seg in soap_segments["farewell"]],
                "count": len(soap_segments["farewell"]),
            },
            "full_transcript": {
                "title": "전체 대화록 (교정 적용)",
                "content": corrected_items,
            },
            "classification_stats": {
                "total_segments": total_classified,
                "category_distribution": category_counts,
                "average_confidence": round(avg_confidence, 3),
            },
        }

        # 핵심 요약 생성
        if self.include_summary:
            soap["summary"] = self._generate_summary(soap_segments, classifications)

        return soap

    def _generate_summary(
        self,
        soap_segments: dict[str, list[dict]],
        classifications: list[SOAPClassification],
    ) -> dict:
        """SOAP 세그먼트로부터 핵심 요약 생성

        Returns:
            {"title": "핵심 요약", "content": "..."}
        """
        summary_parts = []

        # S: 주요 증상 요약
        s_texts = [seg["text"] for seg in soap_segments["S"]]
        if s_texts:
            # 가장 높은 신뢰도의 S 세그먼트에서 핵심 추출
            top_s = sorted(soap_segments["S"], key=lambda x: x["confidence"], reverse=True)
            symptom_summary = top_s[0]["text"]
            if len(symptom_summary) > 80:
                symptom_summary = symptom_summary[:80] + "..."
            summary_parts.append(f"[증상] {symptom_summary}")

        # O: 주요 검사/소견
        o_texts = [seg["text"] for seg in soap_segments["O"]]
        if o_texts:
            top_o = sorted(soap_segments["O"], key=lambda x: x["confidence"], reverse=True)
            finding_summary = top_o[0]["text"]
            if len(finding_summary) > 80:
                finding_summary = finding_summary[:80] + "..."
            summary_parts.append(f"[소견] {finding_summary}")

        # A: 진단
        a_texts = [seg["text"] for seg in soap_segments["A"]]
        if a_texts:
            top_a = sorted(soap_segments["A"], key=lambda x: x["confidence"], reverse=True)
            diagnosis_summary = top_a[0]["text"]
            if len(diagnosis_summary) > 80:
                diagnosis_summary = diagnosis_summary[:80] + "..."
            summary_parts.append(f"[진단] {diagnosis_summary}")

        # P: 치료 계획
        p_texts = [seg["text"] for seg in soap_segments["P"]]
        if p_texts:
            top_p = sorted(soap_segments["P"], key=lambda x: x["confidence"], reverse=True)
            plan_summary = top_p[0]["text"]
            if len(plan_summary) > 80:
                plan_summary = plan_summary[:80] + "..."
            summary_parts.append(f"[계획] {plan_summary}")

        if not summary_parts:
            summary_parts.append("분류 가능한 의료 대화 내용이 부족합니다.")

        return {
            "title": "핵심 요약",
            "content": "\n".join(summary_parts),
            "segment_counts": {
                "subjective": len(s_texts),
                "objective": len(o_texts),
                "assessment": len(a_texts),
                "plan": len(p_texts),
            },
        }


# ──────────────────────────────────────────────────────────────────────
# SOAPGenerator 싱글턴 인스턴스
# ──────────────────────────────────────────────────────────────────────

_soap_generator: Optional[SOAPGenerator] = None


def _get_soap_generator() -> SOAPGenerator:
    """SOAPGenerator 싱글턴 인스턴스 반환"""
    global _soap_generator
    if _soap_generator is None:
        try:
            from app.config import get_settings
            settings = get_settings()
            include_summary = settings.soap_include_summary
        except Exception:
            include_summary = True
        _soap_generator = SOAPGenerator(include_summary=include_summary)
    return _soap_generator


# ──────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ──────────────────────────────────────────────────────────────────────

class FileInfo(BaseModel):
    type_num: int
    type_name: str
    wav_exists: bool
    wav_duration: float | None = None
    donkey_exists: bool
    dalpha_exists: bool
    donkey_segments: int = 0
    dalpha_segments: int = 0


# ──────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────

def _get_wav_duration(wav_path: Path) -> float | None:
    """WAV 파일의 재생 시간(초)을 반환"""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def _load_stt(type_num: int, prefix: str) -> list[dict] | None:
    """STT 결과 파일을 로드하여 세그먼트 리스트로 반환"""
    path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        bracket_end = text.rfind(']')
        if bracket_end >= 0:
            text = text[:bracket_end + 1]
        return json.loads(text)
    except Exception as e:
        logger.warning("STT 파일 로드 실패 (%s): %s", path, e)
        return None


_viewer_engine = None


def _get_engine():
    """교정 엔진 가져오기 (전역 엔진 우선 → 로컬 Tier 1 전용 폴백)."""
    global _viewer_engine

    # 전역 엔진이 있으면 우선 사용 (Tier 2 포함)
    global_engine = get_engine()
    if global_engine is not None:
        return global_engine

    # 전역 엔진이 없으면 로컬 Tier 1 전용 엔진 생성 (폴백)
    if _viewer_engine is not None:
        return _viewer_engine
    dict_path = Path(__file__).parent.parent / "data" / "medical_dict.json"
    if dict_path.exists():
        store = DictionaryStore(dict_path)
        _viewer_engine = MedicalCorrectionEngine(store)
        return _viewer_engine
    return None


# ──────────────────────────────────────────────────────────────────────
# API 엔드포인트
# ──────────────────────────────────────────────────────────────────────

@router.get("/files", response_model=list[FileInfo])
def list_files():
    """사용 가능한 파일 목록 조회."""
    try:
        if not DATA_DIR.exists():
            logger.warning("데이터 디렉토리가 존재하지 않습니다: %s", DATA_DIR)
            return []

        files = []
        for entry in sorted(DATA_DIR.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("type"):
                continue
            try:
                type_num = int(entry.name.replace("type", ""))
            except ValueError:
                continue

            wav_path = entry / f"type{type_num}.wav"
            donkey_data = _load_stt(type_num, "donkey")
            dalpha_data = _load_stt(type_num, "dalpha")

            files.append(FileInfo(
                type_num=type_num,
                type_name=f"Type {type_num}",
                wav_exists=wav_path.exists(),
                wav_duration=_get_wav_duration(wav_path) if wav_path.exists() else None,
                donkey_exists=donkey_data is not None,
                dalpha_exists=dalpha_data is not None,
                donkey_segments=len(donkey_data) if donkey_data else 0,
                dalpha_segments=len(dalpha_data) if dalpha_data else 0,
            ))
        return files
    except Exception as e:
        logger.error("파일 목록 조회 중 오류 발생: %s", e)
        raise HTTPException(500, f"파일 목록 조회 중 오류가 발생했습니다: {e}")


@router.get("/stt/{type_num}/{source}")
def get_stt_result(
    type_num: int,
    source: str,
    fix_speaker: bool = True,
    context_hint: str | None = None,
):
    """STT 결과 조회 (의료용어 교정 + 화자교정 포함).

    Args:
        type_num: 타입 번호
        source: STT 소스 ('donkey' 또는 'dalpha')
        fix_speaker: True면 화자교정도 적용 (기본 활성)
        context_hint: 진료과 힌트 (예: "정형외과") — Tier 2 탐지 범위 한정
    """
    if source not in ("donkey", "dalpha"):
        raise HTTPException(400, "source는 'donkey' 또는 'dalpha'만 가능합니다")

    data = _load_stt(type_num, source)
    if data is None:
        raise HTTPException(404, f"type{type_num}/{source} 데이터가 없습니다")

    try:
        # 1단계: 화자교정 (세그먼트 단위)
        # ★ 대용량 파일(200세그먼트 초과) → AB패턴만 사용 (GPT 타임아웃 방지)
        speaker_corrections = []
        if fix_speaker:
            corrector = get_speaker_corrector()
            if corrector:
                try:
                    if len(data) > 200 and hasattr(corrector, 'correct_ab_only'):
                        logger.info(
                            "대용량 파일 (%d세그먼트): AB패턴만 사용 (GPT 스킵)",
                            len(data),
                        )
                        speaker_results = corrector.correct_ab_only(data)
                    else:
                        speaker_results = corrector.correct(data)
                    speaker_corrections = speaker_results
                except Exception as e:
                    logger.warning("화자교정 실패 (계속 진행): %s", e)

        # 2단계: 의료용어 교정 (전체 파이프라인 — correct_full 사용)
        engine = _get_engine()
        results = []
        total_tier1 = 0
        total_tier2 = 0
        total_pending = 0

        for i, item in enumerate(data):
            content = item.get("content", "")
            original_role = item.get("role", "?")

            # 화자교정 결과 반영
            corrected_role = original_role
            speaker_changed = False
            speaker_strategy = ""
            if i < len(speaker_corrections):
                sc = speaker_corrections[i]
                corrected_role = sc.corrected_role
                speaker_changed = sc.changed
                speaker_strategy = sc.strategy

            entry = {
                "index": item.get("index", i),
                "role": corrected_role,
                "original_role": original_role,
                "speaker_changed": speaker_changed,
                "speaker_strategy": speaker_strategy,
                "original": content,
                "corrected": content,
                "changes": [],
                "tier1_count": 0,
                "tier2_count": 0,
                "pending_count": 0,
            }

            # 의료용어 교정 (correct_full 사용으로 Tier 정보 포함)
            if engine and content:
                try:
                    result = engine.correct_full(content, context_hint=context_hint)
                    entry["corrected"] = result.text
                    entry["tier1_count"] = result.tier1_count
                    entry["tier2_count"] = result.tier2_count
                    entry["pending_count"] = result.pending_count
                    total_tier1 += result.tier1_count
                    total_tier2 += result.tier2_count
                    total_pending += result.pending_count
                    entry["changes"] = [
                        {
                            "original": log.original,
                            "corrected": log.corrected,
                            "strategy": log.strategy.value if hasattr(log.strategy, 'value') else str(log.strategy),
                            "tier": log.tier.value if hasattr(log.tier, 'value') else str(log.tier),
                            "entry_id": log.entry_id,
                        }
                        for log in result.logs
                    ]
                except Exception as e:
                    logger.warning("세그먼트 %d 교정 실패 (원본 유지): %s", i, e)

            results.append(entry)

        return {
            "type_num": type_num,
            "source": source,
            "total_segments": len(results),
            "corrected_segments": sum(1 for r in results if r["changes"]),
            "speaker_corrected": sum(1 for r in results if r["speaker_changed"]),
            "tier1_corrections": total_tier1,
            "tier2_corrections": total_tier2,
            "pending_reviews": total_pending,
            "context_hint": context_hint,
            "segments": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("STT 결과 조회 중 오류 발생 (type%d/%s): %s", type_num, source, e)
        raise HTTPException(500, f"STT 결과 처리 중 오류가 발생했습니다: {e}")


@router.get("/soap/{type_num}/{source}")
def get_soap(type_num: int, source: str, context_hint: str | None = None):
    """STT 결과를 SOAP 노트 형식으로 변환 (화자교정 + 의료용어교정 적용).

    SOAPGenerator 클래스를 사용하여 다중 키워드 스코어링, 화자 역할 인식,
    문맥 연속성, 신뢰도 산출, 핵심 요약을 수행합니다.

    Args:
        type_num: 타입 번호
        source: 'donkey' 또는 'dalpha'
        context_hint: 진료과 힌트 (예: "정형외과") — Tier 2 탐지 범위 한정
    """
    if source not in ("donkey", "dalpha"):
        raise HTTPException(400, "source는 'donkey' 또는 'dalpha'만 가능합니다")

    data = _load_stt(type_num, source)
    if data is None:
        raise HTTPException(404, f"type{type_num}/{source} 데이터가 없습니다")

    try:
        # 1단계: 화자교정
        corrector = get_speaker_corrector()
        speaker_map = {}
        if corrector:
            try:
                speaker_results = corrector.correct(data)
                for i, sc in enumerate(speaker_results):
                    speaker_map[i] = sc.corrected_role
            except Exception as e:
                logger.warning("SOAP 화자교정 실패 (원본 역할 사용): %s", e)

        # 2단계: 의료용어 교정 (전체 파이프라인)
        engine = _get_engine()
        corrected_items = []
        total_corrections = 0

        for i, item in enumerate(data):
            content = item.get("content", "")
            role = speaker_map.get(i, item.get("role", "?"))
            corrected = content
            if engine and content:
                try:
                    result = engine.correct_full(content, context_hint=context_hint)
                    corrected = result.text
                    total_corrections += result.tier1_count + result.tier2_count
                except Exception as e:
                    logger.warning("SOAP 세그먼트 %d 교정 실패 (원본 유지): %s", i, e)
            corrected_items.append({"role": role, "text": corrected})

        # 3단계: SOAPGenerator로 SOAP 노트 생성
        generator = _get_soap_generator()
        soap = generator.generate(corrected_items)

        return {
            "type_num": type_num,
            "source": source,
            "total_corrections": total_corrections,
            "context_hint": context_hint,
            "soap": soap,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("SOAP 생성 중 오류 발생 (type%d/%s): %s", type_num, source, e)
        raise HTTPException(500, f"SOAP 노트 생성 중 오류가 발생했습니다: {e}")


# ──────────────────────────────────────────────────────────────────────
# 의료 용어 교정 통계 API
# ──────────────────────────────────────────────────────────────────────

@router.get("/medterm/stats")
def get_medterm_stats():
    """의료 용어 교정 엔진 전체 통계 반환.

    Tier 1 사전, Tier 2 자동 탐지, 참조 DB, 학습 루프 통계를 포함.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            503,
            "교정 엔진이 초기화되지 않았습니다. 서버를 먼저 시작하세요.",
        )

    try:
        stats = engine.get_stats()
        return {
            "ok": True,
            "stats": stats.model_dump(),
        }
    except Exception as e:
        logger.exception("통계 조회 실패")
        raise HTTPException(500, f"통계 조회 중 오류 발생: {str(e)}")


# ──────────────────────────────────────────────────────────────────────
# 검증 대기 항목 API (Tier 2 pending reviews)
# ──────────────────────────────────────────────────────────────────────

@router.get("/medterm/pending")
def get_pending_reviews(status: str | None = None):
    """검증 대기 항목 목록 조회.

    Args:
        status: 필터 — 'pending', 'approved', 'rejected' (None이면 전체)
    """
    lm = get_learning_manager()
    if lm is None:
        # Tier 2가 비활성인 경우 빈 목록 반환 (에러 아님)
        return {
            "ok": True,
            "tier2_enabled": False,
            "message": "Tier 2 자동 탐지가 비활성 상태입니다 (참조 DB 없음)",
            "reviews": [],
            "total": 0,
        }

    try:
        # status 유효성 검사
        valid_statuses = {"pending", "approved", "rejected", None}
        if status not in valid_statuses:
            raise HTTPException(
                400,
                f"잘못된 status: '{status}'. 'pending', 'approved', 'rejected' 중 하나를 사용하세요.",
            )

        reviews = lm.get_pending_reviews(status=status)
        return {
            "ok": True,
            "tier2_enabled": True,
            "status_filter": status,
            "reviews": reviews,
            "total": len(reviews),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("검증 대기 목록 조회 실패")
        raise HTTPException(500, f"검증 대기 목록 조회 중 오류 발생: {str(e)}")


@router.post("/medterm/pending/{review_id}/approve")
def approve_pending_review(review_id: str, verified_by: str = "human"):
    """검증 대기 항목 승인 → 사전에 추가.

    Args:
        review_id: 검증 대기 항목 ID
        verified_by: 승인자 (기본: 'human')
    """
    lm = get_learning_manager()
    if lm is None:
        raise HTTPException(503, "Tier 2 학습 매니저가 비활성 상태입니다")

    try:
        entry = lm.approve_review(review_id, verified_by=verified_by)
        if entry is None:
            raise HTTPException(
                404,
                f"검증 대기 항목을 찾을 수 없거나 이미 처리되었습니다: {review_id}",
            )

        # 엔진 리컴파일 (새 사전 항목 반영)
        engine = _get_engine()
        if engine:
            engine.reload()

        return {
            "ok": True,
            "message": f"'{entry.wrong}' → '{entry.correct}' 승인 및 사전 추가 완료",
            "entry": entry.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("검증 항목 승인 실패")
        raise HTTPException(500, f"승인 처리 중 오류 발생: {str(e)}")


@router.post("/medterm/pending/{review_id}/reject")
def reject_pending_review(review_id: str):
    """검증 대기 항목 기각.

    Args:
        review_id: 검증 대기 항목 ID
    """
    lm = get_learning_manager()
    if lm is None:
        raise HTTPException(503, "Tier 2 학습 매니저가 비활성 상태입니다")

    try:
        success = lm.reject_review(review_id)
        if not success:
            raise HTTPException(
                404,
                f"검증 대기 항목을 찾을 수 없거나 이미 처리되었습니다: {review_id}",
            )

        return {
            "ok": True,
            "message": f"검증 항목 {review_id} 기각 완료",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("검증 항목 기각 실패")
        raise HTTPException(500, f"기각 처리 중 오류 발생: {str(e)}")


# ──────────────────────────────────────────────────────────────────────
# 파일 업로드
# ──────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    type_name: str = Form(default=""),
):
    """새로운 STT 결과 파일 또는 WAV 파일 업로드."""
    if not file.filename:
        raise HTTPException(400, "파일명이 없습니다")

    try:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 다음 타입 번호 결정
        existing = [
            int(d.name.replace("type", ""))
            for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name.startswith("type") and d.name.replace("type", "").isdigit()
        ]
        next_num = max(existing, default=0) + 1

        # 파일 확장자에 따라 처리
        filename = file.filename.lower()
        content = await file.read()

        if filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".m4a") or filename.endswith(".ogg") or filename.endswith(".flac"):
            # 음성 파일 → 자동 STT 전사 + 화자 분리 + 의료 용어 교정
            type_dir = DATA_DIR / f"type{next_num}"
            type_dir.mkdir(exist_ok=True)
            dest = type_dir / f"type{next_num}.wav"
            dest.write_bytes(content)
            logger.info("음성 파일 저장: type%d (%d bytes)", next_num, len(content))

            # STT 전사 시도
            try:
                import asyncio
                from app.services.audio import ensure_wav_16k_mono

                wav_path = ensure_wav_16k_mono(dest)

                # 1) 화자분리 포함 파이프라인 시도
                # 2) 실패 시 Whisper 전사만 (화자분리 없이)
                segments = None
                try:
                    from app.services.pipeline import transcribe_with_diarization
                    segments = await transcribe_with_diarization(wav_path, language="ko")
                except ImportError:
                    logger.info("pyannote 미설치 — Whisper 전사만 수행")
                except Exception as pipe_err:
                    logger.warning("파이프라인 실패, Whisper만 시도: %s", pipe_err)

                if segments is None:
                    # Whisper 전사만 (화자분리 없이)
                    from app.services.transcription import transcribe_with_segments
                    loop = asyncio.get_running_loop()
                    raw_segs = await loop.run_in_executor(
                        None,
                        lambda: transcribe_with_segments(wav_path, language="ko"),
                    )
                    segments = [{"speaker": None, **s} for s in raw_segs]

                # 뷰어 형식으로 변환
                result_segments = []
                for i, seg in enumerate(segments):
                    speaker = seg.get("speaker")
                    if speaker == "SPEAKER_00":
                        role = "의사"
                    elif speaker == "SPEAKER_01":
                        role = "환자"
                    elif speaker is None:
                        role = "?"
                    else:
                        role = speaker
                    result_segments.append({
                        "index": i,
                        "role": role,
                        "content": seg.get("text", ""),
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                    })

                # 의료 용어 교정
                engine = _get_engine()
                corrections_applied = 0
                if engine:
                    for seg in result_segments:
                        try:
                            result = engine.correct_full(seg["content"])
                            if result.text != result.original_text:
                                seg["content"] = result.text
                                corrections_applied += 1
                        except Exception:
                            pass

                # JSON 저장
                result_path = type_dir / f"donkey_type{next_num}.txt"
                result_path.write_text(
                    json.dumps(result_segments, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                logger.info(
                    "전사+저장 완료: type%d (%d세그먼트, %d교정)",
                    next_num, len(result_segments), corrections_applied,
                )
                return {
                    "ok": True,
                    "type_num": next_num,
                    "segments": len(result_segments),
                    "corrections": corrections_applied,
                    "file": str(dest),
                    "message": f"Type {next_num}: 전사 완료 ({len(result_segments)}세그먼트, {corrections_applied}교정)",
                }

            except ImportError as ie:
                logger.warning("STT 엔진 미설치: %s — 파일만 저장", ie)
                return {
                    "ok": True,
                    "type_num": next_num,
                    "file": str(dest),
                    "message": f"Type {next_num}에 음성 파일 저장됨 (STT 엔진 미설치)",
                }
            except Exception as stt_err:
                logger.error("STT 전사 실패: %s", stt_err)
                return {
                    "ok": False,
                    "type_num": next_num,
                    "file": str(dest),
                    "message": f"음성 저장됨, STT 실패: {stt_err}",
                }

        elif filename.endswith(".txt") or filename.endswith(".json"):
            # STT 결과 파일 — 어떤 타입에 넣을지 결정
            match = re.search(r'type(\d+)', filename)
            if match:
                target_num = int(match.group(1))
            else:
                target_num = next_num

            type_dir = DATA_DIR / f"type{target_num}"
            type_dir.mkdir(exist_ok=True)

            # donkey/dalpha 판별
            if "dalpha" in filename or "d-alpha" in filename.lower():
                prefix = "dalpha"
            else:
                prefix = "donkey"

            dest = type_dir / f"{prefix}_type{target_num}.txt"
            dest.write_bytes(content)
            logger.info("STT 결과 업로드 완료: type%d/%s (%d bytes)", target_num, prefix, len(content))
            return {
                "ok": True,
                "type_num": target_num,
                "file": str(dest),
                "message": f"Type {target_num}에 {prefix} STT 결과 저장됨",
            }

        else:
            raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {filename}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("파일 업로드 중 오류 발생: %s", e)
        raise HTTPException(500, f"파일 업로드 중 오류가 발생했습니다: {e}")


@router.post("/transcribe-and-save")
async def transcribe_and_save(
    file: UploadFile = File(...),
    specialty: str = Form(default=""),
    source_label: str = Form(default="donkey"),
):
    """음성 파일 업로드 → STT 전사 → 결과 자동 저장.

    새로운 음성 파일을 추가할 때 사용. 전체 파이프라인을 한 번에 수행:
    1. WAV 파일 저장
    2. Whisper STT 전사 (진료과에 맞는 프롬프트 사용)
    3. 화자 분리 (diarization)
    4. 의료 용어 후처리
    5. 결과 JSON 저장 → 뷰어에서 바로 확인 가능

    Args:
        file: 음성 파일 (.wav, .mp3, .m4a)
        specialty: 진료과 힌트 (예: "정형외과", "안과", "내과") — 빈 문자열이면 자동
        source_label: 결과 저장 접두사 (기본 "donkey")
    """
    if not file.filename:
        raise HTTPException(400, "파일명이 없습니다")

    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".ogg", ".flac")):
        raise HTTPException(400, f"지원하지 않는 음성 형식입니다: {filename}")

    try:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 다음 타입 번호 결정
        existing = [
            int(d.name.replace("type", ""))
            for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name.startswith("type") and d.name.replace("type", "").isdigit()
        ]
        next_num = max(existing, default=0) + 1

        # 1) WAV 파일 저장
        type_dir = DATA_DIR / f"type{next_num}"
        type_dir.mkdir(exist_ok=True)
        wav_dest = type_dir / f"type{next_num}.wav"
        content = await file.read()
        wav_dest.write_bytes(content)
        logger.info("음성 파일 저장: type%d (%d bytes)", next_num, len(content))

        # 2) STT 전사 + 화자 분리
        try:
            from app.services.pipeline import transcribe_with_diarization
            from app.services.audio import ensure_wav_16k_mono

            wav_path = ensure_wav_16k_mono(wav_dest)
            segments = await transcribe_with_diarization(wav_path, language="ko")
        except ImportError:
            raise HTTPException(
                503,
                "STT 엔진이 로드되지 않았습니다. Faster-Whisper가 필요합니다.",
            )
        except Exception as e:
            logger.error("STT 전사 실패: %s", e)
            raise HTTPException(500, f"STT 전사 중 오류가 발생했습니다: {e}")

        # 3) 결과를 뷰어 형식으로 변환
        #    뷰어가 기대하는 형식: [{"index": int, "role": str, "content": str}, ...]
        result_segments = []
        for i, seg in enumerate(segments):
            # speaker → role 변환
            speaker = seg.get("speaker")
            if speaker == "SPEAKER_00":
                role = "의사"
            elif speaker == "SPEAKER_01":
                role = "환자"
            else:
                role = speaker or "?"

            result_segments.append({
                "index": i,
                "role": role,
                "content": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

        # 4) 의료 용어 교정 적용 (Tier 1 + Tier 2)
        engine = _get_engine()
        corrections_applied = 0
        if engine:
            ctx = specialty if specialty else None
            for seg in result_segments:
                try:
                    result = engine.correct_full(seg["content"], context_hint=ctx)
                    if result.text != result.original_text:
                        seg["content"] = result.text
                        corrections_applied += 1
                except Exception:
                    pass  # 교정 실패해도 원본 유지

        # 5) JSON 저장
        prefix = source_label if source_label in ("donkey", "dalpha") else "donkey"
        result_path = type_dir / f"{prefix}_type{next_num}.txt"
        result_path.write_text(
            json.dumps(result_segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "전사+저장 완료: type%d (%d세그먼트, %d교정)",
            next_num, len(result_segments), corrections_applied,
        )

        return {
            "ok": True,
            "type_num": next_num,
            "segments": len(result_segments),
            "corrections": corrections_applied,
            "specialty": specialty or "(자동)",
            "wav_file": str(wav_dest),
            "result_file": str(result_path),
            "message": f"Type {next_num}: 전사 완료 ({len(result_segments)}세그먼트)",
            "viewer_url": f"/viewer → Type {next_num} 선택",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("전사+저장 중 오류 발생: %s", e)
        raise HTTPException(500, f"전사+저장 중 오류가 발생했습니다: {e}")
