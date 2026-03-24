"""2-Pass 전사 시스템 — 교정 결과를 프롬프트에 반영하여 재전사.

최신 연구 기반:
- 1차 전사: 기본 프롬프트로 전사
- 교정: 의료 사전 + 문맥 교정 적용
- 2차 전사: 1차 교정 결과를 initial_prompt에 포함하여 재전사
  → Whisper가 이전 문맥을 참고하여 더 정확한 전사 생성

이 기법은 Whisper의 condition_on_previous_text 기능을 활용한 것으로,
동일한 모델로도 CER을 추가 개선할 수 있다.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def two_pass_transcribe(
    wav_path: str,
    language: str = "ko",
    specialty: str = None,
    type_num: int = None,
    correction_fn=None,
) -> list:
    """2-Pass 전사.

    Args:
        wav_path: WAV 파일 경로
        language: 언어 코드
        specialty: 진료과 힌트
        type_num: Type 번호
        correction_fn: 교정 함수 (text → corrected_text)

    Returns:
        최종 세그먼트 리스트
    """
    from app.services.transcription import (
        _get_faster_whisper_model,
        get_initial_prompt,
        _filter_transcription_hallucinations,
        _seg_confidence,
    )
    from app.config import get_settings

    settings = get_settings()
    fw_model = _get_faster_whisper_model()
    base_prompt = get_initial_prompt(specialty, type_num=type_num)

    # ── 1차 전사 ──
    logger.info("1차 전사 시작: %s", wav_path)
    segments_1, info_1 = fw_model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
            "threshold": 0.5,
        },
        initial_prompt=base_prompt,
        condition_on_previous_text=True,
        temperature=0.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
    )

    # 1차 결과 수집
    pass1_segments = []
    pass1_texts = []
    for seg in segments_1:
        text = (seg.text or "").strip()
        text = _filter_transcription_hallucinations(text)
        if text:
            pass1_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "confidence": _seg_confidence(seg),
            })
            pass1_texts.append(text)

    # 1차 결과 교정
    pass1_full = " ".join(pass1_texts)
    if correction_fn:
        pass1_corrected = correction_fn(pass1_full)
    else:
        pass1_corrected = pass1_full

    logger.info("1차 전사: %d 세그먼트, %d자 → 교정: %d자",
                len(pass1_segments), len(pass1_full), len(pass1_corrected))

    # ── 2차 전사: 교정된 1차 결과를 프롬프트에 반영 ──
    # 교정된 텍스트의 핵심 키워드를 추출하여 프롬프트에 추가
    enhanced_prompt = _build_enhanced_prompt(base_prompt, pass1_corrected)

    logger.info("2차 전사 시작 (강화 프롬프트)")
    segments_2, info_2 = fw_model.transcribe(
        str(wav_path),
        language=language,
        beam_size=10,  # 더 큰 beam size로 정확도 향상
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
            "threshold": 0.45,  # 약간 더 민감하게
        },
        initial_prompt=enhanced_prompt,
        condition_on_previous_text=True,
        temperature=0.0,
        no_speech_threshold=0.55,
        repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
    )

    # 2차 결과 수집
    pass2_segments = []
    for seg in segments_2:
        text = (seg.text or "").strip()
        text = _filter_transcription_hallucinations(text)
        if text:
            pass2_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "confidence": _seg_confidence(seg),
                "pass": 2,
            })

    # ── 최종 선택: 1차 vs 2차 비교 ──
    pass2_full = " ".join(s["text"] for s in pass2_segments)

    # 2차가 더 나은지 판단 (길이, 환각 비율 등)
    if _is_pass2_better(pass1_full, pass2_full):
        logger.info("2차 전사 채택: %d 세그먼트 (1차: %d)",
                    len(pass2_segments), len(pass1_segments))
        return pass2_segments
    else:
        logger.info("1차 전사 유지: %d 세그먼트 (2차가 열등)",
                    len(pass1_segments))
        return pass1_segments


def _build_enhanced_prompt(base_prompt: str, corrected_text: str) -> str:
    """1차 교정 결과에서 핵심 의료 용어를 추출하여 프롬프트 강화.

    너무 긴 프롬프트는 Whisper 성능을 저하시킬 수 있으므로,
    핵심 용어만 추출하여 224토큰 이내로 유지.
    """
    import re

    # 의료 용어 패턴 추출
    medical_patterns = [
        r"[가-힣]+증",      # ~증 (증후군, 관절염 등)
        r"[가-힣]+술",      # ~술 (수술, 문합술 등)
        r"[가-힣]+제",      # ~제 (해열제, 진통제 등)
        r"[가-힣]+검사",    # ~검사
        r"[가-힣]+치료",    # ~치료
        r"[A-Za-z]{2,}",   # 영어 의학 용어
    ]

    extracted_terms = set()
    for pattern in medical_patterns:
        matches = re.findall(pattern, corrected_text)
        extracted_terms.update(matches)

    # 프롬프트 구성 (기본 + 추출된 용어)
    extra_terms = ", ".join(sorted(extracted_terms)[:30])  # 최대 30개
    enhanced = f"{base_prompt} {extra_terms}"

    # 224토큰 제한 (대략 한국어 500자)
    if len(enhanced) > 500:
        enhanced = enhanced[:500]

    return enhanced


def _is_pass2_better(pass1: str, pass2: str) -> bool:
    """2차 전사가 1차보다 나은지 판단."""
    import re

    # 환각 비율 비교
    def hallucination_score(text):
        score = 0
        if re.search(r"(\d{2,}월부터\.?\s*){3,}", text):
            score += 10
        if re.search(r"(\d\s+){6,}", text):
            score += 5
        # 반복 문장
        sentences = text.split(".")
        if len(sentences) > 2:
            unique = len(set(sentences))
            if unique / len(sentences) < 0.3:
                score += 10
        return score

    h1 = hallucination_score(pass1)
    h2 = hallucination_score(pass2)

    # 환각이 적은 쪽 선택
    if h2 < h1:
        return True
    if h2 > h1:
        return False

    # 길이 비교: 너무 짧지도 길지도 않은 쪽
    len_ratio = len(pass2) / max(len(pass1), 1)
    if 0.7 < len_ratio < 1.5:
        # 비슷한 길이면 2차 (더 큰 beam size 사용) 채택
        return True

    return False
