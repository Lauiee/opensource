"""병렬 전사+화자분리 파이프라인 (v2.0).

v2.0 개선사항:
  - 후처리 파이프라인 완전 통합 (postprocessing.py)
  - 신뢰도 기반 GPT Tier3 교정 통합
  - 세그먼트별 confidence 메타데이터 전달
"""

import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.services.diarization import run_diarization
from app.services.postprocessing import (
    deduplicate_segments,
    postprocess_text,
    postprocess_segments,
)
from app.services.transcription import transcribe_with_segments

logger = logging.getLogger(__name__)


def _assign_speaker_to_segments(
    transcript_segments: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """
    전사 세그먼트에 화자 라벨 할당.
    1) 세그먼트 중점(center)이 속한 diarization 구간의 speaker 사용
    2) 없으면 시간 겹침(overlap)이 가장 큰 구간 사용
    """
    if not diar_segments:
        logger.warning("화자 분리 결과가 비어 있음")
        return [{**s, "speaker": None} for s in transcript_segments]

    result = []
    for seg in transcript_segments:
        s_start, s_end = seg["start"], seg["end"]
        mid = (s_start + s_end) / 2.0

        # 1) 중점이 포함된 diar 구간 찾기
        speaker = None
        for d in diar_segments:
            if d["start"] <= mid <= d["end"]:
                speaker = d["speaker"]
                break

        # 2) 없으면 overlap 최대인 구간
        if speaker is None:
            best_overlap = 0.0
            for d in diar_segments:
                overlap = min(s_end, d["end"]) - max(s_start, d["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    speaker = d["speaker"]
            # 겹침이 너무 작으면 미할당 (노이즈 방지)
            if best_overlap < 0.05:
                speaker = None

        result.append({**seg, "speaker": speaker})
    return result


def _run_transcribe_sync(
    wav_path: Path,
    language: str,
    specialty: str | None = None,
) -> list[dict]:
    """동기 전사 실행 (스레드 풀용).

    파이프라인:
      1. Whisper 전사 (오디오 전처리 포함)
      2. 텍스트 후처리 (환각 제거, 필러 정리, 숫자 정규화 등)
      3. GPT Tier3 교정 (저신뢰도 세그먼트만)
    """
    # 1단계: Whisper 전사 (전처리 + 신뢰도 스코어링 포함)
    segments = transcribe_with_segments(wav_path, language=language, specialty=specialty)

    settings = get_settings()
    if settings.enable_postprocessing:
        # 2단계: 텍스트 후처리 (환각 패턴 제거, 필러, 중복 등)
        segments = postprocess_segments(segments)

    # 3단계: GPT Tier3 교정 (저신뢰도 세그먼트만, OpenAI 키 있을 때)
    if settings.openai_api_key:
        try:
            from app.medterm.gpt_corrector import get_gpt_corrector
            corrector = get_gpt_corrector()
            segments = corrector.correct_low_confidence_segments(
                segments,
                confidence_threshold=0.45,
                specialty=specialty or "",
            )
        except Exception as e:
            logger.debug("GPT Tier3 교정 건너뜀: %s", e)

    return segments


def _run_diarization_sync(wav_path: Path) -> list[dict]:
    """동기 화자분리 실행 (스레드 풀용)."""
    return run_diarization(wav_path)


async def transcribe_with_diarization(
    wav_path: Path,
    language: str = "ko",
    specialty: str | None = None,
) -> list[dict]:
    """
    전사 + 화자분리 병렬 실행 후 결과 병합.

    Args:
        wav_path: WAV 파일 경로
        language: 언어 코드 (기본: "ko")
        specialty: 진료과 힌트 (예: "정형외과") — Whisper 초기 프롬프트 최적화용

    Returns list of {"start": float, "end": float, "text": str, "speaker": str | None}.
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()

    if not settings.enable_diarization:
        # 화자분리 비활성화: 기존 전사만
        segments = await loop.run_in_executor(
            None,
            lambda: _run_transcribe_sync(wav_path, language, specialty),
        )
        return [{"speaker": None, **s} for s in segments]

    # 병렬 실행: 전사 + 화자분리
    trans_task = loop.run_in_executor(
        None,
        lambda: _run_transcribe_sync(wav_path, language, specialty),
    )
    diar_task = loop.run_in_executor(
        None,
        lambda: _run_diarization_sync(wav_path),
    )

    transcript_segments, diar_segments = await asyncio.gather(trans_task, diar_task)

    logger.info(
        "전사 %d구간, 화자분리 %d구간 → 매칭",
        len(transcript_segments),
        len(diar_segments),
    )

    # 화자 라벨 할당
    return _assign_speaker_to_segments(transcript_segments, diar_segments)
