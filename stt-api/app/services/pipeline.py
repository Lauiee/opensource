"""병렬 전사+화자분리 파이프라인."""

import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.executors import get_transcribe_executor
from app.services.diarization import run_diarization
from app.services.postprocessing import deduplicate_segments, postprocess_text
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
    """동기 전사 실행 (스레드 풀용)."""
    segments = transcribe_with_segments(wav_path, language=language, specialty=specialty)
    settings = get_settings()
    if settings.enable_postprocessing:
        segments = deduplicate_segments(segments)
        for seg in segments:
            seg["text"] = postprocess_text(seg.get("text", ""))
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
    pool = get_transcribe_executor()

    if not settings.enable_diarization:
        # 화자분리 비활성화: 기존 전사만
        segments = await loop.run_in_executor(
            pool,
            lambda: _run_transcribe_sync(wav_path, language, specialty),
        )
        return [{"speaker": None, **s} for s in segments]

    # 병렬 실행: 전사 + 화자분리
    trans_task = loop.run_in_executor(
        pool,
        lambda: _run_transcribe_sync(wav_path, language, specialty),
    )
    diar_task = loop.run_in_executor(
        pool,
        lambda: _run_diarization_sync(wav_path),
    )

    transcript_segments, diar_segments = await asyncio.gather(trans_task, diar_task)

    logger.info(
        "전사 %d구간, 화자분리 %d구간 → 매칭",
        len(transcript_segments),
        len(diar_segments),
    )

    # 화자 라벨 할당
    merged = _assign_speaker_to_segments(transcript_segments, diar_segments)

    # Phase 7: 누락 세그먼트 복구 (갭 분석 + 재전사)
    try:
        from app.services.segment_recovery import recover_missing_segments
        recovery = await loop.run_in_executor(
            pool,
            lambda: recover_missing_segments(
                wav_path, merged,
                language=language, specialty=specialty,
                min_gap_sec=1.5,
            ),
        )
        if recovery["segments_recovered"] > 0:
            logger.info(
                "누락 복구: %d개 갭 중 %d개 복구, %d개 세그먼트 추가",
                recovery["gaps_found"],
                recovery["gaps_recovered"],
                recovery["segments_recovered"],
            )
            # 복구된 세그먼트에도 화자 할당
            merged = _assign_speaker_to_segments(
                recovery["merged_segments"], diar_segments,
            )
    except Exception:
        logger.debug("누락 복구 모듈 미사용 (정상 동작)")

    return merged
