"""병렬 전사+화자분리 파이프라인."""

import asyncio
import logging
import tempfile
from pathlib import Path

from app.config import get_settings
from app.services.diarization import run_diarization
from app.services.postprocessing import deduplicate_segments, postprocess_text
from app.services.segment_recovery import slice_audio_meta
from app.services.transcription import (
    _filter_transcription_hallucinations,
    transcribe_with_segments,
    transcribe_words_flat,
)

logger = logging.getLogger(__name__)

# diar_chunk: 슬라이스 앞뒤 패딩(초). 너무 크면 인접 화자 음성이 섞일 수 있음.
DIAR_CHUNK_PADDING_SEC = 0.25


def _smooth_speaker_labels(segments: list[dict], min_stable_sec: float = 1.0) -> list[dict]:
    """짧은 화자 튐(ABA 패턴의 B)을 인접 화자로 보정."""
    if len(segments) < 3:
        return segments

    smoothed = [dict(s) for s in segments]
    for i in range(1, len(smoothed) - 1):
        prev_seg = smoothed[i - 1]
        cur_seg = smoothed[i]
        next_seg = smoothed[i + 1]

        prev_spk = prev_seg.get("speaker")
        cur_spk = cur_seg.get("speaker")
        next_spk = next_seg.get("speaker")
        cur_dur = max(0.0, float(cur_seg["end"]) - float(cur_seg["start"]))

        if (
            prev_spk is not None
            and next_spk is not None
            and prev_spk == next_spk
            and cur_spk not in (None, prev_spk)
            and cur_dur < min_stable_sec
        ):
            cur_seg["speaker"] = prev_spk

    return smoothed


def _speaker_for_interval(
    s_start: float,
    s_end: float,
    diar_segments: list[dict],
    min_overlap_sec: float = 0.05,
) -> str | None:
    """시간 구간에 대응하는 diar 화자 1명."""
    mid = (s_start + s_end) / 2.0
    for d in diar_segments:
        if d["start"] <= mid <= d["end"]:
            return d["speaker"]
    best_overlap = 0.0
    speaker = None
    for d in diar_segments:
        overlap = min(s_end, d["end"]) - max(s_start, d["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            speaker = d["speaker"]
    if best_overlap < min_overlap_sec:
        return None
    return speaker


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
        spk = _speaker_for_interval(seg["start"], seg["end"], diar_segments, min_overlap_sec=0.05)
        result.append({**seg, "speaker": spk})
    return _smooth_speaker_labels(result)


def _merge_words_with_diarization(
    words: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """단어별 화자 할당 후, 같은 화자 연속 구간으로 병합."""
    if not words:
        return []
    if not diar_segments:
        raw = "".join(w.get("word") or "" for w in words).strip()
        text = _filter_transcription_hallucinations(raw)
        if not text:
            return []
        return [{
            "start": round(words[0]["start"], 3),
            "end": round(words[-1]["end"], 3),
            "text": text,
            "speaker": None,
            "confidence": 0.5,
        }]

    merged: list[dict] = []
    cur_spk: str | None = None
    cur_start: float | None = None
    cur_end: float | None = None
    parts: list[str] = []
    probs: list[float] = []

    def flush() -> None:
        nonlocal cur_spk, cur_start, cur_end, parts, probs
        if cur_start is None or not parts:
            cur_spk, cur_start, cur_end, parts, probs = None, None, None, [], []
            return
        raw = "".join(parts).strip()
        text = _filter_transcription_hallucinations(raw)
        if text:
            conf = sum(probs) / len(probs) if probs else 0.5
            merged.append({
                "start": round(cur_start, 3),
                "end": round(cur_end or cur_start, 3),
                "text": text,
                "speaker": cur_spk,
                "confidence": round(min(1.0, max(0.0, conf)), 3),
            })
        cur_spk, cur_start, cur_end, parts, probs = None, None, None, [], []

    for w in words:
        spk = _speaker_for_interval(w["start"], w["end"], diar_segments, min_overlap_sec=0.02)
        wpiece = w.get("word") or ""
        if spk != cur_spk:
            flush()
            cur_spk = spk
            cur_start = w["start"]
            cur_end = w["end"]
            parts = [wpiece]
            p = w.get("probability")
            probs = [float(p)] if p is not None else []
        else:
            cur_end = w["end"]
            parts.append(wpiece)
            p = w.get("probability")
            if p is not None:
                probs.append(float(p))
    flush()
    return _smooth_speaker_labels(merged)


def _apply_postprocess(segments: list[dict], *, use_dedupe: bool) -> list[dict]:
    settings = get_settings()
    if not settings.enable_postprocessing:
        return segments
    segs = [dict(s) for s in segments]
    if use_dedupe:
        segs = deduplicate_segments(segs)
    for s in segs:
        s["text"] = postprocess_text(s.get("text", ""))
    return [s for s in segs if (s.get("text") or "").strip()]


def _run_transcribe_sync(
    wav_path: Path,
    language: str,
    specialty: str | None = None,
) -> list[dict]:
    """동기 전사 실행 (스레드 풀용)."""
    segments = transcribe_with_segments(wav_path, language=language, specialty=specialty)
    return _apply_postprocess(segments, use_dedupe=True)


def _run_transcribe_words_sync(
    wav_path: Path,
    language: str,
    specialty: str | None = None,
) -> list[dict]:
    return transcribe_words_flat(wav_path, language=language, specialty=specialty)


def _run_diarization_sync(wav_path: Path) -> list[dict]:
    return run_diarization(wav_path)


def _run_diar_chunk_transcribe_sync(
    wav_path: Path,
    language: str,
    specialty: str | None = None,
) -> list[dict]:
    """화자 분리 후 구간별 전사. 구간당 단일 화자 라벨."""
    diar_segments = sorted(run_diarization(wav_path), key=lambda d: d["start"])
    out: list[dict] = []
    for d in diar_segments:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        p = Path(tmp.name)
        try:
            chunk_path, t0, _t1 = slice_audio_meta(
                wav_path, d["start"], d["end"], p,
                padding_sec=DIAR_CHUNK_PADDING_SEC,
            )
            segs = transcribe_with_segments(
                chunk_path,
                language=language,
                specialty=specialty,
                vad_filter=False,
            )
            spk = d["speaker"]
            for s in segs:
                g_start = float(s["start"]) + t0
                g_end = float(s["end"]) + t0
                text = _filter_transcription_hallucinations((s.get("text") or "").strip())
                if text:
                    out.append({
                        "start": g_start,
                        "end": g_end,
                        "text": text,
                        "speaker": spk,
                        "confidence": s.get("confidence", 0.5),
                    })
        finally:
            p.unlink(missing_ok=True)
    return _apply_postprocess(_smooth_speaker_labels(out), use_dedupe=False)


async def transcribe_with_diarization(
    wav_path: Path,
    language: str = "ko",
    specialty: str | None = None,
) -> list[dict]:
    """
    전사 + 화자분리 병합.

    speaker_alignment_mode (env SPEAKER_ALIGNMENT_MODE):
    - segment: Whisper 세그먼트에 화자 라벨만 부착 (기존)
    - word_diar: 단어 타임스탬프 + diar로 구간 재구성
    - diar_chunk: diar 구간별 개별 전사 (느리지만 구간=화자에 유리)
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()
    mode = settings.speaker_alignment_mode

    if not settings.enable_diarization:
        segments = await loop.run_in_executor(
            None,
            lambda: _run_transcribe_sync(wav_path, language, specialty),
        )
        return [{"speaker": None, **s} for s in segments]

    if mode == "diar_chunk":
        merged = await loop.run_in_executor(
            None,
            lambda: _run_diar_chunk_transcribe_sync(wav_path, language, specialty),
        )
        logger.info("diar_chunk: %d 세그먼트", len(merged))
        return merged

    if mode == "word_diar":
        words_task = loop.run_in_executor(
            None,
            lambda: _run_transcribe_words_sync(wav_path, language, specialty),
        )
        diar_task = loop.run_in_executor(None, lambda: _run_diarization_sync(wav_path))
        words, diar_segments = await asyncio.gather(words_task, diar_task)
        if not words:
            logger.warning("word_diar: 단어 타임스탬프 없음 → segment 방식으로 폴백")
            transcript_segments = await loop.run_in_executor(
                None,
                lambda: _run_transcribe_sync(wav_path, language, specialty),
            )
            merged = _assign_speaker_to_segments(transcript_segments, diar_segments)
        else:
            merged = _merge_words_with_diarization(words, diar_segments)
            merged = _apply_postprocess(merged, use_dedupe=False)
        logger.info("word_diar: %d 세그먼트", len(merged))
        return merged

    # segment (기본)
    trans_task = loop.run_in_executor(
        None,
        lambda: _run_transcribe_sync(wav_path, language, specialty),
    )
    diar_task = loop.run_in_executor(None, lambda: _run_diarization_sync(wav_path))
    transcript_segments, diar_segments = await asyncio.gather(trans_task, diar_task)

    logger.info(
        "segment: 전사 %d구간, 화자분리 %d구간 → 매칭",
        len(transcript_segments),
        len(diar_segments),
    )
    merged = _assign_speaker_to_segments(transcript_segments, diar_segments)

    try:
        from app.services.segment_recovery import recover_missing_segments
        recovery = await loop.run_in_executor(
            None,
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
            merged = _assign_speaker_to_segments(
                recovery["merged_segments"], diar_segments,
            )
    except Exception:
        logger.debug("누락 복구 모듈 미사용 (정상 동작)")

    return merged
