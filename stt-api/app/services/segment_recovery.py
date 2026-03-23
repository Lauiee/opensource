"""누락 세그먼트 복구 시스템.

VAD(Voice Activity Detection) 필터링으로 인해 통째로 누락된
대화 구간을 탐지하고, 해당 구간만 다른 파라미터로 재전사하여 복구한다.

Phase 7 구현: 세그먼트 재시도 + 갭 분석 + 슬라이싱 재전사
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. 갭 분석: 전사 세그먼트 사이의 빈 구간 탐지
# ──────────────────────────────────────────────────────────────────────

def find_gaps(
    segments: list[dict],
    audio_duration: float,
    min_gap_sec: float = 1.0,
) -> list[dict]:
    """전사 결과에서 누락 가능성이 있는 갭(빈 구간)을 탐지.

    Args:
        segments: 전사 세그먼트 리스트 [{"start": float, "end": float, "text": str}]
        audio_duration: 전체 오디오 길이(초)
        min_gap_sec: 이 초 이상의 갭만 보고 (기본 1초)

    Returns:
        갭 리스트 [{"start": float, "end": float, "duration": float, "position": str}]
    """
    if not segments:
        return [{"start": 0.0, "end": audio_duration, "duration": audio_duration, "position": "전체"}]

    sorted_segs = sorted(segments, key=lambda s: s["start"])
    gaps = []

    # 오디오 시작 ~ 첫 세그먼트
    if sorted_segs[0]["start"] > min_gap_sec:
        gap_dur = sorted_segs[0]["start"]
        gaps.append({
            "start": 0.0,
            "end": sorted_segs[0]["start"],
            "duration": gap_dur,
            "position": "시작부분",
        })

    # 세그먼트 사이 갭
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i]["end"]
        gap_end = sorted_segs[i + 1]["start"]
        gap_dur = gap_end - gap_start

        if gap_dur >= min_gap_sec:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration": gap_dur,
                "position": f"세그먼트 {i+1}~{i+2} 사이",
            })

    # 마지막 세그먼트 ~ 오디오 끝
    if audio_duration - sorted_segs[-1]["end"] > min_gap_sec:
        gap_dur = audio_duration - sorted_segs[-1]["end"]
        gaps.append({
            "start": sorted_segs[-1]["end"],
            "end": audio_duration,
            "duration": gap_dur,
            "position": "끝부분",
        })

    return gaps


# ──────────────────────────────────────────────────────────────────────
# 2. 오디오 슬라이싱: 갭 구간만 추출
# ──────────────────────────────────────────────────────────────────────

def slice_audio(
    wav_path: str | Path,
    start_sec: float,
    end_sec: float,
    output_path: str | Path | None = None,
    padding_sec: float = 0.5,
) -> Path:
    """WAV 파일에서 특정 구간만 추출.

    Args:
        wav_path: 원본 WAV 파일 경로
        start_sec: 시작 시간(초)
        end_sec: 종료 시간(초)
        output_path: 출력 경로 (None이면 임시 파일)
        padding_sec: 앞뒤 여유 (기본 0.5초)

    Returns:
        추출된 WAV 파일 경로
    """
    import wave
    import tempfile

    wav_path = Path(wav_path)
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    else:
        output_path = Path(output_path)

    with wave.open(str(wav_path), 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        total_duration = total_frames / sr

        # 패딩 적용
        actual_start = max(0, start_sec - padding_sec)
        actual_end = min(total_duration, end_sec + padding_sec)

        start_frame = int(actual_start * sr)
        end_frame = int(actual_end * sr)
        n_frames = end_frame - start_frame

        wf.setpos(start_frame)
        frames = wf.readframes(n_frames)

    with wave.open(str(output_path), 'wb') as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(sr)
        out.writeframes(frames)

    logger.info("오디오 슬라이스: %.1f~%.1fs → %s", actual_start, actual_end, output_path)
    return output_path


def get_audio_duration(wav_path: str | Path) -> float:
    """WAV 파일의 총 길이(초) 반환."""
    import wave
    with wave.open(str(wav_path), 'rb') as wf:
        return wf.getnframes() / wf.getframerate()


# ──────────────────────────────────────────────────────────────────────
# 3. 갭 구간 재전사: 다른 VAD 파라미터로 재시도
# ──────────────────────────────────────────────────────────────────────

def retranscribe_gap(
    wav_path: str | Path,
    gap: dict,
    language: str = "ko",
    specialty: str | None = None,
) -> list[dict]:
    """갭 구간을 더 민감한 VAD 파라미터로 재전사.

    일반 전사에서 놓친 구간을 복구하기 위해:
    - VAD threshold를 낮춤 (0.5 → 0.3)
    - min_silence_duration을 늘림 (500 → 800ms)
    - speech_pad를 늘림 (400 → 600ms)

    Args:
        wav_path: 원본 WAV 파일 경로
        gap: 갭 정보 {"start": float, "end": float}
        language: 언어 코드
        specialty: 진료과 힌트

    Returns:
        복구된 세그먼트 리스트
    """
    import tempfile

    # 갭 구간 오디오 추출
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    gap_audio = slice_audio(wav_path, gap["start"], gap["end"], tmp.name)

    try:
        from app.services.transcription import _get_faster_whisper_model, get_initial_prompt

        fw_model = _get_faster_whisper_model()
        initial_prompt = get_initial_prompt(specialty)

        # 더 민감한 파라미터로 전사 시도
        segments_iter, _info = fw_model.transcribe(
            str(gap_audio),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 800,
                "speech_pad_ms": 600,
                "threshold": 0.3,  # 더 민감하게
            },
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
            temperature=0.0,
            no_speech_threshold=0.5,  # 더 관대하게
            repetition_penalty=1.2,
            hallucination_silence_threshold=3.0,
        )

        recovered = []
        padding_offset = min(0.5, gap["start"])  # 패딩 보정
        time_offset = gap["start"] - padding_offset

        for seg in segments_iter:
            text = (seg.text or "").strip()
            if text and len(text) > 1:  # 1글자 이하는 노이즈
                recovered.append({
                    "start": round(seg.start + time_offset, 2),
                    "end": round(seg.end + time_offset, 2),
                    "text": text,
                    "recovered": True,  # 복구된 세그먼트 마킹
                })

        logger.info(
            "갭 복구 결과: %.1f~%.1fs → %d개 세그먼트 복구",
            gap["start"], gap["end"], len(recovered),
        )
        return recovered

    except Exception as e:
        logger.warning("갭 복구 실패 (%.1f~%.1fs): %s", gap["start"], gap["end"], e)
        return []

    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 4. 전체 복구 파이프라인
# ──────────────────────────────────────────────────────────────────────

def recover_missing_segments(
    wav_path: str | Path,
    original_segments: list[dict],
    language: str = "ko",
    specialty: str | None = None,
    min_gap_sec: float = 1.5,
    max_retries: int = 2,
) -> dict:
    """누락 세그먼트 전체 복구 파이프라인.

    1. 오디오 전체 길이 확인
    2. 갭(빈 구간) 분석
    3. 각 갭에 대해 민감한 VAD로 재전사
    4. 복구된 세그먼트를 원본에 병합 (시간순 정렬)

    Args:
        wav_path: WAV 파일 경로
        original_segments: 원래 전사 결과
        language: 언어 코드
        specialty: 진료과 힌트
        min_gap_sec: 이 초 이상의 갭만 복구 시도
        max_retries: 동일 갭에 대한 최대 재시도 횟수

    Returns:
        {
            "merged_segments": [...],   # 원본 + 복구 세그먼트 (시간순)
            "gaps_found": int,          # 발견된 갭 수
            "gaps_recovered": int,      # 복구 성공한 갭 수
            "segments_recovered": int,  # 복구된 세그먼트 수
            "gap_details": [...]        # 각 갭 상세 정보
        }
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        logger.warning("WAV 파일 없음: %s", wav_path)
        return {
            "merged_segments": original_segments,
            "gaps_found": 0,
            "gaps_recovered": 0,
            "segments_recovered": 0,
            "gap_details": [],
        }

    # 1. 오디오 길이
    try:
        audio_duration = get_audio_duration(wav_path)
    except Exception as e:
        logger.warning("오디오 길이 확인 실패: %s", e)
        return {
            "merged_segments": original_segments,
            "gaps_found": 0,
            "gaps_recovered": 0,
            "segments_recovered": 0,
            "gap_details": [],
        }

    # 2. 갭 분석
    gaps = find_gaps(original_segments, audio_duration, min_gap_sec)
    if not gaps:
        return {
            "merged_segments": original_segments,
            "gaps_found": 0,
            "gaps_recovered": 0,
            "segments_recovered": 0,
            "gap_details": [],
        }

    logger.info("갭 분석: %d개 갭 발견 (총 %.1fs 누락 가능)", len(gaps), sum(g["duration"] for g in gaps))

    # 3. 각 갭 복구
    all_recovered = []
    gap_details = []

    for gap in gaps:
        recovered = []
        for attempt in range(max_retries):
            recovered = retranscribe_gap(wav_path, gap, language, specialty)
            if recovered:
                break

        gap_detail = {
            **gap,
            "recovered": len(recovered) > 0,
            "recovered_segments": len(recovered),
            "recovered_texts": [s["text"] for s in recovered],
        }
        gap_details.append(gap_detail)
        all_recovered.extend(recovered)

    # 4. 병합 (시간순)
    merged = list(original_segments) + all_recovered
    merged.sort(key=lambda s: s["start"])

    # 중복 제거 (시간이 겹치는 세그먼트)
    merged = _deduplicate_by_time(merged)

    gaps_recovered = sum(1 for gd in gap_details if gd["recovered"])

    logger.info(
        "복구 완료: %d개 갭 중 %d개 복구, 총 %d개 세그먼트 추가",
        len(gaps), gaps_recovered, len(all_recovered),
    )

    return {
        "merged_segments": merged,
        "gaps_found": len(gaps),
        "gaps_recovered": gaps_recovered,
        "segments_recovered": len(all_recovered),
        "gap_details": gap_details,
    }


def _deduplicate_by_time(
    segments: list[dict],
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """시간 겹침이 큰 세그먼트 중복 제거.

    같은 시간대에 원본과 복구 세그먼트가 겹치면,
    복구 세그먼트는 원본이 없는 구간에서만 유지.
    """
    if len(segments) <= 1:
        return segments

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        overlap = min(seg["end"], prev["end"]) - max(seg["start"], prev["start"])
        seg_dur = seg["end"] - seg["start"]

        if seg_dur > 0 and overlap / seg_dur > overlap_threshold:
            # 겹침이 50% 이상이면 더 긴(정보가 많은) 세그먼트 유지
            if len(seg.get("text", "")) > len(prev.get("text", "")):
                result[-1] = seg
        else:
            result.append(seg)

    return result


# ──────────────────────────────────────────────────────────────────────
# 5. 저신뢰도 세그먼트 재전사
# ──────────────────────────────────────────────────────────────────────

def retranscribe_low_confidence(
    wav_path: str | Path,
    segments: list[dict],
    confidence_threshold: float = 0.5,
    language: str = "ko",
    specialty: str | None = None,
) -> list[dict]:
    """저신뢰도 세그먼트만 다른 파라미터로 재전사.

    Args:
        wav_path: WAV 파일 경로
        segments: 전사 결과 (confidence 필드 포함)
        confidence_threshold: 이 값 미만인 세그먼트만 재시도
        language: 언어 코드
        specialty: 진료과 힌트

    Returns:
        개선된 세그먼트 리스트
    """
    import tempfile

    result = []
    retried = 0

    for seg in segments:
        conf = seg.get("confidence") or seg.get("avg_logprob")
        if conf is not None and conf < confidence_threshold:
            # 해당 구간만 추출하여 재전사
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                gap_audio = slice_audio(wav_path, seg["start"], seg["end"], tmp.name)

                from app.services.transcription import _get_faster_whisper_model, get_initial_prompt
                fw_model = _get_faster_whisper_model()
                initial_prompt = get_initial_prompt(specialty)

                # 다른 temperature로 시도
                for temp in [0.2, 0.4]:
                    new_segs, _ = fw_model.transcribe(
                        str(gap_audio),
                        language=language,
                        beam_size=10,  # 더 큰 beam
                        vad_filter=False,  # VAD 비활성화 (이미 잘라놓았으므로)
                        initial_prompt=initial_prompt,
                        temperature=temp,
                        repetition_penalty=1.3,
                    )
                    new_texts = []
                    for ns in new_segs:
                        t = (ns.text or "").strip()
                        if t:
                            new_texts.append(t)

                    if new_texts:
                        new_text = " ".join(new_texts)
                        # 원본보다 더 나은지 간단 휴리스틱 체크
                        if len(new_text) >= len(seg.get("text", "")) * 0.5:
                            seg = {**seg, "text": new_text, "retried": True}
                            retried += 1
                            break

                Path(tmp.name).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("저신뢰도 재전사 실패: %s", e)

        result.append(seg)

    if retried:
        logger.info("저신뢰도 재전사: %d개 세그먼트 개선", retried)

    return result
