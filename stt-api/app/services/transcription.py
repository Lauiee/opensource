"""STT 백엔드: Faster-Whisper (오픈소스 로컬)."""

import os
import shutil
import sys
import wave
from functools import lru_cache
from pathlib import Path

import logging

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


_FW_DEFAULT_PROMPT = (
    "정형외과 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
    "고관절, 무릎, 척추, 디스크, 인공관절, 수술, 재활, "
    "X-ray, MRI, CT, 골절, 연골, 인대, 관절염, "
    "퇴행성, 류마티스, 스테로이드, 주사, 물리치료, "
    "통증, 저림, 부종, 염증, 감염, 항생제, "
    "대퇴골, 경골, 비골, 슬개골, 반월상연골, 십자인대, "
    "이형성증, 활액막, 사타구니, "
    "고관절 치환술, 슬관절 치환술, 관절경, "
    "진통제, 소염제, 조인스 정, 세레브렉스, 리리카, 파라마셋, "
    "처방, 약, 입원, 퇴원, 외래"
)


def get_initial_prompt(specialty: str | None = None, type_num: int | None = None) -> str:
    """진료과에 맞는 Whisper 초기 프롬프트 생성.

    우선순위:
    1. 진료과별 전문 프롬프트 세트 (specialty_prompts.py)
    2. medical_dict.json의 prompt_terms
    3. 기본 정형외과 프롬프트
    """
    # 1) 진료과별 전문 프롬프트 (가장 높은 우선순위)
    try:
        from app.services.specialty_prompts import get_specialty_prompt
        prompt = get_specialty_prompt(specialty=specialty, type_num=type_num)
        if prompt:
            logger.debug("진료과별 프롬프트 사용: specialty=%s, type_num=%s", specialty, type_num)
            return prompt
    except ImportError:
        pass

    # 2) medical_dict.json의 prompt_terms
    try:
        import json
        dict_path = Path(get_settings().medical_dict_path)
        if dict_path.exists():
            data = json.loads(dict_path.read_text(encoding="utf-8"))
            terms = data.get("prompt_terms", [])
            if terms:
                prefix = "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다."
                if specialty:
                    prefix = f"{specialty} 진료 상담 대화입니다. 의사와 환자가 대화합니다."
                return f"{prefix} {', '.join(terms)}"
    except Exception:
        pass

    # 3) 기본 프롬프트
    return _FW_DEFAULT_PROMPT

_BEAM_TO_REP_PENALTY: dict[int, float] = {5: 1.2, 10: 1.2, 15: 1.1, 20: 1.1}


def _load_wav_float32_16k_mono(wav_path: str | Path) -> np.ndarray:
    """16kHz mono PCM WAV를 float32(-1~1) 배열로 로드.

    faster-whisper가 파일 경로를 직접 디코딩할 때 일부 환경에서
    AudioDecoder NameError가 발생해, 전처리된 WAV를 직접 넘겨 우회한다.
    """
    path = Path(wav_path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if channels != 1:
        raise ValueError(f"Expected mono wav, got channels={channels}: {path}")
    if frame_rate != 16000:
        raise ValueError(f"Expected 16k wav, got fr={frame_rate}: {path}")

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    if sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        return audio

    raise ValueError(f"Unsupported WAV sample width={sample_width} bytes: {path}")


def _setup_cuda_dll_paths() -> None:
    if sys.platform != "win32":
        return
    nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
    for sub in ("cublas", "cudnn"):
        bin_dir = os.path.join(nvidia_base, sub, "bin")
        if os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass


def _apply_hf_symlink_workaround() -> None:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        import huggingface_hub.file_download as _hf_dl
        _orig = _hf_dl._create_symlink

        def _copy_fallback(src, dst, new_blob=False):
            try:
                _orig(src, dst, new_blob)
            except OSError:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy2(src, dst)

        _hf_dl._create_symlink = _copy_fallback
    except Exception:
        pass


@lru_cache(maxsize=1)
def _get_faster_whisper_model():
    _setup_cuda_dll_paths()
    _apply_hf_symlink_workaround()

    from faster_whisper import WhisperModel

    settings = get_settings()
    model_name = settings.faster_whisper_model
    compute_type = settings.faster_whisper_compute_type

    try:
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        logger.info("Faster-Whisper loaded: %s on CUDA (%s)", model_name, compute_type)
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU (int8)")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return model


def transcribe_with_segments(
    wav_path: str | Path,
    language: str = "ko",
    model: str | None = None,
    specialty: str | None = None,
    type_num: int | None = None,
) -> list[dict]:
    """
    Faster-Whisper로 전사.

    Args:
        wav_path: WAV 파일 경로
        language: 언어 코드 (기본: "ko")
        model: 미사용 (API 호환)
        specialty: 진료과 힌트 (예: "정형외과") — 초기 프롬프트 최적화
        type_num: Type 번호 (1~16) — 진료과 자동 매핑용

    Returns list of {"start": float, "end": float, "text": str}.
    """
    settings = get_settings()
    lang = language or settings.default_language

    _ = model  # unused, kept for API compatibility
    fw_model = _get_faster_whisper_model()
    beam_size = settings.faster_whisper_beam_size
    rep_penalty = _BEAM_TO_REP_PENALTY.get(beam_size, 1.05)

    # 진료과에 맞는 초기 프롬프트 사용
    initial_prompt = get_initial_prompt(specialty, type_num=type_num)

    audio_input = _load_wav_float32_16k_mono(wav_path)

    segments, _info = fw_model.transcribe(
        audio_input,
        language=lang,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
            "threshold": 0.5,
        },
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        temperature=0.0,
        no_speech_threshold=0.6,
        repetition_penalty=rep_penalty,
        hallucination_silence_threshold=2.0,
    )

    out: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            # 전사 단계 환각 필터링: 불가능한 날짜 반복 제거
            text = _filter_transcription_hallucinations(text)
            if text:
                out.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": text,
                    "confidence": _seg_confidence(seg),
                })
    return out


def _seg_confidence(seg) -> float:
    """세그먼트의 신뢰도 점수 (0.0~1.0)."""
    try:
        avg_logprob = getattr(seg, 'avg_logprob', None)
        no_speech_prob = getattr(seg, 'no_speech_prob', None)
        if avg_logprob is not None:
            # avg_logprob은 보통 -0.1(높은신뢰) ~ -1.5(낮은신뢰) 범위
            conf = max(0.0, min(1.0, 1.0 + avg_logprob))
            if no_speech_prob is not None and no_speech_prob > 0.5:
                conf *= 0.5  # 비음성 확률이 높으면 신뢰도 낮춤
            return round(conf, 3)
    except Exception:
        pass
    return 0.5


import re as _re

# 불가능한 월 패턴 (13월 이상)
_IMPOSSIBLE_MONTH = _re.compile(r"(?:1[3-9]|[2-9]\d)월")
# 연속 반복 패턴 ("N월부터" 5회 이상 반복)
_REPEATED_MONTH = _re.compile(r"(\d{1,2}월부터\.?\s*){5,}")
# 연속 숫자 나열 환각 ("1 2 3 4 5 6 7 8")
_NUMBER_SEQUENCE = _re.compile(r"(?:\d\s+){6,}\d")


def _filter_transcription_hallucinations(text: str) -> str:
    """전사 단계에서 명백한 환각을 필터링.

    - 13월 이상의 불가능한 월 표현
    - "N월부터" 5회 이상 반복
    - 숫자 나열 환각 (1 2 3 4 5 6 7 8)
    """
    if not text:
        return text

    # 불가능한 월 반복 제거
    text = _REPEATED_MONTH.sub("", text)

    # 개별 불가능한 월도 제거 (13월~99월)
    if _IMPOSSIBLE_MONTH.search(text):
        # 불가능한 월이 포함된 경우, 해당 부분만 제거
        text = _IMPOSSIBLE_MONTH.sub("", text)

    # 숫자 나열 환각 제거
    text = _NUMBER_SEQUENCE.sub("", text)

    # 정리
    text = _re.sub(r'\s+', ' ', text).strip()

    # 남은 텍스트가 너무 짧으면 환각으로 간주
    if len(text) < 2:
        return ""

    return text
