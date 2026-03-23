"""STT 백엔드: Faster-Whisper (오픈소스 로컬)."""

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import logging

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


def get_initial_prompt(specialty: str | None = None) -> str:
    """진료과에 맞는 Whisper 초기 프롬프트 생성.

    1. medical_dict.json의 prompt_terms 로드 (있으면)
    2. 진료과 힌트가 있으면 해당 진료과 강조
    3. 없으면 기본 정형외과 프롬프트 사용
    """
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

    return _FW_DEFAULT_PROMPT

_BEAM_TO_REP_PENALTY: dict[int, float] = {3: 1.2, 5: 1.2, 10: 1.2, 15: 1.1, 20: 1.1}


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
) -> list[dict]:
    """
    Faster-Whisper로 전사.

    Args:
        wav_path: WAV 파일 경로
        language: 언어 코드 (기본: "ko")
        model: 미사용 (API 호환)
        specialty: 진료과 힌트 (예: "정형외과") — 초기 프롬프트 최적화

    Returns list of {"start": float, "end": float, "text": str}.
    """
    settings = get_settings()
    lang = language or settings.default_language

    _ = model  # unused, kept for API compatibility
    fw_model = _get_faster_whisper_model()
    beam_size = settings.faster_whisper_beam_size
    rep_penalty = _BEAM_TO_REP_PENALTY.get(beam_size, 1.05)

    # 진료과에 맞는 초기 프롬프트 사용
    initial_prompt = get_initial_prompt(specialty)

    segments, _info = fw_model.transcribe(
        str(wav_path),
        language=lang,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 500,
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
            out.append({"start": seg.start, "end": seg.end, "text": text})
    return out
