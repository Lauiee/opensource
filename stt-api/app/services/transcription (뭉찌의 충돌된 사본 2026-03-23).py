"""STT 백엔드: Faster-Whisper (오픈소스 로컬).

v2.0 개선사항:
  - condition_on_previous_text=False (환각 전파 차단)
  - temperature fallback [0.0, 0.2, 0.4] (다양한 후보 생성)
  - VAD 민감도 향상 (짧은 발화 감지)
  - 신뢰도 스코어링 (avg_logprob, no_speech_prob, compression_ratio)
  - 환각 세그먼트 자동 필터링
  - 오디오 전처리 (노이즈 제거, 정규화) 연동
"""

import math
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

    1. medical_dict.json의 prompt_terms 및 specialty_prompts 로드
    2. 진료과 힌트가 있으면 해당 진료과 전문 용어 추가
    3. 없으면 일반 의료 프롬프트 사용
    """
    try:
        import json
        dict_path = Path(get_settings().medical_dict_path)
        if dict_path.exists():
            data = json.loads(dict_path.read_text(encoding="utf-8"))
            base_terms = data.get("prompt_terms", [])

            # 진료과별 전문 용어 추가
            specialty_terms = []
            specialty_prompts = data.get("specialty_prompts", {})
            if specialty and specialty_prompts:
                # 진료과명 → 키 매핑
                _SPECIALTY_MAP = {
                    "안과": "ophthalmology",
                    "간담도": "hepatobiliary", "소화기내과": "hepatobiliary",
                    "정신건강의학과": "psychiatry", "정신과": "psychiatry",
                    "정형외과": "orthopedics",
                }
                key = _SPECIALTY_MAP.get(specialty, specialty.lower())
                if key in specialty_prompts:
                    specialty_terms = specialty_prompts[key]

            all_terms = list(dict.fromkeys(base_terms + specialty_terms))  # 중복 제거

            if all_terms:
                prefix = "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다."
                if specialty:
                    prefix = f"{specialty} 진료 상담 대화입니다. 의사와 환자가 대화합니다."
                return f"{prefix} {', '.join(all_terms)}"
    except Exception:
        pass

    return _FW_DEFAULT_PROMPT

_BEAM_TO_REP_PENALTY: dict[int, float] = {5: 1.3, 10: 1.2, 15: 1.1, 20: 1.1}


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


# ──────────────────────────────────────────────
# 신뢰도(confidence) 스코어링
# ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """수치 안정 시그모이드."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def compute_confidence(
    avg_logprob: float,
    no_speech_prob: float,
    compression_ratio: float,
) -> float:
    """세그먼트 신뢰도 0.0~1.0 계산.

    - avg_logprob: 평균 로그 확률 (높을수록 확신)
    - no_speech_prob: 비음성 확률 (높을수록 잡음)
    - compression_ratio: 압축 비율 (높을수록 환각)
    """
    # 기본 점수: logprob 기반 (중심 = -0.5)
    base = _sigmoid((avg_logprob + 0.5) * 3.0)

    # 비음성 페널티
    if no_speech_prob > 0.3:
        base *= (1.0 - no_speech_prob * 0.8)

    # 압축 비율 페널티 (환각 신호)
    if compression_ratio > 2.0:
        penalty = max(0.2, 1.0 - (compression_ratio - 2.0) * 0.3)
        base *= penalty

    return max(0.0, min(1.0, base))


def _is_hallucination(
    no_speech_prob: float,
    compression_ratio: float,
    avg_logprob: float,
) -> bool:
    """환각 세그먼트 판별.

    높은 비음성 확률 + 높은 압축 비율 = 거의 확실한 환각.
    """
    # 조건 1: 비음성 + 높은 압축 (가장 확실)
    if no_speech_prob > 0.5 and compression_ratio > 2.4:
        return True
    # 조건 2: 매우 높은 압축 비율 (반복 텍스트)
    if compression_ratio > 3.0:
        return True
    # 조건 3: 매우 낮은 logprob + 높은 비음성
    if avg_logprob < -1.5 and no_speech_prob > 0.4:
        return True
    return False


# ──────────────────────────────────────────────
# 오디오 전처리
# ──────────────────────────────────────────────

def _preprocess_audio_if_enabled(wav_path: str | Path) -> Path:
    """오디오 전처리 (노이즈 제거 + 정규화).

    원본을 보존하고 전처리된 파일을 별도 경로에 저장.
    """
    wav_path = Path(wav_path)

    try:
        from app.services.audio_preprocessor import preprocess_audio

        # 전처리된 파일은 .preprocessed.wav 로 저장
        preprocessed_path = wav_path.with_suffix(".preprocessed.wav")
        preprocess_audio(wav_path, preprocessed_path)

        if preprocessed_path.exists() and preprocessed_path.stat().st_size > 0:
            logger.info("오디오 전처리 완료: %s", preprocessed_path.name)
            return preprocessed_path
    except Exception as e:
        logger.warning("오디오 전처리 실패 (원본 사용): %s", e)

    return wav_path


# ──────────────────────────────────────────────
# 메인 전사 함수
# ──────────────────────────────────────────────

def transcribe_with_segments(
    wav_path: str | Path,
    language: str = "ko",
    model: str | None = None,
    specialty: str | None = None,
) -> list[dict]:
    """
    Faster-Whisper로 전사 (v2.0 — 강화된 파라미터).

    Args:
        wav_path: WAV 파일 경로
        language: 언어 코드 (기본: "ko")
        model: 미사용 (API 호환)
        specialty: 진료과 힌트 (예: "정형외과") — 초기 프롬프트 최적화

    Returns:
        list of {
            "start": float, "end": float, "text": str,
            "confidence": float,  # 0.0~1.0 신뢰도
            "avg_logprob": float,
            "no_speech_prob": float,
            "compression_ratio": float,
        }
    """
    settings = get_settings()
    lang = language or settings.default_language

    _ = model  # unused, kept for API compatibility
    fw_model = _get_faster_whisper_model()
    beam_size = settings.faster_whisper_beam_size
    rep_penalty = _BEAM_TO_REP_PENALTY.get(beam_size, 1.2)

    # 진료과에 맞는 초기 프롬프트 사용
    initial_prompt = get_initial_prompt(specialty)

    # ── 오디오 전처리 (노이즈 제거 + 정규화) ──
    processed_wav = _preprocess_audio_if_enabled(wav_path)

    # ── Whisper 전사 (v2.1 최적화 파라미터) ──
    # condition_on_previous_text=True 유지: 의료 대화 문맥 활용이 더 유리
    # 환각 방지는 repetition_penalty + 후처리에서 처리
    segments, _info = fw_model.transcribe(
        str(processed_wav),
        language=lang,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,   # 원본 유지 (안정적 세그먼트 분할)
            "speech_pad_ms": 400,             # 원본 유지
            "threshold": 0.45,                # 0.5→0.45: 약간만 민감하게 (과도 분할 방지)
        },
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,       # 의료 문맥 유지 (환각 방지는 후처리에서)
        temperature=0.0,                       # 단일 온도 (일관성)
        no_speech_threshold=0.5,               # 0.6→0.5: 적당히 완화
        repetition_penalty=rep_penalty,        # 1.4: 반복 억제 강화
        hallucination_silence_threshold=1.5,   # 2.0→1.5: 환각 감지 약간 강화
    )

    out: list[dict] = []
    dropped_hallucinations = 0
    dropped_empty = 0

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            dropped_empty += 1
            continue

        # 신뢰도 메트릭 추출
        avg_lp = getattr(seg, "avg_logprob", -0.5)
        nsp = getattr(seg, "no_speech_prob", 0.0)
        cr = getattr(seg, "compression_ratio", 1.0)

        # 환각 세그먼트 필터링
        if _is_hallucination(nsp, cr, avg_lp):
            dropped_hallucinations += 1
            logger.debug(
                "환각 필터링: [%.1f-%.1f] '%s' (nsp=%.2f, cr=%.2f, lp=%.2f)",
                seg.start, seg.end, text[:30], nsp, cr, avg_lp,
            )
            continue

        confidence = compute_confidence(avg_lp, nsp, cr)

        out.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "confidence": round(confidence, 3),
            "avg_logprob": round(avg_lp, 4),
            "no_speech_prob": round(nsp, 4),
            "compression_ratio": round(cr, 4),
        })

    if dropped_hallucinations > 0:
        logger.info("환각 세그먼트 %d개 필터링됨", dropped_hallucinations)

    # 전처리 임시 파일 정리
    if processed_wav != Path(wav_path):
        try:
            processed_wav.unlink(missing_ok=True)
        except Exception:
            pass

    return out
