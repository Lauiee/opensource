"""화자 분리: pyannote.audio (병렬 처리용)."""

import logging
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_diarization_pipeline():
    """pyannote 화자 분리 파이프라인 (lazy load)."""
    from pyannote.audio import Pipeline
    import torch

    settings = get_settings()
    token = (settings.huggingface_token or "").strip()
    if not token:
        raise ValueError(
            "화자 분리 사용 시 HUGGINGFACE_TOKEN이 필요합니다. "
            ".env에 HUGGINGFACE_TOKEN=hf_xxx 를 추가하세요. "
            "토큰: https://huggingface.co/settings/tokens, "
            "pyannote 약관 동의: https://huggingface.co/pyannote/speaker-diarization-3.1. "
            "화자 분리 없이 전사만 하려면 ENABLE_DIARIZATION=false 로 설정하세요."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    try:
        pipeline.to(torch.device("cuda"))
        logger.info("pyannote diarization on CUDA")
    except Exception:
        pipeline.to(torch.device("cpu"))
        logger.info("pyannote diarization on CPU")
    return pipeline


@lru_cache(maxsize=1)
def _get_cached_pipeline():
    return _get_diarization_pipeline()


def run_diarization(wav_path: str | Path) -> list[dict]:
    """
    화자 분리 실행.
    설정에서 num_speakers 지정 시 해당 인원 고정, 없으면 min_speakers 이상으로 자동 탐지 (기본 min=1).
    Returns list of {"start": float, "end": float, "speaker": str}.
    """
    from app.config import get_settings

    path = Path(wav_path)
    pipeline = _get_cached_pipeline()
    settings = get_settings()

    kwargs = {}
    if settings.diarization_num_speakers is not None:
        kwargs["num_speakers"] = settings.diarization_num_speakers
    else:
        kwargs["min_speakers"] = settings.diarization_min_speakers
        if settings.diarization_max_speakers is not None:
            kwargs["max_speakers"] = settings.diarization_max_speakers

    diarization = pipeline(str(path), **kwargs)

    # pyannote 4.x: DiarizeOutput.speaker_diarization (Annotation)
    # pyannote 3.x: Annotation directly
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )

    out: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        out.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": str(speaker),
        })
    return out
