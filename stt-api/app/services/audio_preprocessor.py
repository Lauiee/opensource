"""오디오 전처리: 노이즈 제거, 정규화, 필터링.

Whisper에 입력하기 전에 오디오 품질을 개선하여
저품질 음성(Type2 등)에서의 인식률을 높인다.

파이프라인:
  1. High-pass filter (80Hz) — 저주파 잡음 제거
  2. Noise reduction — 정상(stationary) 잡음 제거
  3. Dynamic range compression — 조용한 음성 증폭
  4. Peak normalization — 일관된 볼륨 수준
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def preprocess_audio(wav_path: str | Path, output_path: str | Path | None = None) -> Path:
    """오디오 전처리 파이프라인 실행.

    Args:
        wav_path: 16kHz mono WAV 파일 경로
        output_path: 출력 경로 (None이면 원본 덮어쓰기)

    Returns:
        전처리된 WAV 파일 경로
    """
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal

    wav_path = Path(wav_path)
    if output_path is None:
        output_path = wav_path
    else:
        output_path = Path(output_path)

    try:
        sample_rate, data = wavfile.read(str(wav_path))
    except Exception as e:
        logger.warning("WAV 읽기 실패, 전처리 건너뜀: %s", e)
        return wav_path

    # int16 → float64 변환
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype != np.float64:
        data = data.astype(np.float64)

    # mono 보장
    if data.ndim > 1:
        data = data.mean(axis=1)

    original_rms = np.sqrt(np.mean(data ** 2))
    if original_rms < 1e-8:
        logger.warning("오디오가 거의 무음, 전처리 건너뜀")
        return wav_path

    # ── 1단계: High-pass filter (80Hz) ──
    # 저주파 잡음 (HVAC, 책상 진동 등) 제거
    try:
        sos = signal.butter(4, 80.0, btype='highpass', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos, data)
    except Exception as e:
        logger.debug("High-pass filter 실패: %s", e)

    # ── 2단계: Noise reduction ──
    # 정상(stationary) 잡음 프로파일 기반 제거
    try:
        import noisereduce as nr
        data = nr.reduce_noise(
            y=data,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.75,  # 75% 감소 (너무 강하면 음성 왜곡)
            n_fft=2048,
            hop_length=512,
        )
    except ImportError:
        logger.debug("noisereduce 미설치, 노이즈 제거 건너뜀")
    except Exception as e:
        logger.debug("노이즈 제거 실패: %s", e)

    # ── 3단계: Dynamic range compression ──
    # 조용한 구간의 음성을 증폭 (환자 음성이 작은 경우 대비)
    try:
        data = _dynamic_range_compression(data, sample_rate)
    except Exception as e:
        logger.debug("동적 범위 압축 실패: %s", e)

    # ── 4단계: Peak normalization (-3dB) ──
    peak = np.max(np.abs(data))
    if peak > 1e-8:
        target_peak = 10 ** (-3.0 / 20.0)  # -3dB ≈ 0.708
        data = data * (target_peak / peak)

    # 클리핑 방지
    data = np.clip(data, -1.0, 1.0)

    # float64 → int16 변환 후 저장
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(str(output_path), sample_rate, data_int16)

    new_rms = np.sqrt(np.mean(data ** 2))
    logger.info(
        "오디오 전처리 완료: RMS %.4f→%.4f, peak=%.4f",
        original_rms, new_rms, np.max(np.abs(data)),
    )

    return output_path


def _dynamic_range_compression(
    data: np.ndarray,
    sample_rate: int,
    window_ms: int = 500,
    target_rms: float = 0.08,
    max_gain: float = 6.0,
) -> np.ndarray:
    """RMS 기반 동적 범위 압축.

    조용한 구간(환자 음성)을 target_rms까지 증폭하되
    max_gain을 초과하지 않는다.
    """
    window_size = int(sample_rate * window_ms / 1000)
    if window_size < 1:
        return data

    result = data.copy()
    num_windows = len(data) // window_size

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment = data[start:end]

        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 1e-6:
            continue  # 무음 구간은 증폭하지 않음

        gain = min(target_rms / rms, max_gain)
        if gain > 1.2:  # 20% 이상 증폭이 필요한 구간만
            # 부드러운 적용 (급격한 볼륨 변화 방지)
            gain = 1.0 + (gain - 1.0) * 0.7
            result[start:end] = segment * gain

    # 남은 구간 처리
    remainder = len(data) % window_size
    if remainder > 0:
        start = num_windows * window_size
        segment = data[start:]
        rms = np.sqrt(np.mean(segment ** 2))
        if rms > 1e-6:
            gain = min(target_rms / rms, max_gain)
            if gain > 1.2:
                gain = 1.0 + (gain - 1.0) * 0.7
                result[start:] = segment * gain

    return result
