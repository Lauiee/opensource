"""오디오 다운로드 및 WAV 변환."""

import subprocess
from pathlib import Path

import httpx
from pydub import AudioSegment


async def download_audio(url: str, dest_path: Path) -> None:
    """Download audio file from URL."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        dest_path.write_bytes(response.content)


def ensure_wav_16k_mono(input_path: str | Path) -> Path:
    """Convert audio to 16kHz mono WAV format for Whisper and downstream processing."""
    input_path = Path(input_path)

    out_path = input_path.with_suffix(".converted.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", str(out_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode(errors='ignore')}")

    return out_path


def get_audio_duration(file_path: str | Path) -> float:
    """Get audio duration in seconds."""
    audio = AudioSegment.from_file(str(file_path))
    return len(audio) / 1000.0


# ──────────────────────────────────────────────────────────────────────
# 오디오 전처리 강화 (노이즈 감소 + 음량 정규화)
# ──────────────────────────────────────────────────────────────────────

def preprocess_audio(input_path: str | Path) -> Path:
    """오디오 전처리: 음량 정규화 + 노이즈 감쇠 + DC offset 제거.

    ffmpeg 기반으로 구현 (무료, 별도 라이브러리 불필요).

    Args:
        input_path: 입력 WAV 파일 (16kHz mono)

    Returns:
        전처리된 WAV 파일 경로
    """
    input_path = Path(input_path)
    out_path = input_path.with_suffix(".preprocessed.wav")

    # ffmpeg audio filter chain:
    # 1. highpass=f=80   → 80Hz 이하 저주파 노이즈 제거 (에어컨, 전기 험)
    # 2. lowpass=f=7500  → 7.5kHz 이상 고주파 노이즈 감쇠 (한국어 음성에 충분)
    # 3. afftdn=nf=-25   → FFT 기반 노이즈 감쇠 (노이즈 플로어 -25dB)
    # 4. loudnorm        → EBU R128 음량 정규화
    # 5. aresample=16000 → 16kHz 리샘플링 보장
    af_chain = (
        "highpass=f=80,"
        "lowpass=f=7500,"
        "afftdn=nf=-25,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "aresample=16000"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", af_chain,
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            # 전처리 실패 시 원본 반환 (degradation 방지)
            return input_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return input_path

    return out_path


def ensure_wav_16k_mono_enhanced(input_path: str | Path) -> Path:
    """WAV 변환 + 오디오 전처리 통합 파이프라인.

    1단계: WAV 16kHz mono 변환
    2단계: 노이즈 감소 + 음량 정규화

    Args:
        input_path: 원본 오디오 파일 경로

    Returns:
        전처리 완료된 WAV 파일 경로
    """
    # 1) WAV 변환
    wav_path = ensure_wav_16k_mono(input_path)

    # 2) 전처리 (실패 시 원본 WAV 반환)
    try:
        enhanced = preprocess_audio(wav_path)
        return enhanced
    except Exception:
        return wav_path
