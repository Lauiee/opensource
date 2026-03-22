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
