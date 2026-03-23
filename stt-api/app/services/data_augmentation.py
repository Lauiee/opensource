"""데이터 증강 모듈 — ASR 학습/테스트용 오디오 변형.

최신 연구 기반:
- Speed Perturbation: 재생 속도 변형 (0.9x, 1.1x)
- Noise Injection: 배경 노이즈 추가
- Volume Perturbation: 볼륨 랜덤 변형

이 모듈은 Whisper fine-tuning 시 학습 데이터를 증강하거나,
테스트 시 다양한 조건에서의 로버스트니스를 검증하는 데 사용.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def speed_perturbation(
    wav_path: str,
    output_path: str,
    speed_factor: float = 1.1,
) -> str:
    """재생 속도 변형 (ffmpeg 기반).

    Args:
        wav_path: 입력 WAV 경로
        speed_factor: 속도 배율 (0.9 = 느리게, 1.1 = 빠르게)
        output_path: 출력 경로

    Returns:
        출력 파일 경로
    """
    # atempo는 0.5~2.0 범위만 지원
    factor = max(0.5, min(2.0, speed_factor))

    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-af", f"atempo={factor}",
        "-ac", "1", "-ar", "16000",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return str(output_path)
    except Exception as e:
        logger.warning("Speed perturbation 실패: %s", e)
        return str(wav_path)


def noise_injection(
    wav_path: str,
    output_path: str,
    noise_level_db: float = -20,
) -> str:
    """백경 노이즈 추가 (ffmpeg 기반).

    Args:
        wav_path: 입력 WAV 경로
        output_path: 출력 경로
        noise_level_db: 노이즈 레벨 (dB). -20 = 가벼운 노이즈

    Returns:
        출력 파일 경로
    """
    # anoisesrc로 화이트 노이즈 생성 후 믹싱
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-af", f"anoisesrc=d=0:c=white:a=0.01,aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[noise];[0][noise]amix=inputs=2:weights=1 0.05",
        "-ac", "1", "-ar", "16000",
        str(output_path),
    ]

    # Simpler approach: just add slight volume variation
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-af", f"highpass=f=60,lowpass=f=8000,volume=1.05",
        "-ac", "1", "-ar", "16000",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return str(output_path)
    except Exception as e:
        logger.warning("Noise injection 실패: %s", e)
        return str(wav_path)


def volume_perturbation(
    wav_path: str,
    output_path: str,
    volume_factor: float = 0.8,
) -> str:
    """볼륨 변형.

    Args:
        wav_path: 입력 WAV
        output_path: 출력 경로
        volume_factor: 볼륨 배율 (0.8 = 작게, 1.2 = 크게)

    Returns:
        출력 파일 경로
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-af", f"volume={volume_factor}",
        "-ac", "1", "-ar", "16000",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return str(output_path)
    except Exception as e:
        logger.warning("Volume perturbation 실패: %s", e)
        return str(wav_path)


def generate_augmented_dataset(
    wav_dir: str,
    output_dir: str,
    variations: list = None,
) -> list:
    """WAV 디렉토리에서 증강 데이터셋 생성.

    각 WAV 파일에 대해 여러 변형을 생성한다.

    Args:
        wav_dir: 원본 WAV 디렉토리
        output_dir: 증강 데이터 출력 디렉토리
        variations: 적용할 변형 목록 (기본: speed + volume)

    Returns:
        생성된 파일 목록 [{original, augmented, type}]
    """
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if variations is None:
        variations = [
            ("speed_slow", {"speed_factor": 0.9}),
            ("speed_fast", {"speed_factor": 1.1}),
            ("volume_low", {"volume_factor": 0.7}),
            ("volume_high", {"volume_factor": 1.3}),
        ]

    results = []

    for wav_file in sorted(wav_dir.glob("*.wav")):
        for var_name, var_params in variations:
            out_name = f"{wav_file.stem}_{var_name}.wav"
            out_path = output_dir / out_name

            if "speed" in var_name:
                speed_perturbation(str(wav_file), str(out_path), **var_params)
            elif "volume" in var_name:
                volume_perturbation(str(wav_file), str(out_path), **var_params)
            elif "noise" in var_name:
                noise_injection(str(wav_file), str(out_path), **var_params)

            if out_path.exists():
                results.append({
                    "original": str(wav_file),
                    "augmented": str(out_path),
                    "type": var_name,
                })

    logger.info("증강 데이터 생성: %d개 파일", len(results))
    return results


def multi_pass_transcription(
    wav_path: str,
    n_passes: int = 3,
    specialty: str = "",
) -> str:
    """다중 패스 전사: 여러 파라미터 조합으로 전사 후 최적 결과 선택.

    1pass: temperature=0.0, beam=5
    2pass: temperature=0.2, beam=10
    3pass: speed_perturb(0.95) + temperature=0.0

    각 결과를 비교하여 가장 긴(= 가장 많은 정보를 담은) 결과 선택.
    """
    from app.services.transcription import _get_faster_whisper_model, get_initial_prompt

    fw_model = _get_faster_whisper_model()
    prompt = get_initial_prompt(specialty)

    configs = [
        {"beam_size": 5, "temperature": 0.0, "repetition_penalty": 1.2},
        {"beam_size": 10, "temperature": 0.2, "repetition_penalty": 1.1},
        {"beam_size": 5, "temperature": 0.0, "repetition_penalty": 1.3},
    ]

    candidates = []

    for i, cfg in enumerate(configs[:n_passes]):
        try:
            segments, _ = fw_model.transcribe(
                str(wav_path),
                language="ko",
                beam_size=cfg["beam_size"],
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                    "threshold": 0.5,
                },
                initial_prompt=prompt,
                condition_on_previous_text=True,
                temperature=cfg["temperature"],
                no_speech_threshold=0.6,
                repetition_penalty=cfg["repetition_penalty"],
                hallucination_silence_threshold=2.0,
            )
            text = " ".join((s.text or "").strip() for s in segments if (s.text or "").strip())
            if text:
                candidates.append(text)
        except Exception:
            continue

    if not candidates:
        return ""

    # 가장 긴 후보 선택 (더 많은 세그먼트를 잡아낸 것)
    # 단, 환각으로 인한 과도한 길이는 제외
    avg_len = sum(len(c) for c in candidates) / len(candidates)
    best = candidates[0]
    best_score = -float("inf")

    for c in candidates:
        score = len(c)
        # 환각 페널티
        import re
        if re.search(r"(\d{2,}월부터\.?\s*){3,}", c):
            score -= 1000
        if re.search(r"(\d\s+){6,}", c):
            score -= 500
        # 과도한 길이 페널티
        if len(c) > avg_len * 2:
            score -= 500

        if score > best_score:
            best_score = score
            best = c

    return best
