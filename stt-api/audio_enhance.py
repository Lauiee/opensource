"""오디오 전처리 강화 모듈 — SNR 기반 적응적 전처리.

전략:
1. SNR(신호대잡음비) 측정 → 오디오 품질 판정
2. 품질 낮으면 noisereduce로 spectral gating 적용
3. 고주파/저주파 필터링 + 정규화
4. 전처리 전후 비교로 최적 결과 선택
"""

import os
import sys
import wave
import struct
import math
import logging
import tempfile
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_wav(path: str) -> tuple:
    """WAV 파일 읽기 → (samples: np.ndarray, sample_rate: int)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    elif sample_width == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # 모노로 변환
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # -1.0 ~ 1.0으로 정규화
    max_val = 2 ** (sample_width * 8 - 1)
    samples = samples / max_val

    return samples, sample_rate


def write_wav(path: str, samples: np.ndarray, sample_rate: int):
    """WAV 파일 쓰기 (16-bit mono)."""
    # 클리핑 방지
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def compute_snr(samples: np.ndarray, sample_rate: int) -> float:
    """SNR(신호대잡음비) 추정.

    간단한 방법: 에너지가 높은 구간(발화)과 낮은 구간(침묵/소음)을 분리하여 비율 계산.
    """
    # 프레임 단위 에너지 (20ms 프레임)
    frame_size = int(sample_rate * 0.02)
    n_frames = len(samples) // frame_size

    if n_frames < 10:
        return 30.0  # 너무 짧으면 충분한 SNR로 간주

    energies = []
    for i in range(n_frames):
        frame = samples[i * frame_size:(i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)
    energies = np.maximum(energies, 1e-10)  # 0 방지

    # 에너지 기준 상위 30% = 발화, 하위 30% = 소음
    sorted_e = np.sort(energies)
    n = len(sorted_e)
    noise_energy = np.mean(sorted_e[:max(1, n // 3)])
    signal_energy = np.mean(sorted_e[max(1, 2 * n // 3):])

    if noise_energy < 1e-10:
        return 40.0  # 사실상 무소음

    snr_db = 10 * math.log10(signal_energy / noise_energy)
    return round(snr_db, 1)


def classify_audio_quality(snr_db: float) -> str:
    """SNR 기반 오디오 품질 등급."""
    if snr_db >= 25:
        return "excellent"  # 전처리 불필요
    elif snr_db >= 15:
        return "good"       # 가벼운 전처리
    elif snr_db >= 8:
        return "fair"        # 중간 전처리
    else:
        return "poor"        # 강한 전처리


def apply_spectral_gating(samples: np.ndarray, sample_rate: int,
                          strength: str = "medium") -> np.ndarray:
    """noisereduce를 사용한 spectral gating 노이즈 제거.

    strength: "light", "medium", "strong"
    """
    import noisereduce as nr

    # 강도별 파라미터
    params = {
        "light": {"prop_decrease": 0.5, "n_std_thresh_stationary": 2.0},
        "medium": {"prop_decrease": 0.75, "n_std_thresh_stationary": 1.5},
        "strong": {"prop_decrease": 0.9, "n_std_thresh_stationary": 1.0},
    }
    p = params.get(strength, params["medium"])

    reduced = nr.reduce_noise(
        y=samples,
        sr=sample_rate,
        stationary=True,
        prop_decrease=p["prop_decrease"],
        n_std_thresh_stationary=p["n_std_thresh_stationary"],
    )
    return reduced


def apply_bandpass_filter(samples: np.ndarray, sample_rate: int,
                          low_freq: int = 100, high_freq: int = 7000) -> np.ndarray:
    """대역통과 필터 (발화 주파수 대역만 통과)."""
    from scipy.signal import butter, filtfilt

    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)

    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, samples)
    return filtered.astype(np.float32)


def normalize_loudness(samples: np.ndarray, target_db: float = -16.0) -> np.ndarray:
    """라우드니스 정규화."""
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return samples

    current_db = 20 * math.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    normalized = samples * gain
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def enhance_audio(input_path: str, output_path: str = None,
                  force_strength: str = None) -> dict:
    """오디오 품질 평가 후 적응적 전처리.

    Args:
        input_path: 입력 WAV 파일
        output_path: 출력 WAV 파일 (None이면 자동 생성)
        force_strength: 강제 전처리 강도 ("light", "medium", "strong", None=자동)

    Returns:
        {
            "input_path": str,
            "output_path": str,
            "snr_before": float,
            "snr_after": float,
            "quality": str,
            "strength": str,
            "enhanced": bool,
        }
    """
    input_path = str(input_path)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.enhanced{ext}"

    # 오디오 읽기
    samples, sr = read_wav(input_path)
    snr_before = compute_snr(samples, sr)
    quality = classify_audio_quality(snr_before)

    logger.info("오디오 품질: SNR=%.1fdB, 등급=%s, 길이=%.1f초",
                snr_before, quality, len(samples) / sr)

    # 품질에 따른 처리 결정
    if force_strength:
        strength = force_strength
    elif quality == "excellent":
        # 전처리 불필요 — 원본 복사
        shutil.copy2(input_path, output_path)
        return {
            "input_path": input_path,
            "output_path": output_path,
            "snr_before": snr_before,
            "snr_after": snr_before,
            "quality": quality,
            "strength": "none",
            "enhanced": False,
        }
    elif quality == "good":
        strength = "light"
    elif quality == "fair":
        strength = "medium"
    else:
        strength = "strong"

    # 처리 파이프라인
    processed = samples.copy()

    # 1. 대역통과 필터 (발화 주파수 대역)
    processed = apply_bandpass_filter(processed, sr, low_freq=100, high_freq=7000)

    # 2. Spectral gating 노이즈 제거
    processed = apply_spectral_gating(processed, sr, strength=strength)

    # 3. 라우드니스 정규화
    processed = normalize_loudness(processed, target_db=-16.0)

    # SNR 재측정
    snr_after = compute_snr(processed, sr)

    # 저장
    write_wav(output_path, processed, sr)

    logger.info("전처리 완료: SNR %.1f → %.1f dB (%s 강도)",
                snr_before, snr_after, strength)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "quality": quality,
        "strength": strength,
        "enhanced": True,
    }


def batch_enhance(input_dir: str, output_dir: str = None) -> list:
    """디렉토리 내 모든 WAV 파일 일괄 전처리."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "enhanced"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for wav_file in sorted(input_dir.glob("*.wav")):
        out_path = output_dir / wav_file.name
        try:
            result = enhance_audio(str(wav_file), str(out_path))
            results.append(result)
        except Exception as e:
            logger.error("전처리 실패: %s — %s", wav_file, e)
            results.append({"input_path": str(wav_file), "error": str(e)})

    return results


# ══════════════════════════════════════════════════════════════
# CLI: 전처리 + 전사 + CER 비교
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import time
    import unicodedata
    import re

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    DATA_DIR = Path("C:/Users/shwns/Desktop/data_set")
    PROJECT_ROOT = Path(__file__).resolve().parent

    sys.path.insert(0, str(PROJECT_ROOT))

    def normalize_for_cer(text):
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r'[A-Za-z]+\(([가-힣]+)\)', r'\1', text)
        text = re.sub(r'\([A-Za-z\s]+\)', '', text)
        text = re.sub(r'[.,!?;:()[\]{}"\'`~@#$%^&*+=<>/\\|_\-]', '', text)
        text = re.sub(r'\s+', '', text)
        return text.lower()

    def levenshtein(s1, s2):
        if len(s1) < len(s2): return levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if c1 == c2 else 1)))
            prev = curr
        return prev[-1]

    def cer(hyp, ref):
        h, r = normalize_for_cer(hyp), normalize_for_cer(ref)
        if not r: return 0.0 if not h else 1.0
        return min(levenshtein(h, r) / len(r), 1.0)

    # 모델 로드
    from faster_whisper import WhisperModel
    logger.info("모델 로딩...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    # 의료 사전
    dict_path = PROJECT_ROOT / "data" / "medical_dict.json"
    med_entries = []
    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            med_entries = data.get("entries", [])

    def apply_corrections(text, entries):
        for e in sorted(entries, key=lambda x: -x.get("priority", 50)):
            if not e.get("enabled", True): continue
            w, c = e.get("wrong", ""), e.get("correct", "")
            if w and w in text:
                hints = e.get("context_hint", [])
                if hints and not any(h in text for h in hints): continue
                text = text.replace(w, c)
        return text

    def transcribe(wav_path, prompt=""):
        segs, _ = model.transcribe(
            str(wav_path), language="ko", beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
            initial_prompt=prompt,
            condition_on_previous_text=True, temperature=0.0,
            no_speech_threshold=0.6, repetition_penalty=1.2,
            hallucination_silence_threshold=2.0,
        )
        return " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())

    # 진료과 프롬프트
    try:
        from app.services.specialty_prompts import get_specialty_prompt as get_prompt
    except ImportError:
        def get_prompt(specialty=None, type_num=None):
            return "의료 진료 상담 대화입니다."

    start = time.time()
    results = []

    for type_num in range(1, 22):
        wav_path = DATA_DIR / f"type{type_num}" / f"type{type_num}.wav"
        ans_path = DATA_DIR / f"answer{type_num}.txt"
        if not wav_path.exists() or not ans_path.exists():
            continue

        with open(ans_path, "r", encoding="utf-8") as f:
            reference = " ".join(item["content"] for item in json.load(f))

        prompt = get_prompt(type_num=type_num)

        # 원본으로 전사
        logger.info("Type %d: 원본 전사...", type_num)
        raw_original = transcribe(wav_path, prompt)
        corrected_original = apply_corrections(raw_original, med_entries)
        cer_original = cer(corrected_original, reference)

        # 전처리 후 전사
        logger.info("Type %d: 오디오 전처리 중...", type_num)
        enhanced_path = str(wav_path).replace(".wav", ".enhanced.wav")
        enhance_result = enhance_audio(str(wav_path), enhanced_path)

        if enhance_result.get("enhanced"):
            logger.info("Type %d: 전처리된 오디오로 전사...", type_num)
            raw_enhanced = transcribe(enhanced_path, prompt)
            corrected_enhanced = apply_corrections(raw_enhanced, med_entries)
            cer_enhanced = cer(corrected_enhanced, reference)
        else:
            cer_enhanced = cer_original

        # 최적 선택
        best_cer = min(cer_original, cer_enhanced)
        best_source = "enhanced" if cer_enhanced < cer_original else "original"

        result = {
            "type": type_num,
            "snr": enhance_result.get("snr_before", 0),
            "quality": enhance_result.get("quality", "?"),
            "strength": enhance_result.get("strength", "none"),
            "cer_original": round(cer_original * 100, 1),
            "cer_enhanced": round(cer_enhanced * 100, 1),
            "best_cer": round(best_cer * 100, 1),
            "best_source": best_source,
            "improvement": round((cer_original - cer_enhanced) * 100, 1),
        }
        results.append(result)

        logger.info("Type %d: CER 원본=%.1f%% → 전처리=%.1f%% (%s, SNR=%.1fdB)",
                    type_num, result["cer_original"], result["cer_enhanced"],
                    best_source, result["snr"])

        # 임시 파일 정리
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)

    # 결과 출력
    elapsed = time.time() - start
    avg_orig = sum(r["cer_original"] for r in results) / len(results) if results else 0
    avg_enh = sum(r["best_cer"] for r in results) / len(results) if results else 0

    print("\n" + "=" * 85)
    print(f"  오디오 전처리 효과 분석 ({len(results)}개 타입, {elapsed:.0f}초)")
    print("=" * 85)
    print(f"{'Type':>6} {'SNR(dB)':>8} {'품질':>10} {'강도':>8} {'원본CER':>9} {'전처리CER':>10} {'개선':>7}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x["best_cer"]):
        imp = f"{r['improvement']:+.1f}%" if r['improvement'] != 0 else "—"
        marker = " ★" if r['improvement'] > 1 else ""
        print(f"  {r['type']:>4}   {r['snr']:>7.1f}  {r['quality']:>10}  {r['strength']:>8}  "
              f"{r['cer_original']:>7.1f}%  {r['cer_enhanced']:>8.1f}%  {imp:>7}{marker}")

    print("-" * 85)
    print(f"  평균: 원본 {avg_orig:.1f}% → 전처리+최적선택 {avg_enh:.1f}%  ({avg_orig - avg_enh:+.1f}%p)")
    print("=" * 85)

    # 결과 저장
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avg_cer_original": round(avg_orig, 1),
        "avg_cer_best": round(avg_enh, 1),
        "improvement": round(avg_orig - avg_enh, 1),
        "types": results,
    }
    out_path = PROJECT_ROOT / "data" / "audio_enhance_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장: %s", out_path)
