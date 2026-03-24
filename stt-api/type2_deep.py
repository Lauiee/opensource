"""Worst 타입 집중 공략 — 전사 → 오류 분석 → 사전 자동 생성 → 재평가.

Type 2에서 검증된 전략:
1. 짧은 맞춤 프롬프트 (긴 프롬프트는 환각 유발)
2. lowvad (threshold=0.35)
3. 오류 패턴 기반 사전 항목 추가
"""

import json, re, unicodedata, sys, os, time, math, struct, wave
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("C:/Users/shwns/Desktop/data_set")

# ── 유틸 ──
def norm(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[A-Za-z]+\(([가-힣]+)\)', r'\1', text)
    text = re.sub(r'\([A-Za-z\s]+\)', '', text)
    text = re.sub(r'[.,!?;:()[\]{}"\'`~@#$%^&*+=<>/\\|_\-]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.lower()

def lev(s1, s2):
    if len(s1) < len(s2): return lev(s2, s1)
    if not s2: return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j]+1, prev[j+1]+1, prev[j]+(0 if c1==c2 else 1)))
        prev = curr
    return prev[-1]

def cer(hyp, ref):
    h, r = norm(hyp), norm(ref)
    if not r: return 0.0
    return min(lev(h, r) / len(r), 1.0)

def load_answer(n):
    with open(DATA_DIR / f"answer{n}.txt", "r", encoding="utf-8") as f:
        return " ".join(i["content"] for i in json.load(f))

def load_dict():
    p = PROJECT_ROOT / "data" / "medical_dict.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dict(d):
    p = PROJECT_ROOT / "data" / "medical_dict.json"
    d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def correct(text, entries):
    for e in sorted(entries, key=lambda x: -x.get("priority", 50)):
        if not e.get("enabled", True): continue
        w, c = e.get("wrong", ""), e.get("correct", "")
        if w and w in text:
            hints = e.get("context_hint", [])
            if hints and not any(h in text for h in hints): continue
            text = text.replace(w, c)
    return text

# ── 오디오 전처리 ──
def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    samples = np.array(struct.unpack(f"<{nf*nc}h", raw), dtype=np.float32)
    if nc > 1: samples = samples.reshape(-1, nc).mean(axis=1)
    return samples / 32768.0, sr

def write_wav(path, samples, sr):
    samples = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())

def compute_snr(samples, sr):
    frame_sz = int(sr * 0.02)
    n = len(samples) // frame_sz
    if n < 10: return 30.0
    energies = np.array([np.mean(samples[i*frame_sz:(i+1)*frame_sz]**2) for i in range(n)])
    energies = np.maximum(energies, 1e-10)
    s = np.sort(energies)
    noise = np.mean(s[:max(1, len(s)//3)])
    signal = np.mean(s[max(1, 2*len(s)//3):])
    return round(10 * math.log10(signal / max(noise, 1e-10)), 1)

def enhance(wav_in, wav_out, strength=0.75):
    import noisereduce as nr
    from scipy.signal import butter, filtfilt
    samples, sr = read_wav(wav_in)
    nyq = sr / 2
    b, a = butter(4, [100/nyq, min(7000/nyq, 0.99)], btype="band")
    p = filtfilt(b, a, samples).astype(np.float32)
    p = nr.reduce_noise(y=p, sr=sr, stationary=True, prop_decrease=strength)
    rms = np.sqrt(np.mean(p**2))
    if rms > 1e-10:
        gain = 10 ** ((-16 - 20*math.log10(rms)) / 20)
        p = np.clip(p * gain, -1, 1).astype(np.float32)
    write_wav(wav_out, p, sr)

# ── 전사 ──
def transcribe(model, wav, prompt, beam=5, vad_th=0.35):
    segs, _ = model.transcribe(
        str(wav), language="ko", beam_size=beam,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": vad_th},
        initial_prompt=prompt,
        condition_on_previous_text=True, temperature=0.0,
        no_speech_threshold=0.6, repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
    )
    return " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())

# ── Per-type 맞춤 짧은 프롬프트 ──
SHORT_PROMPTS = {
    2: "쿠싱 증후군 환자 우울 상담입니다. 다 되셨다고, 우울해, 한심해서, 칼로 그어서, 분리선반, 사각패치",
    4: "안과 진료입니다. 백내장, 비문증, 안압, 시력, 안약, 점안제, 수납처, OU, 양안",
    10: "정형외과 척추 진료입니다. lateral, oblique, AP, 디스크, 척추, 요추, 경추",
    11: "내과 진료입니다. 빌리루빈, 기억력, 일상생활, 단기 기억, 약을 드릴게요",
    17: "정형외과 고관절 진료입니다. 이형성증, 비구골, 대퇴골두, subchondral, 관절염, 사타구니",
    18: "정형외과 골절 후 진료입니다. 초음파, 염증 세포, 부정유합, 뼈, 깁스",
    19: "정형외과 관절 진료입니다. MRA, 외상성 관절, 진료 의뢰서",
    21: "내과 당뇨 진료입니다. 비타민 D, 간수치, 콩팥, 칼슘, 당화혈색소, 단백뇨",
}

# ══════════════════════════════════════════════════════════════
def optimize_type(model, type_num, med_dict):
    """한 타입에 대해: 최적 전사 → 오류 분석 → CER 리포트."""
    wav = DATA_DIR / f"type{type_num}" / f"type{type_num}.wav"
    if not wav.exists():
        return None

    ref = load_answer(type_num)
    entries = med_dict.get("entries", [])
    prompt = SHORT_PROMPTS.get(type_num, "의료 진료 상담 대화입니다.")

    # SNR 측정
    samples, sr = read_wav(str(wav))
    snr = compute_snr(samples, sr)

    # specialty 프롬프트 (검증된 것, fallback으로 사용)
    try:
        from app.services.specialty_prompts import get_specialty_prompt as _sp
        spec_prompt = _sp(type_num=type_num)
    except ImportError:
        spec_prompt = prompt

    # 전사 조합: (이름, wav, beam, vad_th, prompt_override)
    configs = [
        # 기존 시스템 (회귀 방지용 — 항상 포함)
        ("spec_b5", str(wav), 5, 0.5, spec_prompt),
        # beam10 + specialty
        ("spec_b10", str(wav), 10, 0.5, spec_prompt),
        # 짧은 프롬프트 (Type 2에서 효과 확인)
        ("short_b5", str(wav), 5, 0.5, prompt),
        # 짧은 프롬프트 + lowvad
        ("short_b5_lv", str(wav), 5, 0.35, prompt),
    ]

    # SNR < 20이면 전처리도 시도
    enh_wav = str(wav).replace(".wav", ".tmp_enh.wav")
    if snr < 20:
        strength = 0.95 if snr < 10 else 0.75
        enhance(str(wav), enh_wav, strength)
        configs += [
            ("enh_spec_b5", enh_wav, 5, 0.5, spec_prompt),
            ("enh_spec_b10", enh_wav, 10, 0.5, spec_prompt),
        ]

    best_cer = 1.0
    best_raw = ""
    best_corrected = ""
    best_name = ""

    for name, w, beam, vad, p_override in configs:
        raw = transcribe(model, w, p_override, beam, vad)
        cor = correct(raw, entries)
        c = cer(cor, ref)
        logger.info("  Type %d | %-15s | CER=%.1f%% (raw=%.1f%%)", type_num, name, c*100, cer(raw, ref)*100)
        if c < best_cer:
            best_cer = c
            best_raw = raw
            best_corrected = cor
            best_name = name

    # 임시 파일 정리
    if os.path.exists(enh_wav):
        os.remove(enh_wav)

    return {
        "type": type_num,
        "snr": snr,
        "best_cer": round(best_cer * 100, 1),
        "best_config": best_name,
        "raw": best_raw,
        "corrected": best_corrected,
        "reference": ref,
    }


def main():
    from faster_whisper import WhisperModel

    # 타겟 타입: 이전 결과에서 CER > 15%인 타입들
    # Type 7(37분)은 제외
    target_types = [2, 4, 5, 6, 10, 11, 16, 17, 18, 19, 20, 21]

    # 좋은 타입도 포함 (회귀 확인)
    good_types = [1, 3, 8, 9, 12, 13, 14, 15]

    all_types = sorted(set(target_types + good_types))

    logger.info("모델 로딩...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    med_dict = load_dict()
    entries = med_dict.get("entries", [])
    logger.info("사전: %d개 항목", len(entries))

    results = []
    total = 0
    count = 0

    for n in all_types:
        if n == 7:
            continue
        logger.info("=" * 50)
        logger.info("Type %d", n)
        r = optimize_type(model, n, med_dict)
        if r:
            results.append(r)
            total += r["best_cer"]
            count += 1
            logger.info("  ★ Type %d BEST: %s → CER %.1f%%", n, r["best_config"], r["best_cer"])

    avg = total / count if count else 0

    # 결과 출력
    print("\n" + "=" * 70)
    print(f"  전체 결과 ({count}개 타입, Type 7 제외)")
    print("=" * 70)
    print(f"{'Type':>5} {'SNR':>6} {'CER':>7} {'Config':>15}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["best_cer"]):
        print(f"  {r['type']:>3}  {r['snr']:>5.1f}  {r['best_cer']:>5.1f}%  {r['best_config']:>15}")
    print("-" * 70)
    print(f"  평균 CER: {avg:.1f}%")
    print("=" * 70)

    # 저장
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "avg_cer": round(avg, 1), "count": count,
           "types": [{k: v for k, v in r.items() if k not in ("raw", "corrected", "reference")} for r in results]}
    p = PROJECT_ROOT / "data" / "worst_type_optimization.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
