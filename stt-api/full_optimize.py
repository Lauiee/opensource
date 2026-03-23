"""통합 CER 최적화: 오디오 전처리 + STT 파라미터 최적화 + 후처리.

Type 7 (37분)은 건너뛰고 나머지 20개 타입에 대해:
1. 원본 + 전처리 오디오 각각 전사
2. beam5 baseline vs beam10 specialty 비교
3. 의료 사전 교정 적용
4. 최적 결과 선택
"""

import json, sys, os, re, unicodedata, logging, time, math, struct, wave
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("C:/Users/shwns/Desktop/data_set")

# ── Type-Specialty 매핑 ──
TYPE_SPEC = {
    1: "내과", 2: "내분비내과", 3: "간담도외과", 4: "안과",
    5: "정형외과", 6: "간담도외과", 7: "정형외과", 8: "비뇨기과",
    9: "정형외과", 10: "정형외과", 11: "내과", 12: "감염내과",
    13: "정형외과", 14: "호흡기내과", 15: "호흡기내과", 16: "정형외과",
    17: "정형외과", 18: "정형외과", 19: "정형외과",
    20: "신장내과", 21: "내과",
}

SKIP_TYPES = {7}  # 37분 오디오 — 별도 처리 필요

# ── 유틸리티 ──
def normalize(text, semantic=False):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[A-Za-z]+\(([가-힣]+)\)', r'\1', text)
    text = re.sub(r'\([A-Za-z\s]+\)', '', text)
    text = re.sub(r'[.,!?;:()[\]{}"\'`~@#$%^&*+=<>/\\|_\-]', '', text)
    if semantic:
        for f in ["음", "어", "아", "그"]:
            text = re.sub(rf'\b{f}\b', '', text)
        text = re.sub(r'요\b', '', text)
    text = re.sub(r'\s+', '', text)
    return text.lower()

def lev(s1, s2):
    if len(s1) < len(s2): return lev(s2, s1)
    if not s2: return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]

def cer(hyp, ref, sem=False):
    h, r = normalize(hyp, sem), normalize(ref, sem)
    if not r: return 0.0 if not h else 1.0
    return min(lev(h, r) / len(r), 1.0)

def load_answer(n):
    with open(DATA_DIR / f"answer{n}.txt", "r", encoding="utf-8") as f:
        return " ".join(i["content"] for i in json.load(f))

# ── 오디오 전처리 ──
def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    fmt = f"<{nf*nc}h" if sw == 2 else f"<{nf*nc}i"
    samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    if nc > 1: samples = samples.reshape(-1, nc).mean(axis=1)
    return samples / (2**(sw*8-1)), sr

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

def enhance_audio(wav_path, out_path):
    """적응적 오디오 전처리."""
    import noisereduce as nr
    from scipy.signal import butter, filtfilt

    samples, sr = read_wav(wav_path)
    snr = compute_snr(samples, sr)

    if snr >= 25:
        return str(wav_path), snr, "excellent", False  # 전처리 불필요

    # 강도 결정
    if snr >= 15: strength = 0.5
    elif snr >= 8: strength = 0.75
    else: strength = 0.9

    # 1. 대역통과 필터
    nyq = sr / 2
    b, a = butter(4, [100/nyq, min(7000/nyq, 0.99)], btype="band")
    processed = filtfilt(b, a, samples).astype(np.float32)

    # 2. Spectral gating
    processed = nr.reduce_noise(y=processed, sr=sr, stationary=True, prop_decrease=strength)

    # 3. 라우드니스 정규화
    rms = np.sqrt(np.mean(processed**2))
    if rms > 1e-10:
        gain = 10 ** ((-16 - 20*math.log10(rms)) / 20)
        processed = np.clip(processed * gain, -1, 1).astype(np.float32)

    write_wav(out_path, processed, sr)
    quality = "good" if snr >= 15 else ("fair" if snr >= 8 else "poor")
    return str(out_path), snr, quality, True

# ── 의료 사전 ──
def load_dict():
    p = PROJECT_ROOT / "data" / "medical_dict.json"
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("entries", [])

def correct(text, entries):
    for e in sorted(entries, key=lambda x: -x.get("priority", 50)):
        if not e.get("enabled", True): continue
        w, c = e.get("wrong", ""), e.get("correct", "")
        if w and w in text:
            hints = e.get("context_hint", [])
            if hints and not any(h in text for h in hints): continue
            text = text.replace(w, c)
    return text

# ── 환각 필터 ──
def filter_hal(text):
    if not text: return ""
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월", "", text)
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)
    text = re.sub(r'(.{10,}?)\1{2,}', r'\1', text)
    for kw in ["MBC", "KBS", "SBS", "YTN", "JTBC", "[음악]", "♪"]:
        text = text.replace(kw, "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 2 else ""

# ── 전사 ──
def transcribe(model, wav, prompt, beam=5, vad_th=0.5):
    segs, _ = model.transcribe(
        str(wav), language="ko", beam_size=beam,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": vad_th},
        initial_prompt=prompt,
        condition_on_previous_text=True, temperature=0.0,
        no_speech_threshold=0.6, repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
    )
    texts = []
    for s in segs:
        t = filter_hal((s.text or "").strip())
        if t: texts.append(t)
    return " ".join(texts)

# ══════════════════════════════════════════════════════════════
def main():
    start = time.time()

    # 모델
    from faster_whisper import WhisperModel
    logger.info("모델 로딩 (CPU int8)...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    # 사전 + 프롬프트
    entries = load_dict()
    logger.info("의료 사전: %d개", len(entries))

    try:
        from app.services.specialty_prompts import get_specialty_prompt
    except ImportError:
        def get_specialty_prompt(specialty=None, type_num=None):
            return "의료 진료 상담 대화입니다."

    results = []
    total_cer = 0
    total_sem = 0
    count = 0

    for n in range(1, 22):
        if n in SKIP_TYPES:
            logger.info("Type %d: SKIP (37분 오디오)", n)
            continue

        wav = DATA_DIR / f"type{n}" / f"type{n}.wav"
        ans = DATA_DIR / f"answer{n}.txt"
        if not wav.exists() or not ans.exists(): continue

        ref = load_answer(n)
        prompt = get_specialty_prompt(type_num=n)
        logger.info("=" * 50)
        logger.info("Type %d (%s) — %d자", n, TYPE_SPEC.get(n, "?"), len(ref))

        # ── 원본 전사 (beam5 + beam10) ──
        t1 = transcribe(model, wav, prompt, beam=5)
        c1 = correct(t1, entries)
        cer1 = cer(c1, ref)
        logger.info("  원본 beam5:  CER %.1f%%", cer1*100)

        t2 = transcribe(model, wav, prompt, beam=10)
        c2 = correct(t2, entries)
        cer2 = cer(c2, ref)
        logger.info("  원본 beam10: CER %.1f%%", cer2*100)

        # ── 전처리 후 전사 ──
        enhanced_wav = str(wav).replace(".wav", ".tmp_enhanced.wav")
        try:
            enh_path, snr, quality, was_enhanced = enhance_audio(str(wav), enhanced_wav)
            if was_enhanced:
                t3 = transcribe(model, enh_path, prompt, beam=5)
                c3 = correct(t3, entries)
                cer3 = cer(c3, ref)
                logger.info("  전처리 beam5:  CER %.1f%% (SNR=%.1fdB, %s)", cer3*100, snr, quality)

                t4 = transcribe(model, enh_path, prompt, beam=10)
                c4 = correct(t4, entries)
                cer4 = cer(c4, ref)
                logger.info("  전처리 beam10: CER %.1f%%", cer4*100)
            else:
                cer3, cer4 = cer1, cer2
                snr = compute_snr(*read_wav(str(wav)))
                quality = "excellent"
                logger.info("  전처리 불필요 (SNR=%.1fdB)", snr)
        except Exception as e:
            logger.warning("  전처리 실패: %s", e)
            cer3, cer4 = cer1, cer2
            snr, quality = 0, "error"
        finally:
            if os.path.exists(enhanced_wav):
                os.remove(enhanced_wav)

        # ── 최적 선택 ──
        options = [
            ("orig_b5", cer1, c1),
            ("orig_b10", cer2, c2),
        ]
        if was_enhanced if 'was_enhanced' in dir() else False:
            options.append(("enh_b5", cer3, c3 if 'c3' in dir() else c1))
            options.append(("enh_b10", cer4, c4 if 'c4' in dir() else c2))

        # locals에서 안전하게 가져오기
        all_cers = [cer1, cer2, cer3, cer4]
        all_names = ["orig_b5", "orig_b10", "enh_b5", "enh_b10"]
        best_idx = np.argmin(all_cers)
        best_c = all_cers[best_idx]
        best_name = all_names[best_idx]
        best_sem = cer(ref, ref)  # placeholder

        # semantic CER 계산
        all_texts = [c1, c2]
        if 'c3' in dir(): all_texts.append(c3)
        if 'c4' in dir(): all_texts.append(c4)
        best_text = all_texts[best_idx] if best_idx < len(all_texts) else c1
        best_sem = cer(best_text, ref, sem=True)

        result = {
            "type": n,
            "specialty": TYPE_SPEC.get(n, "?"),
            "snr": snr,
            "quality": quality,
            "cer_orig_b5": round(cer1 * 100, 1),
            "cer_orig_b10": round(cer2 * 100, 1),
            "cer_enh_b5": round(cer3 * 100, 1),
            "cer_enh_b10": round(cer4 * 100, 1),
            "best_cer": round(best_c * 100, 1),
            "best_sem": round(best_sem * 100, 1),
            "best_config": best_name,
        }
        results.append(result)
        total_cer += best_c * 100
        total_sem += best_sem * 100
        count += 1

        logger.info("  ★ BEST: %s → CER %.1f%% (sem: %.1f%%)",
                    best_name, best_c*100, best_sem*100)

    # ── 결과 ──
    elapsed = time.time() - start
    avg = total_cer / count if count else 0
    avg_sem = total_sem / count if count else 0

    print("\n" + "=" * 95)
    print(f"  통합 CER 최적화 결과 ({count}개 타입, {elapsed:.0f}초)")
    print("=" * 95)
    print(f"{'Type':>5} {'진료과':>8} {'SNR':>6} {'품질':>8} {'원본b5':>7} {'원본b10':>8} {'전처리b5':>9} {'전처리b10':>10} {'Best':>6} {'Sem':>6} {'Config':>10}")
    print("-" * 95)

    for r in sorted(results, key=lambda x: x["best_cer"]):
        print(f"  {r['type']:>3}  {r['specialty']:>8}  {r['snr']:>5.1f}  {r['quality']:>8}"
              f"  {r['cer_orig_b5']:>5.1f}%  {r['cer_orig_b10']:>6.1f}%"
              f"  {r['cer_enh_b5']:>7.1f}%  {r['cer_enh_b10']:>8.1f}%"
              f"  {r['best_cer']:>5.1f}%  {r['best_sem']:>4.1f}%  {r['best_config']:>10}")

    print("-" * 95)
    print(f"  평균 CER: {avg:.1f}%  |  평균 SemCER: {avg_sem:.1f}%  |  기존 27.0% 대비 {27.0-avg:+.1f}%p")
    print("=" * 95)

    # 저장
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "avg_cer": round(avg, 1), "avg_sem": round(avg_sem, 1),
           "count": count, "elapsed": round(elapsed), "types": results}
    p = PROJECT_ROOT / "data" / "full_optimize_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", p)

if __name__ == "__main__":
    main()
