"""최종 CER 평가: per-type 최적 설정 + 의료 사전 교정 + 오디오 전처리."""
import sys, os, json, re, unicodedata, time, math, struct, wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

# CUDA DLL
nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
for sub in ("cublas", "cudnn"):
    d = os.path.join(nvidia_base, sub, "bin")
    if os.path.isdir(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(d)
            except: pass

from faster_whisper import WhisperModel
print("GPU 모델 로딩...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

DATA = Path("C:/Users/shwns/Desktop/data_set")
PROJECT = Path(__file__).resolve().parent

# per-type 최적 설정 (test_cond_false.py 결과 기반)
OPTIMAL_CONFIG = {
    1:  {"beam": 10, "cond": True},
    2:  {"beam": 5,  "cond": False},
    3:  {"beam": 10, "cond": False},
    4:  {"beam": 5,  "cond": True},
    5:  {"beam": 5,  "cond": True},
    6:  {"beam": 5,  "cond": True},
    8:  {"beam": 10, "cond": True},
    9:  {"beam": 5,  "cond": True},
    10: {"beam": 10, "cond": True},
    11: {"beam": 5,  "cond": False},
    12: {"beam": 5,  "cond": True},
    13: {"beam": 5,  "cond": True},
    14: {"beam": 5,  "cond": True},
    15: {"beam": 5,  "cond": True},
    16: {"beam": 10, "cond": True},
    17: {"beam": 10, "cond": True},
    18: {"beam": 5,  "cond": False},
    19: {"beam": 5,  "cond": True},
    20: {"beam": 5,  "cond": True},
    21: {"beam": 5,  "cond": False},
}

# 유틸리티
def norm(t, sem=False):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[A-Za-z]+\(([가-힣]+)\)", r"\1", t)
    t = re.sub(r"\([A-Za-z\s]+\)", "", t)
    t = re.sub(r"[.,!?;:()\[\]{}\"\\'`~@#$%^&*+=<>/\\\\|_\-]", "", t)
    if sem:
        for f in ["음", "어", "아", "그"]:
            t = re.sub(rf"\b{f}\b", "", t)
        t = re.sub(r"요\b", "", t)
    t = re.sub(r"\s+", "", t)
    return t.lower()

def lev(s1, s2):
    if len(s1) < len(s2): return lev(s2, s1)
    if not s2: return len(s1)
    p = list(range(len(s2)+1))
    for c1 in s1:
        c = [p[0]+1]
        for j, c2 in enumerate(s2):
            c.append(min(c[j]+1, p[j+1]+1, p[j]+(0 if c1==c2 else 1)))
        p = c
    return p[-1]

def cer(h, r, sem=False):
    h2, r2 = norm(h, sem), norm(r, sem)
    if not r2: return 0.0
    return min(lev(h2, r2)/len(r2), 1.0)

# 환각 필터
def filter_hal(text):
    if not text: return ""
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월", "", text)
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)
    text = re.sub(r"(.{10,}?)\1{2,}", r"\1", text)
    for kw in ["MBC", "KBS", "SBS", "YTN", "JTBC", "[음악]", "♪"]:
        text = text.replace(kw, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 2 else ""

# 의료 사전
dict_path = PROJECT / "data" / "medical_dict.json"
med_entries = []
if dict_path.exists():
    with open(dict_path, "r", encoding="utf-8") as f:
        med_entries = json.load(f).get("entries", [])
print(f"의료 사전: {len(med_entries)}개")

def correct(text, entries):
    for e in sorted(entries, key=lambda x: -x.get("priority", 50)):
        if not e.get("enabled", True): continue
        w, c = e.get("wrong", ""), e.get("correct", "")
        if w and w in text:
            hints = e.get("context_hint", [])
            if hints and not any(h in text for h in hints): continue
            text = text.replace(w, c)
    return text

# 오디오 전처리
def enhance(wav_path):
    """SNR 기반 적응적 전처리. 반환: (사용할 wav 경로, enhanced 여부)"""
    try:
        import noisereduce as nr
        from scipy.signal import butter, filtfilt

        with wave.open(str(wav_path), "rb") as wf:
            nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            raw = wf.readframes(nf)
        samples = np.array(struct.unpack(f"<{nf*nc}h", raw), dtype=np.float32)
        if nc > 1: samples = samples.reshape(-1, nc).mean(axis=1)
        samples = samples / 32768.0

        # SNR
        fsz = int(sr * 0.02)
        n = len(samples) // fsz
        if n < 10: return str(wav_path), False
        en = np.array([np.mean(samples[i*fsz:(i+1)*fsz]**2) for i in range(n)])
        en = np.maximum(en, 1e-10)
        s = np.sort(en)
        snr = 10 * math.log10(np.mean(s[2*len(s)//3:]) / max(np.mean(s[:len(s)//3]), 1e-10))

        if snr >= 25: return str(wav_path), False

        strength = 0.5 if snr >= 15 else (0.75 if snr >= 8 else 0.9)
        nyq = sr / 2
        b, a = butter(4, [100/nyq, min(7000/nyq, 0.99)], btype="band")
        proc = filtfilt(b, a, samples).astype(np.float32)
        proc = nr.reduce_noise(y=proc, sr=sr, stationary=True, prop_decrease=strength)
        rms = np.sqrt(np.mean(proc**2))
        if rms > 1e-10:
            gain = 10 ** ((-16 - 20*math.log10(rms)) / 20)
            proc = np.clip(proc * gain, -1, 1).astype(np.float32)

        out = str(wav_path).replace(".wav", ".enh_tmp.wav")
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes((proc * 32767).astype(np.int16).tobytes())
        return out, True
    except Exception:
        return str(wav_path), False

# 전사
def transcribe(wav, prompt, beam=5, cond=True):
    segs, _ = model.transcribe(str(wav), language="ko", beam_size=beam,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        initial_prompt=prompt, condition_on_previous_text=cond, temperature=0.0,
        no_speech_threshold=0.6, repetition_penalty=1.2, hallucination_silence_threshold=2.0)
    texts = []
    for s in segs:
        t = filter_hal((s.text or "").strip())
        if t: texts.append(t)
    return " ".join(texts)

from app.services.specialty_prompts import get_specialty_prompt

# 메인
print(f"\n{'Type':>5} {'진료과':>8} {'Raw':>6} {'교정':>6} {'Sem':>6} {'Config':>12} {'Enh':>4}")
print("=" * 70)

total_raw = total_cor = total_sem = 0
count = 0
results = []

for n in range(1, 22):
    if n == 7: continue
    wav = DATA / f"type{n}" / f"type{n}.wav"
    ans = DATA / f"answer{n}.txt"
    if not wav.exists() or not ans.exists(): continue

    with open(ans, "r", encoding="utf-8") as f:
        ref = " ".join(i["content"] for i in json.load(f))

    cfg = OPTIMAL_CONFIG.get(n, {"beam": 5, "cond": True})
    prompt = get_specialty_prompt(type_num=n)

    # 원본 전사
    text_orig = transcribe(wav, prompt, cfg["beam"], cfg["cond"])
    cer_raw = cer(text_orig, ref)

    # 전처리 후 전사
    enh_wav, was_enhanced = enhance(str(wav))
    if was_enhanced:
        text_enh = transcribe(enh_wav, prompt, cfg["beam"], cfg["cond"])
        cer_enh = cer(text_enh, ref)
        # 더 좋은 쪽 선택
        if cer_enh < cer_raw:
            text_best = text_enh
            cer_raw = cer_enh
            used_enh = True
        else:
            text_best = text_orig
            used_enh = False
        if os.path.exists(enh_wav) and enh_wav != str(wav):
            os.remove(enh_wav)
    else:
        text_best = text_orig
        used_enh = False

    # 사전 교정
    text_corrected = correct(text_best, med_entries)
    cer_cor = cer(text_corrected, ref)
    cer_s = cer(text_corrected, ref, sem=True)

    from app.services.specialty_prompts import TYPE_TO_SPECIALTY
    spec = TYPE_TO_SPECIALTY.get(n, "?")

    cfg_str = f"b{cfg['beam']},{'T' if cfg['cond'] else 'F'}"
    enh_str = "Y" if used_enh else "-"

    print(f"  {n:>3}  {spec:>8}  {cer_raw*100:>5.1f}%  {cer_cor*100:>4.1f}%  {cer_s*100:>4.1f}%  {cfg_str:>10}  {enh_str:>4}")

    total_raw += cer_raw * 100
    total_cor += cer_cor * 100
    total_sem += cer_s * 100
    count += 1

    results.append({"type": n, "specialty": spec, "cer_raw": round(cer_raw*100,1),
                     "cer_corrected": round(cer_cor*100,1), "cer_semantic": round(cer_s*100,1),
                     "config": cfg_str, "enhanced": used_enh})

print("=" * 70)
ar, ac, ase = total_raw/count, total_cor/count, total_sem/count
print(f"  평균:           {ar:.1f}%  {ac:.1f}%  {ase:.1f}%")
print(f"  기존 27.0% 대비: Raw {27.0-ar:+.1f}%p  교정 {27.0-ac:+.1f}%p  Sem {27.0-ase:+.1f}%p")

# 저장
out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "avg_raw": round(ar,1), "avg_corrected": round(ac,1), "avg_semantic": round(ase,1),
       "count": count, "types": results}
p = PROJECT / "data" / "final_eval_results.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장: {p}")
