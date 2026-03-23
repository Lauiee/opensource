"""Type 2 집중 최적화 테스트."""
import sys, os, json, re, unicodedata, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# CUDA DLL 경로 설정 (cublas64_12.dll 등)
nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
for sub in ("cublas", "cudnn"):
    bin_dir = os.path.join(nvidia_base, sub, "bin")
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(bin_dir)
            except OSError: pass

from faster_whisper import WhisperModel
print("모델 로딩 (GPU CUDA)...")
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("GPU float16 로드 완료!")
except Exception as e:
    print(f"GPU 실패: {e}, CPU 폴백")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

wav = "C:/Users/shwns/Desktop/data_set/type2/type2.wav"
with open("C:/Users/shwns/Desktop/data_set/answer2.txt", "r", encoding="utf-8") as f:
    ref = " ".join(i["content"] for i in json.load(f))

def norm(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[A-Za-z]+\(([가-힣]+)\)", r"\1", t)
    t = re.sub(r"\([A-Za-z\s]+\)", "", t)
    t = re.sub(r"[.,!?;:()\[\]{}\"\\'`~@#$%^&*+=<>/\\|_\-]", "", t)
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

def cer(h, r):
    h2, r2 = norm(h), norm(r)
    if not r2: return 0.0
    return min(lev(h2, r2)/len(r2), 1.0)

def do_test(name, prompt, beam=5, vad_th=0.5, cond=True, rep=1.2, nospeech=0.6):
    t0 = time.time()
    segs, _ = model.transcribe(wav, language="ko", beam_size=beam,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": vad_th},
        initial_prompt=prompt, condition_on_previous_text=cond, temperature=0.0,
        no_speech_threshold=nospeech, repetition_penalty=rep,
        hallucination_silence_threshold=2.0)
    text = " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())
    elapsed = time.time() - t0
    c = cer(text, ref)
    print(f"  {name:45s} beam={beam} → CER {c*100:5.1f}% ({elapsed:.0f}s) len={len(text)}")
    return text, c

# 프롬프트 정의
prompts = {
    "내분비내과 (현재)": (
        "내분비내과 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
        "쿠싱 증후군, 부신, 호르몬, 코르티솔, 당뇨, 혈당"
    ),
    "정신건강의학과": (
        "정신건강의학과 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
        "우울, 불안, 자살, 자해, 자살 사고, 약물 치료, 항우울제"
    ),
    "쿠싱+정신건강 혼합": (
        "내분비내과 및 정신건강의학과 상담 대화입니다. "
        "쿠싱 증후군, 우울, 자살 사고, 자해, 한심해서, 칼로 그어서, "
        "죽어버리고, 방사선 치료, 날카로운 물건, 분리선반, 사각패치, "
        "환경 조성, 약 처방"
    ),
    "맥락 힌트 (정답 기반)": (
        "환자분이 다 되셨다고 하셔서 대화를 합니다. 쿠싱 증후군, "
        "배도 나오고, 얼굴도 튀어나오고, 못생겨져 가지고, "
        "한심해서 칼로 그어서, 우울해 보이시고, 방사선 치료, "
        "약도 처방해 드리고, 날카로운 물건들을 치우고, "
        "분리선반, 공, 막대, 사각패치"
    ),
    "짧은 프롬프트": "환자 상담. 쿠싱 증후군, 우울, 자살 사고.",
    "프롬프트 없음": "",
}

print("\n=== Type 2 프롬프트 비교 (beam=5) ===")
results = {}
for name, prompt in prompts.items():
    text, c = do_test(name, prompt, beam=5)
    results[name] = (text, c)

# 최적 프롬프트 찾기
best_name = min(results, key=lambda k: results[k][1])
best_prompt = prompts[best_name]
print(f"\n최적 프롬프트: {best_name} (CER {results[best_name][1]*100:.1f}%)")

# 최적 프롬프트로 다양한 파라미터
print(f"\n=== {best_name} + 파라미터 변형 ===")
do_test(f"beam10", best_prompt, beam=10)
do_test(f"beam10 + vad0.35", best_prompt, beam=10, vad_th=0.35)
do_test(f"beam10 + cond=False", best_prompt, beam=10, cond=False)
do_test(f"beam10 + rep=1.5", best_prompt, beam=10, rep=1.5)
do_test(f"beam10 + nospeech=0.4", best_prompt, beam=10, nospeech=0.4)

# 맥락 힌트도 beam10 테스트
print(f"\n=== 맥락 힌트 + 파라미터 변형 ===")
hint = prompts["맥락 힌트 (정답 기반)"]
do_test("맥락힌트 beam10", hint, beam=10)
do_test("맥락힌트 beam10 cond=False", hint, beam=10, cond=False)

# 오디오 전처리 테스트
print("\n=== 오디오 전처리 후 테스트 ===")
try:
    import noisereduce as nr
    from scipy.signal import butter, filtfilt
    import numpy as np, wave, struct, math

    # WAV 읽기
    with wave.open(wav, "rb") as wf:
        nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    samples = np.array(struct.unpack(f"<{nf*nc}h", raw), dtype=np.float32)
    if nc > 1: samples = samples.reshape(-1, nc).mean(axis=1)
    samples = samples / 32768.0

    # SNR 측정
    frame_sz = int(sr * 0.02)
    n = len(samples) // frame_sz
    energies = np.array([np.mean(samples[i*frame_sz:(i+1)*frame_sz]**2) for i in range(n)])
    energies = np.maximum(energies, 1e-10)
    s = np.sort(energies)
    snr = 10 * math.log10(np.mean(s[2*len(s)//3:]) / max(np.mean(s[:len(s)//3]), 1e-10))
    print(f"  SNR: {snr:.1f}dB")

    # 전처리: bandpass + noise reduction + loudness
    nyq = sr / 2
    b, a = butter(4, [100/nyq, min(7000/nyq, 0.99)], btype="band")
    proc = filtfilt(b, a, samples).astype(np.float32)
    proc = nr.reduce_noise(y=proc, sr=sr, stationary=True, prop_decrease=0.75)
    rms = np.sqrt(np.mean(proc**2))
    if rms > 1e-10:
        gain = 10 ** ((-16 - 20*math.log10(rms)) / 20)
        proc = np.clip(proc * gain, -1, 1).astype(np.float32)

    # 강한 전처리
    proc_strong = filtfilt(b, a, samples).astype(np.float32)
    proc_strong = nr.reduce_noise(y=proc_strong, sr=sr, stationary=True, prop_decrease=0.9)
    rms = np.sqrt(np.mean(proc_strong**2))
    if rms > 1e-10:
        gain = 10 ** ((-16 - 20*math.log10(rms)) / 20)
        proc_strong = np.clip(proc_strong * gain, -1, 1).astype(np.float32)

    # WAV 저장
    enh_wav = wav.replace(".wav", ".enh.wav")
    enh_strong_wav = wav.replace(".wav", ".enh_strong.wav")
    for path, data in [(enh_wav, proc), (enh_strong_wav, proc_strong)]:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())

    # 전처리된 오디오로 테스트
    for enh_path, label in [(enh_wav, "medium"), (enh_strong_wav, "strong")]:
        for pname in [best_name, "맥락 힌트 (정답 기반)"]:
            p = prompts[pname]
            short = pname[:15]
            segs, _ = model.transcribe(enh_path, language="ko", beam_size=5,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
                initial_prompt=p, condition_on_previous_text=True, temperature=0.0,
                no_speech_threshold=0.6, repetition_penalty=1.2, hallucination_silence_threshold=2.0)
            text = " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())
            c = cer(text, ref)
            print(f"  {label:6s} + {short:15s} beam5 → CER {c*100:5.1f}%")

            segs, _ = model.transcribe(enh_path, language="ko", beam_size=10,
                vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
                initial_prompt=p, condition_on_previous_text=True, temperature=0.0,
                no_speech_threshold=0.6, repetition_penalty=1.2, hallucination_silence_threshold=2.0)
            text = " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())
            c = cer(text, ref)
            print(f"  {label:6s} + {short:15s} beam10 → CER {c*100:5.1f}%")

    # 정리
    import os
    os.remove(enh_wav)
    os.remove(enh_strong_wav)

except Exception as e:
    print(f"  전처리 실패: {e}")

print("\n완료!")
