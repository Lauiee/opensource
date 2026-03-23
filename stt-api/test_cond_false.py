"""condition_on_previous_text=False 효과 테스트 (worst CER 타입들)."""
import sys, os, json, re, unicodedata, time
from pathlib import Path
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

def norm(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[A-Za-z]+\(([가-힣]+)\)", r"\1", t)
    t = re.sub(r"\([A-Za-z\s]+\)", "", t)
    t = re.sub(r"[.,!?;:()\[\]{}\"\\'`~@#$%^&*+=<>/\\\\|_\-]", "", t)
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

def transcribe(wav, prompt, beam=5, cond=True):
    segs, _ = model.transcribe(str(wav), language="ko", beam_size=beam,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        initial_prompt=prompt, condition_on_previous_text=cond, temperature=0.0,
        no_speech_threshold=0.6, repetition_penalty=1.2, hallucination_silence_threshold=2.0)
    return " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())

# 진료과 프롬프트
from app.services.specialty_prompts import get_specialty_prompt

# 모든 21개 타입 테스트
print(f"\n{'Type':>5} {'기존(b5,T)':>11} {'b10,T':>8} {'b5,F':>8} {'b10,F':>8} {'최적':>8} {'변화':>8} {'Best':>12}")
print("-" * 80)

total_old = 0
total_best = 0
count = 0

for n in range(1, 22):
    if n == 7: continue  # 37분 skip
    wav = DATA / f"type{n}" / f"type{n}.wav"
    ans = DATA / f"answer{n}.txt"
    if not wav.exists() or not ans.exists(): continue

    with open(ans, "r", encoding="utf-8") as f:
        ref = " ".join(i["content"] for i in json.load(f))

    prompt = get_specialty_prompt(type_num=n)

    # 4가지 조합 테스트
    configs = [
        ("b5,T", 5, True),
        ("b10,T", 10, True),
        ("b5,F", 5, False),
        ("b10,F", 10, False),
    ]

    cers = {}
    for name, beam, cond in configs:
        text = transcribe(wav, prompt, beam=beam, cond=cond)
        c = cer(text, ref)
        cers[name] = c

    best_name = min(cers, key=cers.get)
    best_c = cers[best_name]
    old_c = cers["b5,T"]
    delta = old_c - best_c

    total_old += old_c
    total_best += best_c
    count += 1

    marker = " ★" if delta > 0.02 else ""
    print(f"  {n:>3}  {cers['b5,T']*100:>9.1f}%  {cers['b10,T']*100:>6.1f}%  "
          f"{cers['b5,F']*100:>6.1f}%  {cers['b10,F']*100:>6.1f}%  "
          f"{best_c*100:>6.1f}%  {delta*100:>+6.1f}%  {best_name:>10}{marker}")

avg_old = total_old / count * 100 if count else 0
avg_best = total_best / count * 100 if count else 0

print("-" * 80)
print(f"  평균: 기존 {avg_old:.1f}% → 최적 {avg_best:.1f}% ({avg_old-avg_best:+.1f}%p)")
print(f"  (GPU 기준, Type 7 제외 {count}개 타입)")
