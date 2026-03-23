"""Fine-tuned Whisper 모델로 전체 평가."""
import json, re, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
MODEL_DIR = Path(__file__).parent / "models" / "whisper-medical-ko"
DICT_PATH = Path(__file__).parent / "data" / "medical_dict.json"
RESULTS_PATH = Path(__file__).parent / "data" / "finetuned_eval_results.json"

def norm(t):
    t = unicodedata.normalize("NFC", t).strip()
    t = re.sub(r'[A-Za-z\-]+\(([가-힣\s]+)\)', r'\1', t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r'[.,!?;:()\\[\]{}"\'"]+', "", t)
    return t.lower()

def lev(a, b):
    n, m = len(a), len(b)
    p = list(range(m+1))
    for i in range(1, n+1):
        c = [i]+[0]*m
        for j in range(1, m+1):
            c[j] = p[j-1] if a[i-1]==b[j-1] else 1+min(p[j-1],p[j],c[j-1])
        p = c
    return p[m]

def cer(ref, hyp):
    r = list(norm(ref).replace(' ',''))
    h = list(norm(hyp).replace(' ',''))
    if not r: return 0.0
    return min(lev(r, h) / len(r), 1.0)

def load_json_text(path):
    raw = path.read_text(encoding='utf-8').strip()
    be = raw.rfind(']')
    if be >= 0: raw = raw[:be+1]
    data = json.loads(raw)
    return ' '.join(item.get('content','') for item in data if item.get('content'))

def load_corrections():
    with open(DICT_PATH, encoding='utf-8') as f:
        d = json.load(f)
    return sorted(
        [e for e in d['entries'] if e.get('enabled', True) and e.get('strategy')=='exact'],
        key=lambda e: (-e.get('priority',50), -len(e['wrong']))
    )

def correct(text, entries, ctx=''):
    ctx = ctx or text
    for e in entries:
        if e['wrong'] in text:
            hints = e.get('context_hint', [])
            if hints and not any(h in ctx for h in hints): continue
            text = text.replace(e['wrong'], e['correct'])
    if re.search(r'소변|사구체|콩팥|신우|여과율|단백뇨', ctx):
        text = re.sub(r'심장\s*(기능|안에|안쪽|길이가)', lambda m: m.group(0).replace('심장','신장'), text)
    text = re.sub(r'진로\s*(의뢰서|를|을)', lambda m: m.group(0).replace('진로','진료'), text)
    text = re.sub(r'(?:1[3-9]|[2-9]\d)월부터\.?\s*', '', text)
    text = re.sub(r'(\d{1,2}월부터\.?\s*){5,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    import torch
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("=" * 80)
    print("  Fine-tuned Whisper 모델 평가")
    print("=" * 80)

    # Load fine-tuned model
    print(f"  모델 로딩: {MODEL_DIR}")
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  디바이스: {device}")

    entries = load_corrections()
    print(f"  사전: {len(entries)}개\n")

    print(f'  Type    CER(원본)   CER(교정후)  개선율    상태')
    print('-' * 80)

    results = []
    tb = ta = 0; cnt = 0

    for t in range(1, 22):
        ans_p = DATA_DIR / f"answer{t}.txt"
        wav_p = DATA_DIR / f"type{t}" / f"type{t}.wav"
        if not ans_p.exists() or not wav_p.exists(): continue

        try: gt = load_json_text(ans_p)
        except: gt = ans_p.read_text(encoding='utf-8').strip()

        # Transcribe with fine-tuned model
        print(f"  Type{t}: 전사 중...", end=" ", flush=True)
        t0 = time.time()

        waveform, sr = torchaudio.load(str(wav_p))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze().numpy()

        # Process in 30-sec chunks
        chunk_size = 30 * 16000
        all_text = []
        for start in range(0, len(waveform), chunk_size):
            chunk = waveform[start:start+chunk_size]
            if len(chunk) < 1600: continue  # Too short

            input_features = processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language="ko", task="transcribe",
                    max_new_tokens=440,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=4,
                )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if text:
                all_text.append(text)

        stt = " ".join(all_text)
        elapsed = time.time() - t0

        cb = cer(gt, stt)
        corrected = correct(stt, entries, stt)
        ca = cer(gt, corrected)
        imp = ((cb-ca)/max(cb,0.001))*100 if cb > 0 else 0
        icon = "[OK]" if ca<0.15 else "[!!]" if ca<0.3 else "[XX]"

        print(f"CER {cb*100:.1f}%->{ca*100:.1f}% ({imp:+.1f}%) {icon} {elapsed:.1f}s")

        results.append({"type": t, "cer_before": round(cb,4), "cer_after": round(ca,4),
                        "stt_text": stt, "corrected_text": corrected})
        tb += cb; ta += ca; cnt += 1

    if cnt > 0:
        ab = tb/cnt; aa = ta/cnt; ai = ((ab-aa)/max(ab,0.001))*100
        print('-' * 80)
        print(f'  평균    {ab*100:.1f}%     {aa*100:.1f}%   {ai:+.1f}%')
        print(f'  평가: {cnt}개 타입')

        summary = {"avg_before": round(ab,4), "avg_after": round(aa,4), "results": results}
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == "__main__":
    main()
