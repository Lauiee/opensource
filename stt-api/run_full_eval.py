"""전체 21개 타입 WAV 전사 + 교정 + CER 평가 스크립트.

사용법:
    python run_full_eval.py

필요 환경:
    - faster-whisper 설치
    - CUDA GPU 권장 (CPU도 가능하지만 느림)
"""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 경로 설정 ──
DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
DICT_PATH = Path(__file__).parent / "data" / "medical_dict.json"
RESULTS_PATH = Path(__file__).parent / "data" / "full_eval_results.json"


# ── CER 계산 ──
def normalize(t):
    t = unicodedata.normalize("NFC", t).strip()
    # 영어(한국어) 포맷 정규화: "insulin(인슐린)" → "인슐린"
    # 정답지에 영어+한국어가 병기된 경우, 한국어만 추출
    t = re.sub(r'[A-Za-z\-]+\(([가-힣\s]+)\)', r'\1', t)
    # 남은 영어 의학 용어를 한국어 음역으로 표준화
    # 괄호 안의 한국어가 없는 순수 영어 → 그대로 유지
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r'[.,!?;:()\\[\]{}"\'"]+', "", t)
    return t.lower()


def levenshtein(a, b):
    n, m = len(a), len(b)
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m
        for j in range(1, m + 1):
            c[j] = p[j - 1] if a[i - 1] == b[j - 1] else 1 + min(p[j - 1], p[j], c[j - 1])
        p = c
    return p[m]


def compute_cer(ref, hyp):
    r = list(normalize(ref).replace(" ", ""))
    h = list(normalize(hyp).replace(" ", ""))
    if not r:
        return 0.0
    return min(levenshtein(r, h) / len(r), 1.0)


# ── JSON 텍스트 로드 ──
def load_json_text(path):
    raw = path.read_text(encoding="utf-8").strip()
    be = raw.rfind("]")
    if be >= 0:
        raw = raw[: be + 1]
    data = json.loads(raw)
    return " ".join(item.get("content", "") for item in data if item.get("content"))


# ── 사전 교정 ──
def load_correction_dict():
    with open(DICT_PATH, encoding="utf-8") as f:
        d = json.load(f)
    entries = sorted(
        [e for e in d["entries"] if e.get("enabled", True) and e.get("strategy") == "exact"],
        key=lambda e: (-e.get("priority", 50), -len(e["wrong"])),
    )
    return entries


def apply_corrections(text, entries, ctx=""):
    ctx = ctx or text
    for e in entries:
        if e["wrong"] in text:
            hints = e.get("context_hint", [])
            if hints and not any(h in ctx for h in hints):
                continue
            text = text.replace(e["wrong"], e["correct"])

    # 문맥 교정: 심장→신장
    if re.search(r"소변|사구체|콩팥|신우|여과율|단백뇨", ctx):
        text = re.sub(
            r"심장\s*(기능|안에|안쪽|길이가)",
            lambda m: m.group(0).replace("심장", "신장"),
            text,
        )
    # 진로→진료
    text = re.sub(r"진로\s*(의뢰서|를|을)", lambda m: m.group(0).replace("진로", "진료"), text)

    # 환각 제거
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월부터\.?\s*", "", text)
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Whisper 전사 ──
def transcribe_wav(wav_path, specialty_prompt=""):
    """Faster-Whisper로 WAV 전사."""
    from faster_whisper import WhisperModel

    # 모델 로드 (첫 호출 시)
    if not hasattr(transcribe_wav, "_model"):
        print("  Whisper 모델 로딩 중 (CPU int8)...")
        transcribe_wav._model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("  → CPU 로드 완료")

    model = transcribe_wav._model

    prompt = specialty_prompt or (
        "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
        "고관절, 무릎, 척추, 디스크, 골절, 연골, 인대, 관절염, "
        "백내장, 녹내장, 비문증, 안압, 시력, 안약, "
        "담즙, 총담관, 낭종, 담석, 담관암, "
        "호흡 곤란, 흉부, 엑스레이, 배뇨장애, 전립선, "
        "해열진통제, 대증 치료, 요추 염좌, 좌골 신경통, 처방전, "
        "수술, 검사, 치료, 진단, 처방, 약, 입원, 퇴원, 외래"
    )

    segments, info = model.transcribe(
        str(wav_path),
        language="ko",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        initial_prompt=prompt,
        condition_on_previous_text=True,
        temperature=0.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.2,
        hallucination_silence_threshold=2.0,
    )

    texts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            texts.append(text)

    return " ".join(texts)


# ── 진료과별 프롬프트 ──
def get_specialty_prompt(type_num):
    try:
        from app.services.specialty_prompts import get_specialty_prompt as _get
        return _get(type_num=type_num)
    except ImportError:
        return ""


# ── 메인 ──
def main():
    # --reuse 옵션: 이전 전사 결과 재사용 (교정만 다시 적용)
    reuse_mode = "--reuse" in sys.argv

    print("=" * 80)
    if reuse_mode:
        print("  [교정 재적용 모드] 이전 전사 결과 재사용")
    else:
        print("  전체 21개 타입 WAV 전사 + 교정 + CER 평가")
    print("=" * 80)

    entries = load_correction_dict()
    print(f"  사전 항목: {len(entries)}개\n")

    # 이전 결과 로드 (reuse 모드)
    prev_stt = {}
    if reuse_mode and RESULTS_PATH.exists():
        prev = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        for r in prev.get("results", []):
            if r.get("stt_text") and len(r["stt_text"]) > 10:
                prev_stt[r["type"]] = r["stt_text"]
        print(f"  이전 전사 결과 {len(prev_stt)}개 로드\n")

    results = []
    total_before = total_after = 0
    count = 0

    for t in range(1, 22):
        ans_path = DATA_DIR / f"answer{t}.txt"
        wav_path = DATA_DIR / f"type{t}" / f"type{t}.wav"

        if not ans_path.exists() or not wav_path.exists():
            print(f"  Type{t}: 파일 없음 (skip)")
            continue

        # 정답 로드
        try:
            gt = load_json_text(ans_path)
        except Exception:
            gt = ans_path.read_text(encoding="utf-8").strip()

        # WAV 전사 (또는 이전 결과 재사용)
        if reuse_mode and t in prev_stt:
            stt = prev_stt[t]
            elapsed = 0.0
            print(f"  Type{t}: 이전 결과 재사용...", end=" ", flush=True)
        else:
            print(f"  Type{t}: 전사 중...", end=" ", flush=True)
            t0 = time.time()
            prompt = get_specialty_prompt(t)
            try:
                stt = transcribe_wav(wav_path, prompt)
            except Exception as e:
                print(f"실패: {e}")
                continue
            elapsed = time.time() - t0

        # CER 계산
        cer_before = compute_cer(gt, stt)

        # 교정 적용
        corrected = apply_corrections(stt, entries, stt)
        cer_after = compute_cer(gt, corrected)

        improvement = ((cer_before - cer_after) / max(cer_before, 0.001)) * 100 if cer_before > 0 else 0
        matches = sum(
            1
            for e in entries
            if e["wrong"] in stt
            and (not e.get("context_hint") or any(h in stt for h in e["context_hint"]))
        )

        status = "Good" if cer_after < 0.15 else "Fair" if cer_after < 0.3 else "Poor"
        icon = "[OK]" if cer_after < 0.15 else "[!!]" if cer_after < 0.3 else "[XX]"

        print(
            f"CER {cer_before*100:.1f}%→{cer_after*100:.1f}% "
            f"({improvement:+.1f}%) 교정{matches}회 {icon} {elapsed:.1f}s"
        )

        results.append({
            "type": t,
            "cer_before": round(cer_before, 4),
            "cer_after": round(cer_after, 4),
            "improvement_pct": round(improvement, 1),
            "corrections": matches,
            "status": status,
            "elapsed_sec": round(elapsed, 1),
            "stt_text": stt,
            "corrected_text": corrected,
        })

        total_before += cer_before
        total_after += cer_after
        count += 1

    # 요약
    if count > 0:
        avg_b = total_before / count
        avg_a = total_after / count
        avg_imp = ((avg_b - avg_a) / max(avg_b, 0.001)) * 100

        print("\n" + "=" * 80)
        print(f"  평균 CER: {avg_b*100:.1f}% → {avg_a*100:.1f}% ({avg_imp:+.1f}% 개선)")
        print(f"  평가 타입: {count}개")
        print("=" * 80)

        # JSON 저장
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "types_evaluated": count,
            "avg_cer_before": round(avg_b, 4),
            "avg_cer_after": round(avg_a, 4),
            "improvement_pct": round(avg_imp, 1),
            "results": results,
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
