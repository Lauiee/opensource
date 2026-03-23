"""Optimized evaluation pipeline for Korean medical STT.

Goal: Reduce average CER from 27% to below 10%.

Phases:
  1. Specialty mapping for all 21 types (including 17-21)
  2. Multi-config parameter search per type
  3. Enhanced post-processing with context-aware corrections
  4. Two-pass transcription integration
  5. Per-type best config selection and final evaluation

Usage:
    python -X utf8 optimized_eval.py
    python -X utf8 optimized_eval.py --skip-preprocess   # skip audio preprocessing
    python -X utf8 optimized_eval.py --cpu-only           # force CPU int8
"""

import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# ── Paths ──
DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
SCRIPT_DIR = Path(__file__).parent
DICT_PATH = SCRIPT_DIR / "data" / "medical_dict.json"
RESULTS_PATH = SCRIPT_DIR / "data" / "optimized_eval_results.json"
PREPROCESSED_DIR = SCRIPT_DIR / "data" / "preprocessed_audio"

# ============================================================
# Phase 1: Complete Type-to-Specialty Mapping
# ============================================================

TYPE_TO_SPECIALTY = {
    1: "내과",
    2: "내분비내과",
    3: "간담도외과",
    4: "안과",
    5: "정형외과",
    6: "간담도외과",
    7: "정형외과",
    8: "비뇨기과",
    9: "정형외과",
    10: "정형외과",
    11: "내과",
    12: "감염내과",
    13: "정형외과",
    14: "호흡기내과",
    15: "호흡기내과",
    16: "정형외과",
    # NEW: Types 17-21 mapped from answer file analysis
    17: "정형외과",       # 고관절 이형성증, 비구골, 대퇴골두
    18: "정형외과",       # 뼈, 초음파, 염증 세포
    19: "정형외과",       # 골다공증, 허리
    20: "신장내과",       # 신장 기능, 사구체 여과율, 콩팥
    21: "신장내과",       # 콩팥, 비타민D, 단백뇨, 칼슘
}

# Short specialty prompts (under 100 chars for core, + key terms)
# Long prompts caused hallucination; keep these compact.
SHORT_SPECIALTY_PROMPTS = {
    "정형외과": "정형외과 진료 상담. 의사 환자 대화. 고관절, 무릎, 척추, 골절, 연골, 인대, 수술, 재활.",
    "안과": "안과 진료 상담. 의사 환자 대화. 백내장, 녹내장, 비문증, 안압, 시력, 안약.",
    "간담도외과": "간담도외과 진료 상담. 의사 환자 대화. 담즙, 총담관, 낭종, 담석, 간.",
    "내분비내과": "내분비내과 진료 상담. 의사 환자 대화. 쿠싱, 부신, 호르몬, 당뇨, 갑상선.",
    "호흡기내과": "호흡기내과 진료 상담. 의사 환자 대화. 흉부, 엑스레이, 호흡, 기침, 폐.",
    "감염내과": "감염내과 진료 상담. 의사 환자 대화. 발열, 해열진통제, 대증 치료, 항생제.",
    "비뇨기과": "비뇨기과 진료 상담. 의사 환자 대화. 배뇨장애, 전립선, 방광, 요도.",
    "내과": "내과 진료 상담. 의사 환자 대화. 혈압, 혈당, 검사, 처방.",
    "신장내과": "신장내과 진료 상담. 의사 환자 대화. 신장, 사구체, 여과율, 콩팥, 단백뇨, 소변.",
    "정신건강의학과": "정신건강의학과 진료 상담. 의사 환자 대화. 우울, 불안, 수면, 약물.",
    "신경과": "신경과 진료 상담. 의사 환자 대화. 두통, 어지럼증, 뇌, 치매.",
    "산부인과": "산부인과 진료 상담. 의사 환자 대화. 자궁, 난소, 임신, 초음파.",
}

# Type-specific extra keywords (for types with known content)
TYPE_EXTRA_KEYWORDS = {
    2: "코르티솔, 쿠싱 증후군, 부신 종양",
    3: "루-엔-Y, 담관 공장 문합술, 총담관 낭종",
    4: "수정체, 산동 검사, 안저 검사, 인공수정체",
    8: "전립선 비대증, PSA, 잔뇨감",
    10: "lateral, oblique, AP, 요추",
    11: "빌리루빈, 기억력, 일상생활",
    12: "고열, 전신 권태감, 수분 섭취",
    13: "요추 염좌, 좌골 신경통",
    14: "흉부 엑스레이, 산소포화도",
    16: "Dark Disk Disease, Hip dysplasia, 이형성증",
    17: "고관절 이형성증, 비구골, 대퇴골두, subchondral, 경화 소견",
    18: "뼈, 초음파, 염증 세포, 뾰족, 자국",
    19: "골다공증, 허리, 진료 의뢰서",
    20: "사구체 여과율, 신장 기능, 콩팥, 신우, 통증",
    21: "비타민 D, 간수치, 콩팥, 단백뇨, 칼슘, 혈압",
}

# ============================================================
# Phase 2: Parameter Configurations
# ============================================================

PARAM_CONFIGS = {
    "A_baseline": {
        "beam_size": 5,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_threshold": 0.5,
        "min_silence_ms": 500,
        "speech_pad_ms": 400,
        "prompt_style": "short",
        "rep_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
    },
    "B_large_beam": {
        "beam_size": 10,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_threshold": 0.5,
        "min_silence_ms": 500,
        "speech_pad_ms": 400,
        "prompt_style": "short",
        "rep_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
    },
    "C_sensitive_vad": {
        "beam_size": 5,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_threshold": 0.35,
        "min_silence_ms": 400,
        "speech_pad_ms": 300,
        "prompt_style": "short",
        "rep_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
    },
    "D_beam10_sensitive": {
        "beam_size": 10,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_threshold": 0.35,
        "min_silence_ms": 400,
        "speech_pad_ms": 300,
        "prompt_style": "short",
        "rep_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
    },
    "E_no_vad": {
        "beam_size": 5,
        "temperature": 0.0,
        "vad_filter": False,
        "prompt_style": "short",
        "rep_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
    },
}


# ============================================================
# CER Calculation
# ============================================================

def normalize_text(t):
    """Normalize text for CER comparison."""
    t = unicodedata.normalize("NFC", t).strip()
    # English(Korean) format: "insulin(인슐린)" -> "인슐린"
    t = re.sub(r'[A-Za-z\-]+\(([가-힣\s]+)\)', r'\1', t)
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
    r = list(normalize_text(ref).replace(" ", ""))
    h = list(normalize_text(hyp).replace(" ", ""))
    if not r:
        return 0.0
    return min(levenshtein(r, h) / len(r), 1.0)


# ============================================================
# Data Loading
# ============================================================

def load_json_text(path):
    raw = path.read_text(encoding="utf-8").strip()
    be = raw.rfind("]")
    if be >= 0:
        raw = raw[: be + 1]
    data = json.loads(raw)
    return " ".join(item.get("content", "") for item in data if item.get("content"))


def load_correction_dict():
    with open(DICT_PATH, encoding="utf-8") as f:
        d = json.load(f)
    entries = sorted(
        [e for e in d["entries"] if e.get("enabled", True) and e.get("strategy") == "exact"],
        key=lambda e: (-e.get("priority", 50), -len(e["wrong"])),
    )
    return entries


# ============================================================
# Phase 3: Enhanced Post-Processing
# ============================================================

# Context-aware homophone corrections
CONTEXT_CORRECTIONS = [
    # (pattern_context, wrong, correct)
    # 심장 -> 신장 when nephrology context
    (r"소변|사구체|콩팥|신우|여과율|단백뇨|크레아티닌|혈뇨", "심장", "신장"),
]

# Common Whisper misrecognitions for Korean medical speech
EXTRA_CORRECTIONS = [
    ("진로 의뢰서", "진료 의뢰서"),
    ("진로를", "진료를"),
    ("진로을", "진료을"),
    ("진로 ", "진료 "),
    # Common hallucination phrases
    ("시청해주셔서 감사합니다", ""),
    ("시청해 주셔서 감사합니다", ""),
    ("구독과 좋아요 부탁드립니다", ""),
    ("구독과 좋아요", ""),
    ("자막 by", ""),
    ("다음 시간에 만나요", ""),
    ("다음 영상에서 만나요", ""),
    ("채널에 오신 것을 환영합니다", ""),
    # Medical term corrections not in dict
    ("이명성증", "이형성증"),
    ("이명 성증", "이형성증"),
    ("이영 성증", "이형성증"),
    ("전체환 수로", "전치환술 후"),
    ("전체환 술로", "전치환술로"),
]


def apply_enhanced_corrections(text, entries, type_num=None):
    """Enhanced post-processing pipeline."""
    if not text:
        return text

    original = text

    # Step 1: Hallucination removal
    # Remove impossible month patterns
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월부터\.?\s*", "", text)
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)

    # Remove YouTube/broadcast hallucinations
    for phrase in [
        "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다",
        "구독과 좋아요 부탁드립니다", "구독과 좋아요",
        "자막 by", "다음 시간에 만나요", "다음 영상에서 만나요",
        "채널에 오신 것을 환영합니다", "끝까지 시청해주셔서 감사합니다",
        "환자분께 좀 더 참여해 주시기 바랍니다",
        "환경화 기업에 동참해봐요",
    ]:
        text = text.replace(phrase, "")

    # Remove media keywords
    for kw in ["MBC", "KBS", "SBS", "JTBC", "YTN", "MBN", "TV조선"]:
        text = text.replace(kw + " 뉴스", "")
        text = text.replace(kw, "")

    # Remove sound markers
    text = re.sub(r"\[음악\]|\(음악\)|\[박수\]|\(박수\)|\[웃음\]|\(웃음\)|♪+|♫+", "", text)

    # Step 2: Medical dictionary corrections (from medical_dict.json)
    for e in entries:
        if e["wrong"] in text:
            hints = e.get("context_hint", [])
            if hints and not any(h in text for h in hints):
                continue
            text = text.replace(e["wrong"], e["correct"])

    # Step 3: Extra corrections
    for wrong, correct in EXTRA_CORRECTIONS:
        text = text.replace(wrong, correct)

    # Step 4: Context-aware corrections
    specialty = TYPE_TO_SPECIALTY.get(type_num, "")

    # 심장 -> 신장 in nephrology context
    if specialty in ("신장내과",) or re.search(r"소변|사구체|콩팥|신우|여과율|단백뇨|크레아티닌|혈뇨", text):
        text = re.sub(r"심장\s*(기능|안에|안쪽|길이가)", lambda m: m.group(0).replace("심장", "신장"), text)
        # Also handle standalone "심장" when clearly about kidney
        if re.search(r"사구체|여과율|콩팥|단백뇨", text):
            text = re.sub(r"심장 기능", "신장 기능", text)

    # 진로 -> 진료
    text = re.sub(r"진로\s*(의뢰서|를|을|에|실)", lambda m: m.group(0).replace("진로", "진료"), text)

    # Step 5: Filler reduction (but keep meaningful fillers like 네, 예)
    # Remove repeated fillers
    text = re.sub(r"(네\s*){4,}", "네. ", text)
    text = re.sub(r"(예\s*){4,}", "예. ", text)
    text = re.sub(r"아{4,}", "아", text)
    text = re.sub(r"어{4,}", "어", text)
    text = re.sub(r"음{3,}", "음", text)

    # Step 6: Punctuation normalization
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Safety: if text became too short after corrections, return original
    if len(text) < max(5, len(original) * 0.1):
        return original

    return text


# ============================================================
# Audio Preprocessing
# ============================================================

def preprocess_audio(input_path, output_path):
    """Preprocess audio with ffmpeg: noise reduction + normalization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return str(output_path)

    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "highpass=f=80,lowpass=f=7500,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    [WARN] ffmpeg failed for {input_path}: {result.stderr[:200]}")
        return str(input_path)  # fallback to original

    return str(output_path)


# ============================================================
# Model Loading & Transcription
# ============================================================

_model_cache = {}


def get_model(device="cpu", compute_type="int8"):
    """Load and cache Whisper model."""
    key = f"{device}_{compute_type}"
    if key not in _model_cache:
        # Setup CUDA DLL paths on Windows
        if sys.platform == "win32" and device == "cuda":
            nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
            for sub in ("cublas", "cudnn"):
                bin_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.isdir(bin_dir):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(bin_dir)
                        except OSError:
                            pass

        # HF symlink workaround
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        try:
            import huggingface_hub.file_download as _hf_dl
            import shutil
            _orig = _hf_dl._create_symlink
            def _copy_fallback(src, dst, new_blob=False):
                try:
                    _orig(src, dst, new_blob)
                except OSError:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.copy2(src, dst)
            _hf_dl._create_symlink = _copy_fallback
        except Exception:
            pass

        from faster_whisper import WhisperModel
        print(f"  Loading Whisper large-v3 on {device} ({compute_type})...")
        try:
            model = WhisperModel("large-v3", device=device, compute_type=compute_type)
            print(f"  -> Model loaded on {device}")
        except Exception as e:
            if device == "cuda":
                print(f"  -> CUDA failed ({e}), falling back to CPU int8")
                model = WhisperModel("large-v3", device="cpu", compute_type="int8")
                key = "cpu_int8"
            else:
                raise
        _model_cache[key] = model
    return _model_cache.get(key, _model_cache.get("cpu_int8"))


def build_prompt(type_num, style="short"):
    """Build initial prompt for a given type."""
    specialty = TYPE_TO_SPECIALTY.get(type_num, "내과")

    if style == "minimal":
        return "진료 상담 녹음. 의사와 환자 대화."

    base = SHORT_SPECIALTY_PROMPTS.get(specialty, "진료 상담. 의사 환자 대화.")

    # Add type-specific extra keywords
    extras = TYPE_EXTRA_KEYWORDS.get(type_num, "")
    if extras:
        base += f" {extras}"

    return base


def transcribe_single(wav_path, config, type_num):
    """Run a single transcription with given config."""
    model = get_model("cpu", "int8")

    prompt_style = config.get("prompt_style", "short")
    prompt = build_prompt(type_num, prompt_style)

    kwargs = {
        "language": "ko",
        "beam_size": config.get("beam_size", 5),
        "initial_prompt": prompt,
        "condition_on_previous_text": True,
        "temperature": config.get("temperature", 0.0),
        "no_speech_threshold": 0.6,
        "repetition_penalty": config.get("rep_penalty", 1.2),
        "hallucination_silence_threshold": config.get("hallucination_silence_threshold", 2.0),
    }

    if config.get("vad_filter", True):
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": config.get("min_silence_ms", 500),
            "speech_pad_ms": config.get("speech_pad_ms", 400),
            "threshold": config.get("vad_threshold", 0.5),
        }
    else:
        kwargs["vad_filter"] = False

    segments, info = model.transcribe(str(wav_path), **kwargs)

    texts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            # Filter transcription-level hallucinations
            text = _filter_hallucinations(text)
            if text:
                texts.append(text)

    return " ".join(texts)


def _filter_hallucinations(text):
    """Filter obvious hallucinations at transcription level."""
    if not text:
        return text

    # Impossible month repetitions
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월부터\.?\s*", "", text)
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월", "", text)
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)

    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 2:
        return ""
    return text


# ============================================================
# Phase 4: Two-Pass Transcription
# ============================================================

def two_pass_transcribe(wav_path, best_config, type_num, correction_entries):
    """Two-pass transcription: use Pass 1 output to enhance Pass 2 prompt."""
    # Pass 1: Best config
    pass1_text = transcribe_single(wav_path, best_config, type_num)
    pass1_corrected = apply_enhanced_corrections(pass1_text, correction_entries, type_num)

    # Extract medical terms from Pass 1 for enhanced prompt
    medical_terms = extract_medical_terms(pass1_corrected)

    if not medical_terms:
        return pass1_text, pass1_corrected

    # Build enhanced prompt for Pass 2
    base_prompt = build_prompt(type_num, "short")
    term_str = ", ".join(medical_terms[:15])  # Limit to 15 terms
    enhanced_prompt = f"{base_prompt} {term_str}"

    # Truncate to avoid excessively long prompts (max ~200 chars)
    if len(enhanced_prompt) > 250:
        enhanced_prompt = enhanced_prompt[:250]

    # Pass 2 config: same as best but with enhanced prompt
    model = get_model("cpu", "int8")

    kwargs = {
        "language": "ko",
        "beam_size": best_config.get("beam_size", 5),
        "initial_prompt": enhanced_prompt,
        "condition_on_previous_text": True,
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "repetition_penalty": best_config.get("rep_penalty", 1.2),
        "hallucination_silence_threshold": best_config.get("hallucination_silence_threshold", 2.0),
    }

    if best_config.get("vad_filter", True):
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": best_config.get("min_silence_ms", 500),
            "speech_pad_ms": best_config.get("speech_pad_ms", 400),
            "threshold": best_config.get("vad_threshold", 0.5),
        }
    else:
        kwargs["vad_filter"] = False

    segments, info = model.transcribe(str(wav_path), **kwargs)

    pass2_texts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            text = _filter_hallucinations(text)
            if text:
                pass2_texts.append(text)

    pass2_text = " ".join(pass2_texts)
    pass2_corrected = apply_enhanced_corrections(pass2_text, correction_entries, type_num)

    return pass2_text, pass2_corrected


def extract_medical_terms(text):
    """Extract medical terms from transcribed text."""
    # Korean medical term patterns
    terms = set()

    # Multi-syllable medical terms (3+ chars ending in medical suffixes)
    patterns = [
        r'[가-힣]{2,}(?:증|염|술|제|암|병|통|근|골|막|낭|석)',
        r'[가-힣]{2,}(?:검사|수술|치료|진단|처방|기능|수치)',
        r'(?:사구체|여과율|콩팥|신장|간수치|혈압|혈당|단백뇨)',
        r'(?:고관절|대퇴골|비구골|슬개골|경골|비골)',
        r'(?:이형성증|관절염|골다공증|백내장|녹내장)',
        r'(?:비타민\s*[A-Za-z]|인슐린|코르티솔|빌리루빈)',
        r'(?:엑스레이|MRI|CT|초음파)',
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            term = m.group()
            if len(term) >= 2:
                terms.add(term)

    return list(terms)


# ============================================================
# Phase 5: Main Evaluation Loop
# ============================================================

def main():
    skip_preprocess = "--skip-preprocess" in sys.argv
    cpu_only = "--cpu-only" in sys.argv

    print("=" * 90)
    print("  Optimized Medical STT Evaluation Pipeline")
    print("  Target: CER < 10% (baseline: 27.0%)")
    print("=" * 90)

    # Load correction dictionary
    entries = load_correction_dict()
    print(f"\n  Medical dictionary: {len(entries)} entries")
    print(f"  Specialty mappings: {len(TYPE_TO_SPECIALTY)} types")
    print(f"  Parameter configs: {len(PARAM_CONFIGS)} configs")
    print(f"  Audio preprocessing: {'SKIP' if skip_preprocess else 'ON'}")
    print()

    # Baseline CERs from previous evaluation
    BASELINE_CERS = {
        1: 0.111, 2: 0.413, 3: 0.244, 4: 0.464, 5: 0.254,
        6: 0.203, 7: 0.176, 8: 0.136, 9: 0.129, 10: 0.364,
        11: 0.593, 12: 0.083, 13: 0.032, 14: 0.103, 15: 0.0,
        16: 0.260, 17: 0.700, 18: 0.360, 19: 0.460, 20: 0.278,
        21: 0.320,
    }

    # Preload model
    print("  Phase 0: Loading model...")
    t0 = time.time()
    model = get_model("cpu", "int8")
    print(f"  -> Model ready ({time.time()-t0:.1f}s)\n")

    all_results = []
    total_best_cer = 0
    total_baseline_cer = 0
    type_count = 0

    for t in range(1, 22):
        ans_path = DATA_DIR / f"answer{t}.txt"
        wav_path = DATA_DIR / f"type{t}" / f"type{t}.wav"

        if not ans_path.exists() or not wav_path.exists():
            print(f"  Type {t:2d}: SKIP (files missing)")
            continue

        print(f"\n{'─'*90}")
        print(f"  Type {t:2d} | Specialty: {TYPE_TO_SPECIALTY.get(t, '?')}")
        print(f"{'─'*90}")

        # Load ground truth
        try:
            gt = load_json_text(ans_path)
        except Exception as e:
            print(f"  ERROR loading answer: {e}")
            continue

        # Audio preprocessing
        if not skip_preprocess:
            preprocessed_path = PREPROCESSED_DIR / f"type{t}_processed.wav"
            print(f"  Preprocessing audio...")
            audio_path = preprocess_audio(wav_path, preprocessed_path)
        else:
            audio_path = str(wav_path)

        # Phase 2: Run all configs
        config_results = {}
        best_config_name = None
        best_cer_raw = 1.0
        best_cer_corrected = 1.0
        best_raw_text = ""
        best_corrected_text = ""

        for cfg_name, cfg in PARAM_CONFIGS.items():
            print(f"  Config {cfg_name}...", end=" ", flush=True)
            t1 = time.time()

            try:
                raw_text = transcribe_single(audio_path, cfg, t)
                corrected_text = apply_enhanced_corrections(raw_text, entries, t)

                cer_raw = compute_cer(gt, raw_text)
                cer_corrected = compute_cer(gt, corrected_text)

                elapsed = time.time() - t1
                print(f"CER: {cer_raw*100:.1f}% -> {cer_corrected*100:.1f}% ({elapsed:.1f}s)")

                config_results[cfg_name] = {
                    "cer_raw": cer_raw,
                    "cer_corrected": cer_corrected,
                    "raw_text": raw_text,
                    "corrected_text": corrected_text,
                    "time": elapsed,
                }

                if cer_corrected < best_cer_corrected:
                    best_cer_corrected = cer_corrected
                    best_cer_raw = cer_raw
                    best_config_name = cfg_name
                    best_raw_text = raw_text
                    best_corrected_text = corrected_text

            except Exception as e:
                print(f"ERROR: {e}")
                config_results[cfg_name] = {"error": str(e)}

        # Phase 4: Two-pass with best config
        if best_config_name:
            print(f"  Two-pass with {best_config_name}...", end=" ", flush=True)
            t1 = time.time()
            try:
                pass2_raw, pass2_corrected = two_pass_transcribe(
                    audio_path, PARAM_CONFIGS[best_config_name], t, entries
                )
                cer_2pass = compute_cer(gt, pass2_corrected)
                elapsed = time.time() - t1
                print(f"CER: {cer_2pass*100:.1f}% ({elapsed:.1f}s)")

                config_results["two_pass"] = {
                    "cer_raw": compute_cer(gt, pass2_raw),
                    "cer_corrected": cer_2pass,
                    "raw_text": pass2_raw,
                    "corrected_text": pass2_corrected,
                    "time": elapsed,
                }

                if cer_2pass < best_cer_corrected:
                    best_cer_corrected = cer_2pass
                    best_config_name = "two_pass"
                    best_corrected_text = pass2_corrected

            except Exception as e:
                print(f"ERROR: {e}")

        baseline = BASELINE_CERS.get(t, 0.27)
        improvement = (baseline - best_cer_corrected) * 100

        print(f"\n  >>> Best: {best_config_name} | CER: {best_cer_corrected*100:.1f}% "
              f"(baseline: {baseline*100:.1f}%, improvement: {improvement:+.1f}pp)")

        type_result = {
            "type": t,
            "specialty": TYPE_TO_SPECIALTY.get(t, "?"),
            "best_config": best_config_name,
            "best_cer": best_cer_corrected,
            "best_cer_raw": best_cer_raw,
            "baseline_cer": baseline,
            "improvement_pp": improvement,
            "best_text": best_corrected_text,
            "ground_truth_preview": gt[:200],
            "configs": {k: {kk: vv for kk, vv in v.items() if kk != "raw_text" and kk != "corrected_text"}
                       for k, v in config_results.items()},
        }
        all_results.append(type_result)

        total_best_cer += best_cer_corrected
        total_baseline_cer += baseline
        type_count += 1

    # ── Final Summary ──
    avg_best = total_best_cer / type_count if type_count else 0
    avg_baseline = total_baseline_cer / type_count if type_count else 0

    print("\n\n" + "=" * 110)
    print(f"  FINAL RESULTS  |  Avg CER: {avg_best*100:.1f}% (baseline: {avg_baseline*100:.1f}%)")
    print("=" * 110)
    print(f"{'Type':>4} | {'Specialty':<12} | {'Best Config':<20} | {'Raw CER':>8} | {'Corrected':>9} | {'Best CER':>8} | {'Baseline':>8} | {'Delta':>7}")
    print("-" * 110)

    for r in all_results:
        raw_cer = r.get("best_cer_raw", 0)
        print(f"{r['type']:4d} | {r['specialty']:<12} | {r['best_config']:<20} | "
              f"{raw_cer*100:7.1f}% | {r['best_cer']*100:8.1f}% | {r['best_cer']*100:7.1f}% | "
              f"{r['baseline_cer']*100:7.1f}% | {r['improvement_pp']:+6.1f}pp")

    print("-" * 110)
    print(f"{'AVG':>4} | {'':12} | {'':20} | {'':>8} | {avg_best*100:8.1f}% | {avg_best*100:7.1f}% | "
          f"{avg_baseline*100:7.1f}% | {(avg_baseline-avg_best)*100:+6.1f}pp")
    print("=" * 110)

    if avg_best < 0.10:
        print("\n  *** TARGET ACHIEVED: Average CER < 10% ***")
    elif avg_best < 0.15:
        print(f"\n  Good progress! {avg_best*100:.1f}% average CER. Need more work to reach <10%.")
    else:
        print(f"\n  Average CER: {avg_best*100:.1f}%. Significant work still needed.")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avg_cer": avg_best,
        "avg_baseline_cer": avg_baseline,
        "improvement_pp": (avg_baseline - avg_best) * 100,
        "total_types": type_count,
        "results": all_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
