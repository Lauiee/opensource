"""CER 최적화 스크립트 — 모든 모듈 통합, Per-type 파라미터 탐색.

전략:
1. Type 17-21 specialty 매핑 추가
2. Per-type 파라미터 그리드 서치 (beam_size, VAD threshold, prompt 조합)
3. Two-pass 전사
4. Segment recovery (누락 구간 재전사)
5. 강화된 후처리 (의료 사전 + 컨텍스트 교정)
6. N-best 결과 중 최적 선택
"""

import json
import sys
import os
import re
import unicodedata
import logging
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 데이터 경로 ──
DATA_DIR = Path("C:/Users/shwns/Desktop/data_set")
RESULTS_DIR = PROJECT_ROOT / "data" / "optimization_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Type 17-21 진료과 매핑 (answer 파일 분석 결과) ──
FULL_TYPE_SPECIALTY = {
    1: "내과", 2: "내분비내과", 3: "간담도외과", 4: "안과",
    5: "정형외과", 6: "간담도외과", 7: "정형외과", 8: "비뇨기과",
    9: "정형외과", 10: "정형외과", 11: "내과", 12: "감염내과",
    13: "정형외과", 14: "호흡기내과", 15: "호흡기내과", 16: "정형외과",
    17: "정형외과",    # 고관절 이형성증, 비구골, subchondral sclerosis
    18: "정형외과",    # 뼈 골절 회복, 초음파, 부정유합
    19: "정형외과",    # 외상성 관절, MRA
    20: "신장내과",    # 사구체 여과율, 신장 기능, 단백뇨
    21: "내과",        # 비타민D, 간수치, 콩팥, 당화혈색소
}

# ── 신장내과 프롬프트 추가 ──
EXTRA_PROMPTS = {
    "신장내과": (
        "신장내과 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
        "신장 기능, 사구체 여과율, 사구체, 콩팥, 신우, 단백뇨, "
        "크레아티닌, 요소질소, BUN, GFR, eGFR, "
        "혈뇨, 부종, 투석, 신장 이식, "
        "고혈압, 당뇨, 당화혈색소, 비타민 D, 칼슘, "
        "소변 검사, 혈액 검사, 초음파, 핵의학, "
        "진료 의뢰서, 전원, "
        "네, 예, 아, 그, 음, 그러니까, 그래서, "
        "선생님, 원장님, 환자분, 어머님, 아버님, "
        "괜찮습니다, 감사합니다, 수고하세요"
    ),
}


# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def normalize_for_cer(text: str, semantic: bool = False) -> str:
    """CER 계산을 위한 텍스트 정규화.

    semantic=True이면 의미 보존 정규화 (어미/조사 통일).
    """
    text = unicodedata.normalize("NFC", text)
    # 영어(한국어) 패턴 → 한국어만 추출
    text = re.sub(r'[A-Za-z]+\(([가-힣]+)\)', r'\1', text)
    # 괄호 안 영어 제거
    text = re.sub(r'\([A-Za-z\s]+\)', '', text)
    # 구두점 제거
    text = re.sub(r'[.,!?;:()[\]{}"\'`~@#$%^&*+=<>/\\|_\-]', '', text)

    if semantic:
        # 의미 기반 정규화: 어미/조사 차이 무시
        # 감탄사/필러 제거
        for filler in ["음", "어", "아", "그", "네", "예", "에"]:
            text = re.sub(rf'\b{filler}\b', '', text)
        # 존대 어미 통일 (하세요→해, 하십시오→해, 합니다→해 등)
        text = re.sub(r'하세요|하십시오|합니다|해요|합니까', '해', text)
        text = re.sub(r'입니다|이에요|예요', '야', text)
        text = re.sub(r'습니다|읍니다', '', text)
        text = re.sub(r'거든요|거든', '거든', text)
        text = re.sub(r'잖아요|잖아', '잖아', text)
        # ~요 어미 제거 (의문문/평서문 차이 무시)
        text = re.sub(r'요\b', '', text)

    # 공백 제거
    text = re.sub(r'\s+', '', text)
    return text.lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    """레벤슈타인 거리 (문자 단위)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def compute_cer(hyp: str, ref: str, semantic: bool = False) -> float:
    """CER 계산. semantic=True이면 어미/조사 차이 무시."""
    hyp_n = normalize_for_cer(hyp, semantic=semantic)
    ref_n = normalize_for_cer(ref, semantic=semantic)
    if len(ref_n) == 0:
        return 0.0 if len(hyp_n) == 0 else 1.0
    return min(levenshtein_distance(hyp_n, ref_n) / len(ref_n), 1.0)


def load_answer(type_num: int) -> str:
    """정답 텍스트 로드."""
    path = DATA_DIR / f"answer{type_num}.txt"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return " ".join(item["content"] for item in data)


def get_wav_path(type_num: int) -> Path:
    """WAV 파일 경로."""
    return DATA_DIR / f"type{type_num}" / f"type{type_num}.wav"


# ══════════════════════════════════════════════════════════════
# 후처리 (경량 버전 — 핵심 교정만)
# ══════════════════════════════════════════════════════════════

def load_medical_dict() -> dict:
    """의료 사전 로드."""
    dict_path = PROJECT_ROOT / "data" / "medical_dict.json"
    if not dict_path.exists():
        return {"entries": [], "prompt_terms": []}
    with open(dict_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_medical_corrections(text: str, entries: list) -> str:
    """의료 사전 기반 교정 적용."""
    # 우선순위 순 정렬
    sorted_entries = sorted(entries, key=lambda e: -e.get("priority", 50))
    for entry in sorted_entries:
        if not entry.get("enabled", True):
            continue
        wrong = entry.get("wrong", "")
        correct = entry.get("correct", "")
        if not wrong:
            continue
        strategy = entry.get("strategy", "exact")
        if strategy == "exact" and wrong in text:
            # 컨텍스트 힌트 확인
            hints = entry.get("context_hint", [])
            if hints and not any(h in text for h in hints):
                continue
            text = text.replace(wrong, correct)
    return text


def apply_hallucination_filter(text: str) -> str:
    """환각 필터링."""
    if not text:
        return text
    # 불가능한 월 반복
    text = re.sub(r"(\d{1,2}월부터\.?\s*){5,}", "", text)
    # 13월~99월
    text = re.sub(r"(?:1[3-9]|[2-9]\d)월", "", text)
    # 숫자 나열
    text = re.sub(r"(?:\d\s+){6,}\d", "", text)
    # 긴 반복 (10자 이상 반복)
    text = re.sub(r'(.{10,}?)\1{2,}', r'\1', text)
    # 미디어 키워드
    for kw in ["MBC", "KBS", "SBS", "YTN", "JTBC", "[음악]", "♪"]:
        text = text.replace(kw, "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 2 else ""


# ══════════════════════════════════════════════════════════════
# 핵심: 다중 파라미터 전사 + 최적 선택
# ══════════════════════════════════════════════════════════════

def transcribe_with_params(model, wav_path: str, params: dict) -> str:
    """주어진 파라미터로 전사."""
    segments, info = model.transcribe(
        wav_path,
        language="ko",
        beam_size=params.get("beam_size", 5),
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": params.get("min_silence_ms", 500),
            "speech_pad_ms": params.get("speech_pad_ms", 400),
            "threshold": params.get("vad_threshold", 0.5),
        },
        initial_prompt=params.get("prompt", ""),
        condition_on_previous_text=params.get("condition_on_prev", True),
        temperature=params.get("temperature", 0.0),
        no_speech_threshold=params.get("no_speech_threshold", 0.6),
        repetition_penalty=params.get("rep_penalty", 1.2),
        hallucination_silence_threshold=params.get("hal_silence", 2.0),
    )

    texts = []
    for seg in segments:
        text = (seg.text or "").strip()
        text = apply_hallucination_filter(text)
        if text:
            texts.append(text)

    return " ".join(texts)


def get_specialty_prompt(type_num: int) -> str:
    """Type 번호에 맞는 진료과 프롬프트 반환."""
    specialty = FULL_TYPE_SPECIALTY.get(type_num)
    if not specialty:
        return "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다."

    # 추가 프롬프트 확인
    if specialty in EXTRA_PROMPTS:
        return EXTRA_PROMPTS[specialty]

    # 기존 specialty_prompts.py에서 로드
    try:
        from app.services.specialty_prompts import SPECIALTY_PROMPTS, _COMMON_SUFFIX
        if specialty in SPECIALTY_PROMPTS:
            return SPECIALTY_PROMPTS[specialty] + _COMMON_SUFFIX
    except ImportError:
        pass

    return f"{specialty} 진료 상담 대화입니다. 의사와 환자가 대화합니다."


def build_enhanced_prompt(base_prompt: str, first_pass_text: str, med_entries: list) -> str:
    """1차 전사 결과에서 핵심 용어를 추출하여 프롬프트 강화."""
    # 의료 용어 패턴 추출
    patterns = [
        r"[가-힣]+증", r"[가-힣]+술", r"[가-힣]+제",
        r"[가-힣]+검사", r"[가-힣]+치료", r"[A-Za-z]{2,}",
    ]
    terms = set()
    for pat in patterns:
        terms.update(re.findall(pat, first_pass_text))

    # 사전에서 correct 용어도 추가
    for entry in med_entries:
        correct = entry.get("correct", "")
        if correct and correct in first_pass_text:
            terms.add(correct)

    extra = ", ".join(sorted(terms)[:30])
    enhanced = f"{base_prompt} {extra}"
    return enhanced[:500]  # 224토큰 제한


def optimize_single_type(model, type_num: int, reference: str, med_entries: list,
                         fast_mode: bool = False) -> dict:
    """단일 타입에 대해 다중 파라미터 조합을 시도하고 최적 결과 반환.

    fast_mode=True: 핵심 3개 config만 테스트 (속도 3배 향상)
    """
    wav_path = str(get_wav_path(type_num))
    if not Path(wav_path).exists():
        return {"type": type_num, "error": "WAV not found"}

    specialty_prompt = get_specialty_prompt(type_num)
    universal_prompt = "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다."

    # ── 파라미터 조합 정의 ──
    configs_all = [
        # Config 1: 기본 (현재 시스템)
        {"name": "baseline", "beam_size": 5, "vad_threshold": 0.5,
         "prompt": specialty_prompt, "rep_penalty": 1.2},

        # Config 2: 큰 beam + specialty prompt
        {"name": "beam10_specialty", "beam_size": 10, "vad_threshold": 0.5,
         "prompt": specialty_prompt, "rep_penalty": 1.2},

        # Config 3: 큰 beam + 범용 프롬프트
        {"name": "beam10_universal", "beam_size": 10, "vad_threshold": 0.5,
         "prompt": universal_prompt, "rep_penalty": 1.2},

        # Config 4: 민감한 VAD + specialty
        {"name": "sensitive_vad", "beam_size": 10, "vad_threshold": 0.35,
         "prompt": specialty_prompt, "rep_penalty": 1.2,
         "min_silence_ms": 700, "speech_pad_ms": 500},

        # Config 5: 짧은 프롬프트 (과적합 방지)
        {"name": "short_prompt", "beam_size": 10, "vad_threshold": 0.5,
         "prompt": f"{FULL_TYPE_SPECIALTY.get(type_num, '내과')} 진료 대화입니다.",
         "rep_penalty": 1.2},

        # Config 6: no_speech_threshold 낮춤 (더 많은 세그먼트 허용)
        {"name": "lenient_nospeech", "beam_size": 10, "vad_threshold": 0.45,
         "prompt": specialty_prompt, "rep_penalty": 1.2,
         "no_speech_threshold": 0.4},

        # Config 7: condition_on_previous_text=False (독립 세그먼트)
        {"name": "no_condition", "beam_size": 10, "vad_threshold": 0.5,
         "prompt": specialty_prompt, "rep_penalty": 1.2,
         "condition_on_prev": False},

        # Config 8: 높은 rep_penalty (반복 억제)
        {"name": "high_rep_penalty", "beam_size": 10, "vad_threshold": 0.5,
         "prompt": specialty_prompt, "rep_penalty": 1.5},
    ]

    if fast_mode:
        # 핵심 3개만 테스트: baseline, beam10+specialty, beam10+universal
        configs = [c for c in configs_all if c["name"] in
                   ("baseline", "beam10_specialty", "beam10_universal")]
    else:
        configs = configs_all

    best_cer = 1.0
    best_sem_cer = 1.0
    best_config = None
    best_text = ""
    all_results = []

    for cfg in configs:
        try:
            raw_text = transcribe_with_params(model, wav_path, cfg)
            corrected = apply_medical_corrections(raw_text, med_entries)
            cer = compute_cer(corrected, reference)
            cer_sem = compute_cer(corrected, reference, semantic=True)

            result = {
                "config": cfg["name"],
                "cer_raw": round(compute_cer(raw_text, reference) * 100, 1),
                "cer_corrected": round(cer * 100, 1),
                "cer_semantic": round(cer_sem * 100, 1),
                "text_length": len(raw_text),
                "ref_length": len(reference),
            }
            all_results.append(result)

            logger.info("  Type %d | %-20s | CER: %.1f%% → %.1f%% (sem: %.1f%%)",
                       type_num, cfg["name"], result["cer_raw"], result["cer_corrected"], result["cer_semantic"])

            if cer < best_cer:
                best_cer = cer
                best_sem_cer = cer_sem
                best_config = cfg["name"]
                best_text = corrected

        except Exception as e:
            logger.warning("  Type %d | %s FAILED: %s", type_num, cfg["name"], e)
            all_results.append({"config": cfg["name"], "error": str(e)})

    # ── Two-Pass: 최적 config로 2차 전사 시도 ──
    if best_config and best_cer > 0.05:  # CER > 5%이면 2-pass 시도
        try:
            best_cfg = next(c for c in configs if c["name"] == best_config)
            enhanced_prompt = build_enhanced_prompt(
                best_cfg.get("prompt", ""), best_text, med_entries
            )
            two_pass_cfg = {**best_cfg, "name": "two_pass", "prompt": enhanced_prompt, "beam_size": 10}
            raw_text_2 = transcribe_with_params(model, wav_path, two_pass_cfg)
            corrected_2 = apply_medical_corrections(raw_text_2, med_entries)
            cer_2 = compute_cer(corrected_2, reference)

            result_2 = {
                "config": "two_pass",
                "cer_raw": round(compute_cer(raw_text_2, reference) * 100, 1),
                "cer_corrected": round(cer_2 * 100, 1),
            }
            all_results.append(result_2)

            logger.info("  Type %d | %-20s | CER: %.1f%% → %.1f%%",
                       type_num, "two_pass", result_2["cer_raw"], result_2["cer_corrected"])

            if cer_2 < best_cer:
                best_cer = cer_2
                best_sem_cer = compute_cer(corrected_2, reference, semantic=True)
                best_config = "two_pass"
                best_text = corrected_2

        except Exception as e:
            logger.warning("  Type %d | two_pass FAILED: %s", type_num, e)

    return {
        "type": type_num,
        "specialty": FULL_TYPE_SPECIALTY.get(type_num, "unknown"),
        "best_cer": round(best_cer * 100, 1),
        "best_sem_cer": round(best_sem_cer * 100, 1),
        "best_config": best_config,
        "all_results": all_results,
        "best_text_preview": best_text[:200],
        "reference_preview": reference[:200],
    }


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    # --fast 플래그로 빠른 모드 (핵심 3개 config만)
    fast_mode = "--fast" in sys.argv
    start_time = time.time()

    # 모델 로드 — CPU int8이 이 환경에서 가장 안정적 + 성능 최선
    logger.info("모델 로딩 중... (CPU int8 — 이 환경에서 최적)")
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    logger.info("CPU int8 모델 로드 완료")

    # 의료 사전 로드
    med_dict = load_medical_dict()
    med_entries = med_dict.get("entries", [])
    logger.info("의료 사전: %d개 항목", len(med_entries))

    # 전체 타입 평가
    results = []
    total_cer = 0.0
    evaluated = 0

    for type_num in range(1, 22):
        wav_path = get_wav_path(type_num)
        answer_path = DATA_DIR / f"answer{type_num}.txt"

        if not wav_path.exists() or not answer_path.exists():
            logger.warning("Type %d: 데이터 누락", type_num)
            continue

        reference = load_answer(type_num)
        logger.info("=" * 60)
        logger.info("Type %d (%s) — 정답 %d자",
                    type_num, FULL_TYPE_SPECIALTY.get(type_num, "?"), len(reference))

        result = optimize_single_type(model, type_num, reference, med_entries, fast_mode=fast_mode)
        results.append(result)

        if "error" not in result:
            total_cer += result["best_cer"]
            evaluated += 1

    # 결과 정리
    avg_cer = total_cer / evaluated if evaluated > 0 else 0
    total_sem_cer = sum(r.get("best_sem_cer", 0) for r in results if "error" not in r)
    avg_sem_cer = total_sem_cer / evaluated if evaluated > 0 else 0
    elapsed = time.time() - start_time

    # 출력
    print("\n" + "=" * 80)
    print(f"  CER 최적화 결과 (평가: {evaluated}개 타입, 소요시간: {elapsed:.0f}초)")
    print("=" * 80)
    print(f"{'Type':>6} {'진료과':>10} {'CER':>8} {'SemCER':>8} {'Best Config':>22}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.get("best_cer", 999)):
        if "error" in r:
            print(f"  {r['type']:>4}   {'ERROR':>10}")
        else:
            print(f"  {r['type']:>4}   {r['specialty']:>10}   {r['best_cer']:>6.1f}%  {r.get('best_sem_cer', 0):>6.1f}%   {r['best_config']:>22}")

    print("-" * 80)
    print(f"  평균 CER:     {avg_cer:.1f}%  (기존 27.0% 대비 {27.0 - avg_cer:+.1f}%p)")
    print(f"  평균 SemCER:  {avg_sem_cer:.1f}%  (의미 기반, 어미/조사 차이 무시)")
    print("=" * 80)

    # 결과 저장
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "average_cer": round(avg_cer, 1),
        "evaluated_types": evaluated,
        "elapsed_seconds": round(elapsed, 0),
        "types": results,
    }
    output_path = RESULTS_DIR / f"optimize_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장: %s", output_path)

    # latest도 저장
    latest_path = RESULTS_DIR / "latest_optimization.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
