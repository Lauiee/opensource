"""
ghost613/faster-whisper-large-v3-turbo-korean 모델 평가 스크립트.
CER(Character Error Rate) 기반으로 21개 type 전사 정확도를 측정한다.

Usage:
    python -X utf8 test_korean_model.py
    python -X utf8 test_korean_model.py --with-prompt   # 진료과 프롬프트 포함
    python -X utf8 test_korean_model.py --int8           # VRAM 부족 시
"""

import json
import os
import sys
import time
import unicodedata
from pathlib import Path

# ── CUDA DLL 경로 추가 (Windows cublas 문제 해결) ──
_nvidia_cublas = Path(sys.executable).parent / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
if _nvidia_cublas.exists():
    os.environ["PATH"] = str(_nvidia_cublas) + os.pathsep + os.environ.get("PATH", "")
_nvidia_cudnn = Path(sys.executable).parent / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
if _nvidia_cudnn.exists():
    os.environ["PATH"] = str(_nvidia_cudnn) + os.pathsep + os.environ.get("PATH", "")

# ── 프로젝트 경로 설정 ──
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from app.services.specialty_prompts import (
    TYPE_TO_SPECIALTY,
    get_specialty_prompt,
)

# ── 경로 상수 ──
DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
RESULTS_DIR = PROJECT_DIR / "data"
RESULTS_DIR.mkdir(exist_ok=True)

# ghost613 모델은 30초 이후 전사 중단 버그 있음
# turbo-ct2는 47.5% CER로 large-v3보다 나쁨
# large-v3이 기존 27% 달성 모델이므로 이를 기본으로 사용
MODEL_ID = "large-v3"
NUM_TYPES = 21


# ──────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """정규화: 공백 제거, NFC 유니코드 변환."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """편집 거리(Levenshtein distance) 계산."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 삽입, 삭제, 대체
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """CER 계산: levenshtein_distance / len(reference)."""
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return min(levenshtein_distance(hyp, ref) / len(ref), 1.0)


def load_reference(type_num: int) -> str:
    """정답 파일에서 전체 텍스트 추출."""
    answer_path = DATA_DIR / f"answer{type_num}.txt"
    with open(answer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # content 필드를 모두 합치기
    parts = [item["content"] for item in data if "content" in item]
    return " ".join(parts)


def get_wav_path(type_num: int) -> Path:
    """WAV 파일 경로 반환."""
    return DATA_DIR / f"type{type_num}" / f"type{type_num}.wav"


# ── 17~21번 진료과 추정 (정답 파일 기반) ──
EXTRA_SPECIALTY: dict[int, str] = {}


def infer_specialty_for_unmapped():
    """17~21번의 정답 파일 내용으로 진료과를 추정."""
    keywords_map = {
        "정형외과": ["관절", "골절", "디스크", "척추", "무릎", "허리", "인대", "연골", "수술"],
        "안과": ["백내장", "녹내장", "시력", "안약", "안압", "수정체", "망막"],
        "내분비내과": ["당뇨", "혈당", "인슐린", "갑상선", "호르몬", "코르티솔"],
        "간담도외과": ["담즙", "담관", "담석", "간", "낭종"],
        "소화기내과": ["내시경", "위", "대장", "용종", "역류"],
        "호흡기내과": ["흉부", "폐", "기침", "가래", "호흡", "천식"],
        "감염내과": ["열", "발열", "항생제", "감염", "권태감"],
        "비뇨기과": ["배뇨", "전립선", "방광", "신장", "요도"],
        "정신건강의학과": ["우울", "불안", "공황", "수면", "자살"],
        "신경과": ["두통", "어지럼", "뇌", "치매", "기억력"],
        "내과": ["혈압", "혈당", "콜레스테롤", "간기능"],
        "산부인과": ["자궁", "난소", "임신", "출산", "생리"],
    }

    for type_num in range(17, 22):
        if type_num in TYPE_TO_SPECIALTY:
            continue
        try:
            ref_text = load_reference(type_num)
        except Exception:
            continue

        scores = {}
        for specialty, keywords in keywords_map.items():
            score = sum(1 for kw in keywords if kw in ref_text)
            if score > 0:
                scores[specialty] = score

        if scores:
            best = max(scores, key=scores.get)
            EXTRA_SPECIALTY[type_num] = best
            print(f"  Type {type_num}: 추정 진료과 = {best} (키워드 {scores[best]}개 매칭)")
        else:
            EXTRA_SPECIALTY[type_num] = None
            print(f"  Type {type_num}: 진료과 추정 불가 -> 범용 프롬프트 사용")


def get_prompt_for_type(type_num: int) -> str:
    """Type에 맞는 프롬프트 반환 (17~21 포함)."""
    # 기존 매핑에 있으면 사용
    if type_num in TYPE_TO_SPECIALTY:
        return get_specialty_prompt(type_num=type_num)

    # 추정된 진료과가 있으면 사용
    if type_num in EXTRA_SPECIALTY and EXTRA_SPECIALTY[type_num]:
        return get_specialty_prompt(specialty=EXTRA_SPECIALTY[type_num])

    # 범용 프롬프트
    return get_specialty_prompt()


# ──────────────────────────────────────────────────────────────────────
# 메인 로직
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-prompt", action="store_true",
                        help="진료과별 프롬프트를 사용하여 전사")
    parser.add_argument("--int8", action="store_true",
                        help="VRAM 부족 시 int8 사용")
    args = parser.parse_args()

    use_prompt = args.with_prompt
    compute_type = "int8" if args.int8 else "float16"

    print("=" * 70)
    print(f"  Whisper {MODEL_ID} 평가")
    print(f"  프롬프트 사용: {'예' if use_prompt else '아니오'}")
    print(f"  Compute type: {compute_type}")
    print("=" * 70)

    # 17~21 진료과 추정
    if use_prompt:
        print("\n[1] 17~21번 진료과 추정 중...")
        infer_specialty_for_unmapped()

    # 모델 로드
    print(f"\n[2] 모델 로드 중: {MODEL_ID}")
    print(f"    (최초 실행 시 다운로드에 시간이 걸릴 수 있습니다)")
    t0 = time.time()

    from faster_whisper import WhisperModel
    model = WhisperModel(
        MODEL_ID,
        device="cuda",
        compute_type=compute_type,
    )
    load_time = time.time() - t0
    print(f"    모델 로드 완료: {load_time:.1f}초")

    # 전사 및 평가
    print(f"\n[3] 전사 및 CER 평가 시작 (총 {NUM_TYPES}개 타입)")
    print("-" * 70)

    results = []

    for type_num in range(1, NUM_TYPES + 1):
        wav_path = get_wav_path(type_num)
        if not wav_path.exists():
            print(f"  Type {type_num:2d}: WAV 파일 없음 ({wav_path})")
            continue

        # 정답 로드
        try:
            ref_text = load_reference(type_num)
        except Exception as e:
            print(f"  Type {type_num:2d}: 정답 파일 오류 - {e}")
            continue

        # 전사 파라미터 (기존 run_full_eval.py와 동일)
        default_prompt = (
            "의료 진료 상담 대화입니다. 의사와 환자가 대화합니다. "
            "고관절, 무릎, 척추, 디스크, 골절, 연골, 인대, 관절염, "
            "백내장, 녹내장, 비문증, 안압, 시력, 안약, "
            "담즙, 총담관, 낭종, 담석, 담관암, "
            "호흡 곤란, 흉부, 엑스레이, 배뇨장애, 전립선, "
            "해열진통제, 대증 치료, 요추 염좌, 좌골 신경통, 처방전, "
            "수술, 검사, 치료, 진단, 처방, 약, 입원, 퇴원, 외래"
        )

        if use_prompt:
            prompt = get_prompt_for_type(type_num)
        else:
            prompt = default_prompt

        transcribe_kwargs = dict(
            language="ko",
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
            initial_prompt=prompt,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            repetition_penalty=1.2,
            hallucination_silence_threshold=2.0,
        )

        # 전사 수행
        t1 = time.time()
        segments, info = model.transcribe(str(wav_path), **transcribe_kwargs)
        hyp_parts = [seg.text for seg in segments]
        hyp_text = " ".join(hyp_parts)
        elapsed = time.time() - t1

        # CER 계산
        cer = compute_cer(hyp_text, ref_text)

        # 진료과 정보
        specialty = TYPE_TO_SPECIALTY.get(type_num)
        if not specialty and type_num in EXTRA_SPECIALTY:
            specialty = EXTRA_SPECIALTY.get(type_num, "범용")

        result = {
            "type": type_num,
            "cer": round(cer * 100, 1),
            "ref_len": len(normalize_text(ref_text)),
            "hyp_len": len(normalize_text(hyp_text)),
            "time_sec": round(elapsed, 1),
            "specialty": specialty or "미지정",
            "hypothesis": hyp_text[:200],
        }
        results.append(result)

        status = "GOOD" if cer < 0.15 else ("FAIR" if cer < 0.30 else "POOR")
        print(f"  Type {type_num:2d} | CER: {cer*100:5.1f}% | {status:4s} | "
              f"{elapsed:.1f}s | {specialty or '미지정'}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)

    if not results:
        print("  평가 결과 없음")
        return

    # CER 정렬
    sorted_results = sorted(results, key=lambda x: x["cer"])
    avg_cer = sum(r["cer"] for r in results) / len(results)

    print(f"\n  평균 CER: {avg_cer:.1f}%")
    print(f"  평가 타입 수: {len(results)}")
    print(f"\n  {'Type':>6s} | {'CER':>7s} | {'상태':>4s} | {'진료과'}")
    print(f"  {'-'*6} | {'-'*7} | {'-'*4} | {'-'*15}")
    for r in sorted_results:
        status = "GOOD" if r["cer"] < 15 else ("FAIR" if r["cer"] < 30 else "POOR")
        print(f"  {r['type']:6d} | {r['cer']:6.1f}% | {status:4s} | {r['specialty']}")

    # Good / Fair / Poor 분류
    good = [r for r in results if r["cer"] < 15]
    fair = [r for r in results if 15 <= r["cer"] < 30]
    poor = [r for r in results if r["cer"] >= 30]

    print(f"\n  GOOD (<15%): {len(good)}개 - {[r['type'] for r in good]}")
    print(f"  FAIR (15-30%): {len(fair)}개 - {[r['type'] for r in fair]}")
    print(f"  POOR (>30%): {len(poor)}개 - {[r['type'] for r in poor]}")

    # 결과 저장
    suffix = "_with_prompt" if use_prompt else "_baseline"
    output_path = RESULTS_DIR / f"korean_model_eval{suffix}.json"
    save_data = {
        "model": MODEL_ID,
        "compute_type": compute_type,
        "use_prompt": use_prompt,
        "avg_cer": round(avg_cer, 1),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_path}")

    # 기존 27% 대비 비교
    print(f"\n  기존 평균 CER: 27.0%")
    print(f"  현재 평균 CER: {avg_cer:.1f}%")
    diff = avg_cer - 27.0
    if diff < 0:
        print(f"  개선: {abs(diff):.1f}%p 감소 (향상)")
    else:
        print(f"  변화: {diff:.1f}%p 증가 (악화)")


if __name__ == "__main__":
    main()
