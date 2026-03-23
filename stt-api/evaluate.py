"""WER(Word Error Rate) 자동 평가 시스템.

정답지(dalpha)와 STT 결과(donkey)를 비교하여
타입별/전체 성능을 측정한다.

사용법:
    python evaluate.py                    # 전체 평가
    python evaluate.py --types 2 6 8      # 특정 타입만
    python evaluate.py --detail           # 세그먼트별 상세 비교
    python evaluate.py --save report.json # 결과 저장
"""

import io
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
# WER 계산
# ──────────────────────────────────────────────

def _tokenize_korean(text: str) -> list[str]:
    """한국어 텍스트를 토큰(어절) 단위로 분리."""
    # 구두점 제거, 공백 기준 분리
    text = re.sub(r"[.,!?;:\"'()\[\]{}…·\-–—]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def compute_wer(reference: str, hypothesis: str) -> dict:
    """WER 계산 (Levenshtein distance 기반).

    Returns:
        {
            "wer": float (0.0~1.0+),
            "substitutions": int,
            "insertions": int,
            "deletions": int,
            "ref_words": int,
            "hyp_words": int,
        }
    """
    ref_tokens = _tokenize_korean(reference)
    hyp_tokens = _tokenize_korean(hypothesis)

    r_len = len(ref_tokens)
    h_len = len(hyp_tokens)

    if r_len == 0:
        return {
            "wer": 1.0 if h_len > 0 else 0.0,
            "substitutions": 0, "insertions": h_len, "deletions": 0,
            "ref_words": 0, "hyp_words": h_len,
        }

    # DP 테이블
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    # 역추적으로 S/I/D 분류
    i, j = r_len, h_len
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    wer = (subs + ins + dels) / r_len if r_len > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": r_len,
        "hyp_words": h_len,
    }


def compute_cer(reference: str, hypothesis: str) -> float:
    """CER(Character Error Rate) 계산."""
    ref_chars = list(re.sub(r"\s+", "", reference))
    hyp_chars = list(re.sub(r"\s+", "", hypothesis))

    r_len = len(ref_chars)
    h_len = len(hyp_chars)

    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0

    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

    return round(d[r_len][h_len] / r_len, 4)


# ──────────────────────────────────────────────
# 정답지 로드
# ──────────────────────────────────────────────

def load_reference(type_num: int, test_set_dir: str) -> list[dict]:
    """정답지(dalpha) 로드."""
    dalpha_path = Path(test_set_dir) / f"type{type_num}" / f"dalpha_type{type_num}.txt"
    if not dalpha_path.exists():
        return []
    try:
        data = json.loads(dalpha_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_stt_result(type_num: int, api_base: str = "http://localhost:8205") -> list[dict]:
    """API에서 STT 결과 로드."""
    url = f"{api_base}/api/viewer/stt/{type_num}/donkey"
    try:
        r = urllib.request.urlopen(url, timeout=300)
        data = json.loads(r.read())
        if isinstance(data, dict):
            return data.get("segments", [])
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  [ERROR] Type {type_num} API 호출 실패: {e}")
        return []


# ──────────────────────────────────────────────
# 텍스트 추출 및 정렬
# ──────────────────────────────────────────────

def extract_texts(segments: list[dict]) -> str:
    """세그먼트 목록에서 전체 텍스트 추출.

    정답지(dalpha): "content" 필드
    STT 결과(donkey): "corrected" or "text" or "original" 필드
    """
    texts = []
    for seg in segments:
        # dalpha 형식 (content 필드)
        text = seg.get("content", "")
        if not text:
            # donkey STT 형식
            text = seg.get("corrected") or seg.get("text") or seg.get("original", "")
        if text.strip():
            texts.append(text.strip())
    return " ".join(texts)


def extract_segment_texts(segments: list[dict]) -> list[str]:
    """세그먼트별 텍스트 추출."""
    result = []
    for seg in segments:
        text = seg.get("content", "")
        if not text:
            text = seg.get("corrected") or seg.get("text") or seg.get("original", "")
        if text.strip():
            result.append(text.strip())
    return result


# ──────────────────────────────────────────────
# 세그먼트 정렬 비교
# ──────────────────────────────────────────────

def align_segments(ref_segs: list[dict], hyp_segs: list[dict]) -> list[dict]:
    """정답지와 STT 세그먼트를 정렬하여 비교.

    인덱스 기반 정렬 (시간 정보가 다를 수 있으므로).
    """
    ref_texts = extract_segment_texts(ref_segs)
    hyp_texts = extract_segment_texts(hyp_segs)

    alignments = []
    max_len = max(len(ref_texts), len(hyp_texts))

    for i in range(max_len):
        ref = ref_texts[i] if i < len(ref_texts) else "[누락]"
        hyp = hyp_texts[i] if i < len(hyp_texts) else "[누락]"

        seg_wer = compute_wer(ref, hyp) if ref != "[누락]" and hyp != "[누락]" else {"wer": 1.0}

        alignments.append({
            "index": i,
            "reference": ref,
            "hypothesis": hyp,
            "wer": seg_wer["wer"],
            "match": ref == hyp,
        })

    return alignments


# ──────────────────────────────────────────────
# 평가 실행
# ──────────────────────────────────────────────

@dataclass
class TypeResult:
    type_num: int
    wer: float = 0.0
    cer: float = 0.0
    ref_words: int = 0
    hyp_words: int = 0
    ref_segments: int = 0
    hyp_segments: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    elapsed: float = 0.0
    errors: list = field(default_factory=list)  # 주요 오류 목록


def evaluate_type(
    type_num: int,
    test_set_dir: str,
    api_base: str = "http://localhost:8205",
    detail: bool = False,
) -> TypeResult:
    """단일 타입 평가."""
    result = TypeResult(type_num=type_num)

    # 정답지 로드
    ref_segs = load_reference(type_num, test_set_dir)
    if not ref_segs:
        print(f"  Type {type_num}: 정답지 없음, 건너뜀")
        return result

    result.ref_segments = len(ref_segs)

    # STT 실행
    start = time.time()
    hyp_segs = load_stt_result(type_num, api_base)
    result.elapsed = time.time() - start
    result.hyp_segments = len(hyp_segs)

    if not hyp_segs:
        result.wer = 1.0
        result.cer = 1.0
        return result

    # 전체 텍스트 WER/CER
    ref_text = extract_texts(ref_segs)
    hyp_text = extract_texts(hyp_segs)

    wer_result = compute_wer(ref_text, hyp_text)
    result.wer = wer_result["wer"]
    result.cer = compute_cer(ref_text, hyp_text)
    result.ref_words = wer_result["ref_words"]
    result.hyp_words = wer_result["hyp_words"]
    result.substitutions = wer_result["substitutions"]
    result.insertions = wer_result["insertions"]
    result.deletions = wer_result["deletions"]

    # 세그먼트 정렬 비교 (상세 모드)
    if detail:
        alignments = align_segments(ref_segs, hyp_segs)
        for a in alignments:
            if not a["match"] and a["wer"] > 0.3:
                result.errors.append({
                    "index": a["index"],
                    "ref": a["reference"][:60],
                    "hyp": a["hypothesis"][:60],
                    "wer": a["wer"],
                })

    return result


def evaluate_all(
    test_set_dir: str,
    api_base: str = "http://localhost:8205",
    type_nums: list[int] | None = None,
    detail: bool = False,
) -> list[TypeResult]:
    """전체 평가 실행."""
    if type_nums is None:
        type_nums = list(range(1, 22))

    results = []
    total_ref_words = 0
    total_errors = 0

    print("=" * 70)
    print("  의료 STT 성능 평가 (WER/CER)")
    print("=" * 70)
    print()

    for tn in type_nums:
        print(f"  Type {tn:2d} 평가 중...", end=" ", flush=True)
        r = evaluate_type(tn, test_set_dir, api_base, detail)
        results.append(r)

        wer_pct = r.wer * 100
        cer_pct = r.cer * 100

        # 성능 등급
        if wer_pct <= 10:
            grade = "S"
        elif wer_pct <= 20:
            grade = "A"
        elif wer_pct <= 35:
            grade = "B"
        elif wer_pct <= 50:
            grade = "C"
        elif wer_pct <= 70:
            grade = "D"
        else:
            grade = "F"

        print(
            f"WER={wer_pct:5.1f}% CER={cer_pct:5.1f}% "
            f"[{grade}] ({r.ref_words}w, {r.ref_segments}→{r.hyp_segments}seg, "
            f"S={r.substitutions} I={r.insertions} D={r.deletions}) "
            f"{r.elapsed:.1f}s"
        )

        total_ref_words += r.ref_words
        total_errors += r.substitutions + r.insertions + r.deletions

        # 상세 오류 출력
        if detail and r.errors:
            for err in r.errors[:5]:
                print(f"    [{err['index']}] WER={err['wer']:.0%}")
                print(f"      정답: {err['ref']}")
                print(f"      STT:  {err['hyp']}")

    # 전체 요약
    print()
    print("=" * 70)
    avg_wer = sum(r.wer for r in results) / len(results) if results else 0
    avg_cer = sum(r.cer for r in results) / len(results) if results else 0
    weighted_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

    print(f"  평균 WER: {avg_wer * 100:.1f}% (가중: {weighted_wer * 100:.1f}%)")
    print(f"  평균 CER: {avg_cer * 100:.1f}%")
    print(f"  총 단어: {total_ref_words}, 총 오류: {total_errors}")
    print()

    # 등급별 분포
    grades = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        w = r.wer * 100
        if w <= 10: grades["S"] += 1
        elif w <= 20: grades["A"] += 1
        elif w <= 35: grades["B"] += 1
        elif w <= 50: grades["C"] += 1
        elif w <= 70: grades["D"] += 1
        else: grades["F"] += 1

    print(f"  등급 분포: S={grades['S']} A={grades['A']} B={grades['B']} "
          f"C={grades['C']} D={grades['D']} F={grades['F']}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="의료 STT WER 평가")
    parser.add_argument("--types", nargs="+", type=int, help="평가할 타입 번호")
    parser.add_argument("--detail", action="store_true", help="세그먼트별 상세 비교")
    parser.add_argument("--api", default="http://localhost:8205", help="API 베이스 URL")
    parser.add_argument("--save", help="결과 저장 경로 (JSON)")
    parser.add_argument("--test-set", default=r"C:\Users\USER\Dropbox\패밀리룸\N Park\튜링\test_set")

    args = parser.parse_args()

    results = evaluate_all(
        test_set_dir=args.test_set,
        api_base=args.api,
        type_nums=args.types,
        detail=args.detail,
    )

    if args.save:
        save_data = []
        for r in results:
            save_data.append({
                "type_num": r.type_num,
                "wer": r.wer,
                "cer": r.cer,
                "ref_words": r.ref_words,
                "hyp_words": r.hyp_words,
                "substitutions": r.substitutions,
                "insertions": r.insertions,
                "deletions": r.deletions,
                "elapsed": r.elapsed,
            })
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.save}")
