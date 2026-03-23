"""WER/CER 자동 평가 시스템 — 한국어 의료 STT 품질 측정.

한국어 특성상 어절(Word) 단위 WER보다 글자(Character) 단위 CER이
더 의미 있으므로, CER을 주요 지표로 사용하되 WER도 함께 산출한다.
"""

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 기본 경로 설정 (환경에 따라 오버라이드 가능)
# ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET_PATH = _PROJECT_ROOT.parent / "data_set"
_DEFAULT_DICT_PATH = _PROJECT_ROOT / "data" / "medical_dict.json"
_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "data" / "evaluation_results"


# ──────────────────────────────────────────────────────────────────────
# 텍스트 정규화
# ──────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """비교 전 텍스트 정규화.

    - 유니코드 NFC 정규화
    - 연속 공백 → 단일 공백
    - 구두점 통일 (전각 → 반각)
    - 양끝 공백 제거
    """
    text = unicodedata.normalize("NFC", text)
    # 전각 구두점 → 반각
    replacements = {
        "\u3002": ".",  # 。 → .
        "\uff0c": ",",  # ， → ,
        "\uff1f": "?",  # ？ → ?
        "\uff01": "!",  # ！ → !
        "\u00b7": " ",  # · → space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # 연속 공백 축소
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Levenshtein 거리 및 정렬 (편집 연산 추적)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """Levenshtein 정렬 결과."""
    distance: int
    insertions: int
    deletions: int
    substitutions: int
    ref_length: int
    hyp_length: int
    error_rate: float
    alignment_ops: list[tuple[str, str, str]] = field(default_factory=list)
    # alignment_ops: [(op_type, ref_token, hyp_token), ...]
    # op_type: "C"(correct), "S"(sub), "I"(ins), "D"(del)


def _levenshtein_align(ref: list[str], hyp: list[str]) -> AlignmentResult:
    """DP 기반 Levenshtein 정렬 + 역추적.

    ref, hyp: 토큰(글자 또는 단어) 리스트
    """
    n = len(ref)
    m = len(hyp)

    # DP 테이블: dp[i][j] = (cost, op)
    dp = [[(0, "")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, "D")
    for j in range(1, m + 1):
        dp[0][j] = (j, "I")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], "C")
            else:
                sub_cost = dp[i - 1][j - 1][0] + 1
                del_cost = dp[i - 1][j][0] + 1
                ins_cost = dp[i][j - 1][0] + 1
                min_cost = min(sub_cost, del_cost, ins_cost)
                if min_cost == sub_cost:
                    dp[i][j] = (sub_cost, "S")
                elif min_cost == del_cost:
                    dp[i][j] = (del_cost, "D")
                else:
                    dp[i][j] = (ins_cost, "I")

    # 역추적
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j][1] == "C":
            ops.append(("C", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j][1] == "S":
            ops.append(("S", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j][1] == "D":
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()

    subs = sum(1 for op, _, _ in ops if op == "S")
    dels = sum(1 for op, _, _ in ops if op == "D")
    ins = sum(1 for op, _, _ in ops if op == "I")
    dist = subs + dels + ins
    rate = dist / max(n, 1)

    return AlignmentResult(
        distance=dist,
        insertions=ins,
        deletions=dels,
        substitutions=subs,
        ref_length=n,
        hyp_length=m,
        error_rate=rate,
        alignment_ops=ops,
    )


# ──────────────────────────────────────────────────────────────────────
# CER / WER 산출
# ──────────────────────────────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> AlignmentResult:
    """글자(Character) 단위 오류율 산출.

    한국어에서는 CER이 WER보다 직관적이다.
    공백은 제거한 후 글자 단위로 비교한다.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # 공백 제거 (순수 글자 비교)
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    return _levenshtein_align(ref_chars, hyp_chars)


def compute_wer(reference: str, hypothesis: str) -> AlignmentResult:
    """어절(Word) 단위 오류율 산출."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    return _levenshtein_align(ref_words, hyp_words)


# ──────────────────────────────────────────────────────────────────────
# 자주 발생하는 오류 추출
# ──────────────────────────────────────────────────────────────────────

def extract_common_errors(
    alignment: AlignmentResult, top_n: int = 20,
) -> list[dict]:
    """정렬 결과에서 치환/삽입/삭제 오류를 빈도순으로 추출."""
    error_counter: Counter = Counter()
    for op, ref_tok, hyp_tok in alignment.alignment_ops:
        if op == "S":
            error_counter[("치환", ref_tok, hyp_tok)] += 1
        elif op == "D":
            error_counter[("삭제", ref_tok, "")] += 1
        elif op == "I":
            error_counter[("삽입", "", hyp_tok)] += 1

    results = []
    for (err_type, ref_tok, hyp_tok), count in error_counter.most_common(top_n):
        results.append({
            "유형": err_type,
            "정답": ref_tok,
            "인식": hyp_tok,
            "횟수": count,
        })
    return results


# ──────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────────────

def load_ground_truth(dataset_path: Path, type_num: int) -> str:
    """type{N}_full_transcript.txt 에서 정답 텍스트를 로드."""
    gt_file = dataset_path / f"type{type_num}" / f"type{type_num}_full_transcript.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"정답 파일을 찾을 수 없습니다: {gt_file}")
    return gt_file.read_text(encoding="utf-8").strip()


def load_stt_result(dataset_path: Path, type_num: int) -> str:
    """donkey_type{N}.txt (JSON) 에서 STT 결과 텍스트를 추출.

    JSON 형식: [{"role": str, "content": str, ...}, ...]
    모든 content를 순서대로 합쳐 반환한다.
    """
    stt_file = dataset_path / f"type{type_num}" / f"donkey_type{type_num}.txt"
    if not stt_file.exists():
        raise FileNotFoundError(f"STT 결과 파일을 찾을 수 없습니다: {stt_file}")

    raw = stt_file.read_text(encoding="utf-8")
    try:
        segments = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"STT 결과 JSON 파싱 실패 ({stt_file}): {e}") from e

    if not isinstance(segments, list):
        raise ValueError(f"STT 결과가 리스트가 아닙니다: {stt_file}")

    contents = [seg["content"] for seg in segments if seg.get("content")]
    return " ".join(contents)


# ──────────────────────────────────────────────────────────────────────
# 의료 용어 교정 엔진 로딩
# ──────────────────────────────────────────────────────────────────────

def _get_correction_engine(dict_path: Path | None = None):
    """의료 용어 교정 엔진 인스턴스를 반환.

    이미 초기화된 엔진이 있으면 재사용하고,
    없으면 새로 초기화한다.
    """
    try:
        from app.medterm.engine import get_engine, init_engine
        from app.medterm.store import DictionaryStore

        engine = get_engine()
        if engine is not None:
            return engine

        # 엔진이 초기화되지 않았으면 새로 생성
        dp = dict_path or _DEFAULT_DICT_PATH
        if dp.exists():
            engine = init_engine(dp)
            return engine
        else:
            logger.warning("의료 사전 파일 없음: %s — 교정 없이 진행", dp)
            return None
    except Exception as e:
        logger.warning("교정 엔진 로딩 실패: %s — 교정 없이 진행", e)
        return None


def apply_correction(text: str, engine) -> str:
    """교정 엔진을 적용하여 교정된 텍스트 반환."""
    if engine is None:
        return text
    try:
        result = engine.correct(text)
        return result.text
    except Exception as e:
        logger.warning("교정 적용 실패: %s", e)
        return text


# ──────────────────────────────────────────────────────────────────────
# 단일 타입 평가
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TypeEvaluationResult:
    """단일 type 평가 결과."""
    type_num: int
    # 원본 (교정 전) 지표
    cer_before: float
    wer_before: float
    cer_detail_before: dict = field(default_factory=dict)
    wer_detail_before: dict = field(default_factory=dict)
    # 교정 후 지표
    cer_after: float = 0.0
    wer_after: float = 0.0
    cer_detail_after: dict = field(default_factory=dict)
    wer_detail_after: dict = field(default_factory=dict)
    # 개선율
    cer_improvement: float = 0.0
    wer_improvement: float = 0.0
    # 오류 분석
    common_errors_before: list = field(default_factory=list)
    common_errors_after: list = field(default_factory=list)
    # 메타 정보
    ground_truth_length: int = 0
    stt_text_length: int = 0
    corrected_text_length: int = 0
    correction_applied: bool = False
    error: str | None = None


def evaluate_single_type(
    type_num: int,
    dataset_path: Path | None = None,
    dict_path: Path | None = None,
    engine=None,
    top_n_errors: int = 20,
) -> TypeEvaluationResult:
    """단일 type에 대한 CER/WER 평가를 수행."""
    ds_path = dataset_path or _DEFAULT_DATASET_PATH
    result = TypeEvaluationResult(
        type_num=type_num,
        cer_before=0.0,
        wer_before=0.0,
    )

    # 1. 데이터 로딩
    try:
        ground_truth = load_ground_truth(ds_path, type_num)
        stt_text = load_stt_result(ds_path, type_num)
    except (FileNotFoundError, ValueError) as e:
        result.error = str(e)
        logger.error("type%d 데이터 로딩 실패: %s", type_num, e)
        return result

    result.ground_truth_length = len(ground_truth)
    result.stt_text_length = len(stt_text)

    # 2. 교정 전 CER/WER
    cer_align = compute_cer(ground_truth, stt_text)
    wer_align = compute_wer(ground_truth, stt_text)

    result.cer_before = round(cer_align.error_rate * 100, 2)
    result.wer_before = round(wer_align.error_rate * 100, 2)
    result.cer_detail_before = {
        "삽입": cer_align.insertions,
        "삭제": cer_align.deletions,
        "치환": cer_align.substitutions,
        "정답_길이": cer_align.ref_length,
        "인식_길이": cer_align.hyp_length,
    }
    result.wer_detail_before = {
        "삽입": wer_align.insertions,
        "삭제": wer_align.deletions,
        "치환": wer_align.substitutions,
        "정답_어절수": wer_align.ref_length,
        "인식_어절수": wer_align.hyp_length,
    }
    result.common_errors_before = extract_common_errors(cer_align, top_n_errors)

    # 3. 교정 엔진 적용
    if engine is None:
        engine = _get_correction_engine(dict_path)

    if engine is not None:
        corrected_text = apply_correction(stt_text, engine)
        result.correction_applied = True
        result.corrected_text_length = len(corrected_text)

        cer_align_after = compute_cer(ground_truth, corrected_text)
        wer_align_after = compute_wer(ground_truth, corrected_text)

        result.cer_after = round(cer_align_after.error_rate * 100, 2)
        result.wer_after = round(wer_align_after.error_rate * 100, 2)
        result.cer_detail_after = {
            "삽입": cer_align_after.insertions,
            "삭제": cer_align_after.deletions,
            "치환": cer_align_after.substitutions,
            "정답_길이": cer_align_after.ref_length,
            "인식_길이": cer_align_after.hyp_length,
        }
        result.wer_detail_after = {
            "삽입": wer_align_after.insertions,
            "삭제": wer_align_after.deletions,
            "치환": wer_align_after.substitutions,
            "정답_어절수": wer_align_after.ref_length,
            "인식_어절수": wer_align_after.hyp_length,
        }
        result.common_errors_after = extract_common_errors(cer_align_after, top_n_errors)

        # 개선율 (양수면 개선, 음수면 악화)
        if result.cer_before > 0:
            result.cer_improvement = round(
                (result.cer_before - result.cer_after) / result.cer_before * 100, 2
            )
        if result.wer_before > 0:
            result.wer_improvement = round(
                (result.wer_before - result.wer_after) / result.wer_before * 100, 2
            )
    else:
        # 교정 엔진 없으면 교정 후 = 교정 전 동일
        result.cer_after = result.cer_before
        result.wer_after = result.wer_before
        result.cer_detail_after = result.cer_detail_before.copy()
        result.wer_detail_after = result.wer_detail_before.copy()
        result.common_errors_after = result.common_errors_before[:]

    return result


# ──────────────────────────────────────────────────────────────────────
# 전체 평가 파이프라인
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FullEvaluationReport:
    """전체 평가 리포트."""
    timestamp: str
    dataset_path: str
    dict_path: str
    type_results: list[TypeEvaluationResult] = field(default_factory=list)
    # 요약 지표
    avg_cer_before: float = 0.0
    avg_wer_before: float = 0.0
    avg_cer_after: float = 0.0
    avg_wer_after: float = 0.0
    avg_cer_improvement: float = 0.0
    avg_wer_improvement: float = 0.0
    total_types_evaluated: int = 0
    total_types_failed: int = 0


def _discover_types(dataset_path: Path) -> list[int]:
    """데이터셋 경로에서 사용 가능한 type 번호를 탐색."""
    types = []
    if not dataset_path.exists():
        return types
    for d in sorted(dataset_path.iterdir()):
        if d.is_dir() and d.name.startswith("type"):
            try:
                num = int(d.name.replace("type", ""))
                # 필수 파일 존재 확인
                gt = d / f"type{num}_full_transcript.txt"
                stt = d / f"donkey_type{num}.txt"
                if gt.exists() and stt.exists():
                    types.append(num)
            except ValueError:
                continue
    return types


def run_full_evaluation(
    dataset_path: Path | None = None,
    dict_path: Path | None = None,
    type_nums: list[int] | None = None,
    top_n_errors: int = 20,
    save_json: bool = True,
) -> FullEvaluationReport:
    """전체 평가를 실행하고 리포트를 생성.

    Args:
        dataset_path: 데이터셋 루트 경로 (기본: _DEFAULT_DATASET_PATH)
        dict_path: 의료 사전 경로 (기본: _DEFAULT_DICT_PATH)
        type_nums: 평가할 type 번호 리스트 (None이면 자동 탐색)
        top_n_errors: 상위 오류 개수
        save_json: True면 결과를 JSON 파일로 저장

    Returns:
        FullEvaluationReport
    """
    ds_path = dataset_path or _DEFAULT_DATASET_PATH
    dp = dict_path or _DEFAULT_DICT_PATH

    # 엔진을 한 번만 로딩해서 재사용
    engine = _get_correction_engine(dp)

    if type_nums is None:
        type_nums = _discover_types(ds_path)

    if not type_nums:
        logger.warning("평가 가능한 type이 없습니다. 데이터셋 경로: %s", ds_path)

    now = datetime.now(timezone.utc)
    report = FullEvaluationReport(
        timestamp=now.isoformat(),
        dataset_path=str(ds_path),
        dict_path=str(dp),
    )

    for t in type_nums:
        logger.info("평가 중: type%d", t)
        tr = evaluate_single_type(
            type_num=t,
            dataset_path=ds_path,
            dict_path=dp,
            engine=engine,
            top_n_errors=top_n_errors,
        )
        report.type_results.append(tr)

    # 요약 계산 (오류 없는 것만)
    valid = [r for r in report.type_results if r.error is None]
    report.total_types_evaluated = len(valid)
    report.total_types_failed = len(report.type_results) - len(valid)

    if valid:
        report.avg_cer_before = round(sum(r.cer_before for r in valid) / len(valid), 2)
        report.avg_wer_before = round(sum(r.wer_before for r in valid) / len(valid), 2)
        report.avg_cer_after = round(sum(r.cer_after for r in valid) / len(valid), 2)
        report.avg_wer_after = round(sum(r.wer_after for r in valid) / len(valid), 2)
        report.avg_cer_improvement = round(
            sum(r.cer_improvement for r in valid) / len(valid), 2
        )
        report.avg_wer_improvement = round(
            sum(r.wer_improvement for r in valid) / len(valid), 2
        )

    # JSON 저장
    if save_json:
        _save_report_json(report)

    return report


def _save_report_json(report: FullEvaluationReport) -> Path:
    """평가 결과를 JSON 파일로 저장."""
    results_dir = _DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"eval_{ts}.json"

    data = _report_to_dict(report)
    filepath.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("평가 결과 저장: %s", filepath)

    # latest 심볼릭 링크 / 복사 (latest.json)
    latest_path = results_dir / "latest.json"
    latest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return filepath


def _report_to_dict(report: FullEvaluationReport) -> dict:
    """리포트를 직렬화 가능한 dict로 변환."""
    d = {
        "timestamp": report.timestamp,
        "dataset_path": report.dataset_path,
        "dict_path": report.dict_path,
        "summary": {
            "평가_타입_수": report.total_types_evaluated,
            "실패_타입_수": report.total_types_failed,
            "평균_CER_교정전": report.avg_cer_before,
            "평균_WER_교정전": report.avg_wer_before,
            "평균_CER_교정후": report.avg_cer_after,
            "평균_WER_교정후": report.avg_wer_after,
            "평균_CER_개선율": report.avg_cer_improvement,
            "평균_WER_개선율": report.avg_wer_improvement,
        },
        "types": [],
    }
    for tr in report.type_results:
        td = {
            "type": tr.type_num,
            "error": tr.error,
            "교정전": {
                "CER": tr.cer_before,
                "WER": tr.wer_before,
                "CER_상세": tr.cer_detail_before,
                "WER_상세": tr.wer_detail_before,
            },
            "교정후": {
                "CER": tr.cer_after,
                "WER": tr.wer_after,
                "CER_상세": tr.cer_detail_after,
                "WER_상세": tr.wer_detail_after,
            },
            "개선율": {
                "CER_개선율": tr.cer_improvement,
                "WER_개선율": tr.wer_improvement,
            },
            "주요_오류_교정전": tr.common_errors_before[:10],
            "주요_오류_교정후": tr.common_errors_after[:10],
            "correction_applied": tr.correction_applied,
            "정답_길이": tr.ground_truth_length,
            "STT_길이": tr.stt_text_length,
            "교정후_길이": tr.corrected_text_length,
        }
        d["types"].append(td)
    return d


# ──────────────────────────────────────────────────────────────────────
# 회귀 테스트
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """회귀 테스트 결과."""
    passed: bool
    details: list[dict] = field(default_factory=list)
    baseline_timestamp: str = ""
    current_timestamp: str = ""
    message: str = ""


def run_regression_test(
    baseline_path: Path | None = None,
    dataset_path: Path | None = None,
    dict_path: Path | None = None,
    threshold_pct: float = 1.0,
) -> RegressionResult:
    """회귀 테스트: 기준선 대비 CER이 threshold_pct% 이상 악화된 type을 감지.

    Args:
        baseline_path: 기준선 JSON 경로 (None이면 latest.json)
        dataset_path: 데이터셋 경로
        dict_path: 사전 경로
        threshold_pct: CER 악화 허용 임계값 (%, 기본 1.0)

    Returns:
        RegressionResult: pass/fail 및 상세 내역
    """
    results_dir = _DEFAULT_RESULTS_DIR
    if baseline_path is None:
        baseline_path = results_dir / "latest.json"

    if not baseline_path.exists():
        return RegressionResult(
            passed=True,
            message="기준선 파일이 없습니다. 첫 번째 실행으로 간주합니다.",
        )

    # 기준선 로딩
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_type: dict[int, float] = {}
    for td in baseline_data.get("types", []):
        if td.get("error") is None:
            baseline_by_type[td["type"]] = td["교정후"]["CER"]

    # 현재 평가 실행
    current_report = run_full_evaluation(
        dataset_path=dataset_path,
        dict_path=dict_path,
        save_json=False,
    )

    regressions = []
    for tr in current_report.type_results:
        if tr.error is not None:
            continue
        baseline_cer = baseline_by_type.get(tr.type_num)
        if baseline_cer is None:
            continue
        current_cer = tr.cer_after
        diff = current_cer - baseline_cer
        if diff > threshold_pct:
            regressions.append({
                "type": tr.type_num,
                "기준선_CER": baseline_cer,
                "현재_CER": current_cer,
                "차이": round(diff, 2),
                "상태": "악화",
            })
        else:
            regressions.append({
                "type": tr.type_num,
                "기준선_CER": baseline_cer,
                "현재_CER": current_cer,
                "차이": round(diff, 2),
                "상태": "개선" if diff < -threshold_pct else "유지",
            })

    failed_types = [r for r in regressions if r["상태"] == "악화"]
    passed = len(failed_types) == 0

    if passed:
        msg = f"회귀 테스트 통과: 전체 {len(regressions)}개 type 중 악화 없음 (임계값: {threshold_pct}%)"
    else:
        type_nums_str = ", ".join(str(r["type"]) for r in failed_types)
        msg = (
            f"회귀 테스트 실패: {len(failed_types)}개 type 악화 "
            f"(type {type_nums_str}, 임계값: {threshold_pct}%)"
        )

    return RegressionResult(
        passed=passed,
        details=regressions,
        baseline_timestamp=baseline_data.get("timestamp", ""),
        current_timestamp=current_report.timestamp,
        message=msg,
    )


# ──────────────────────────────────────────────────────────────────────
# 평가 이력 관리
# ──────────────────────────────────────────────────────────────────────

def get_evaluation_history(limit: int = 20) -> list[dict]:
    """저장된 평가 결과 이력 반환 (최신순)."""
    results_dir = _DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        return []

    files = sorted(
        [f for f in results_dir.glob("eval_*.json")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    history = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            history.append({
                "파일명": f.name,
                "timestamp": data.get("timestamp", ""),
                "평균_CER_교정전": summary.get("평균_CER_교정전", 0),
                "평균_CER_교정후": summary.get("평균_CER_교정후", 0),
                "평균_CER_개선율": summary.get("평균_CER_개선율", 0),
                "평가_타입_수": summary.get("평가_타입_수", 0),
            })
        except Exception:
            continue

    return history


def compare_evaluations(file1: str, file2: str) -> dict:
    """두 평가 결과를 비교.

    Args:
        file1, file2: 파일 이름 (eval_YYYYMMDD_HHMMSS.json) 또는 전체 경로

    Returns:
        비교 결과 dict
    """
    results_dir = _DEFAULT_RESULTS_DIR

    def _resolve(name: str) -> Path:
        p = Path(name)
        if p.exists():
            return p
        return results_dir / name

    p1 = _resolve(file1)
    p2 = _resolve(file2)

    if not p1.exists():
        return {"error": f"파일을 찾을 수 없습니다: {file1}"}
    if not p2.exists():
        return {"error": f"파일을 찾을 수 없습니다: {file2}"}

    d1 = json.loads(p1.read_text(encoding="utf-8"))
    d2 = json.loads(p2.read_text(encoding="utf-8"))

    # type별 비교
    types1 = {t["type"]: t for t in d1.get("types", []) if t.get("error") is None}
    types2 = {t["type"]: t for t in d2.get("types", []) if t.get("error") is None}

    all_types = sorted(set(types1.keys()) | set(types2.keys()))

    comparison = {
        "파일1": {"이름": p1.name, "timestamp": d1.get("timestamp", "")},
        "파일2": {"이름": p2.name, "timestamp": d2.get("timestamp", "")},
        "요약1": d1.get("summary", {}),
        "요약2": d2.get("summary", {}),
        "type별_비교": [],
    }

    for t in all_types:
        t1 = types1.get(t)
        t2 = types2.get(t)
        entry = {"type": t}
        if t1 and t2:
            cer1 = t1["교정후"]["CER"]
            cer2 = t2["교정후"]["CER"]
            entry["파일1_CER"] = cer1
            entry["파일2_CER"] = cer2
            entry["차이"] = round(cer2 - cer1, 2)
            entry["상태"] = "개선" if cer2 < cer1 else ("악화" if cer2 > cer1 else "동일")
        else:
            entry["파일1_CER"] = t1["교정후"]["CER"] if t1 else None
            entry["파일2_CER"] = t2["교정후"]["CER"] if t2 else None
            entry["상태"] = "데이터_부족"
        comparison["type별_비교"].append(entry)

    return comparison


# ──────────────────────────────────────────────────────────────────────
# 콘솔 출력 (한글)
# ──────────────────────────────────────────────────────────────────────

def print_report(report: FullEvaluationReport) -> None:
    """평가 리포트를 콘솔에 출력."""
    sep = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("  의료 STT 품질 평가 리포트")
    print(f"  실행 시각: {report.timestamp}")
    print(f"  데이터셋: {report.dataset_path}")
    print(f"{sep}\n")

    # 요약 테이블
    print(f"  [요약]")
    print(f"  평가 완료: {report.total_types_evaluated}개 type  |  "
          f"실패: {report.total_types_failed}개")
    print(f"{thin}")
    print(f"  {'지표':<20} {'교정 전':>10} {'교정 후':>10} {'개선율':>10}")
    print(f"  {thin}")
    print(f"  {'평균 CER (%)':<20} {report.avg_cer_before:>10.2f} "
          f"{report.avg_cer_after:>10.2f} {report.avg_cer_improvement:>9.1f}%")
    print(f"  {'평균 WER (%)':<20} {report.avg_wer_before:>10.2f} "
          f"{report.avg_wer_after:>10.2f} {report.avg_wer_improvement:>9.1f}%")
    print()

    # type별 상세
    print(f"  [Type별 상세 결과]")
    print(f"  {thin}")
    header = (
        f"  {'Type':>5} | {'CER 전':>8} | {'CER 후':>8} | "
        f"{'개선율':>7} | {'WER 전':>8} | {'WER 후':>8} | "
        f"{'삽입':>5} | {'삭제':>5} | {'치환':>5} | {'상태'}"
    )
    print(header)
    print(f"  {thin}")

    for tr in report.type_results:
        if tr.error:
            print(f"  type{tr.type_num:>2} | {'오류':>8} | {tr.error}")
            continue

        status = "OK"
        if tr.cer_improvement < 0:
            status = "악화"
        elif tr.cer_improvement > 0:
            status = "개선"
        else:
            status = "동일"

        cer_detail = tr.cer_detail_after or tr.cer_detail_before
        print(
            f"  type{tr.type_num:>2} | "
            f"{tr.cer_before:>7.2f}% | {tr.cer_after:>7.2f}% | "
            f"{tr.cer_improvement:>6.1f}% | "
            f"{tr.wer_before:>7.2f}% | {tr.wer_after:>7.2f}% | "
            f"{cer_detail.get('삽입', 0):>5} | "
            f"{cer_detail.get('삭제', 0):>5} | "
            f"{cer_detail.get('치환', 0):>5} | "
            f"{status}"
        )

    print()

    # 주요 오류 (전체 합산)
    all_errors: Counter = Counter()
    for tr in report.type_results:
        if tr.error:
            continue
        for err in tr.common_errors_after:
            key = (err["유형"], err["정답"], err["인식"])
            all_errors[key] += err["횟수"]

    if all_errors:
        print(f"  [전체 주요 오류 Top 15 (교정 후)]")
        print(f"  {thin}")
        print(f"  {'유형':<6} | {'정답':<10} | {'인식':<10} | {'횟수':>6}")
        print(f"  {thin}")
        for (etype, ref, hyp), cnt in all_errors.most_common(15):
            ref_display = ref if ref else "(없음)"
            hyp_display = hyp if hyp else "(없음)"
            print(f"  {etype:<6} | {ref_display:<10} | {hyp_display:<10} | {cnt:>6}")

    print(f"\n{sep}\n")


def print_regression_result(result: RegressionResult) -> None:
    """회귀 테스트 결과를 콘솔에 출력."""
    sep = "=" * 72
    thin = "-" * 72
    status_str = "통과" if result.passed else "실패"
    status_mark = "[PASS]" if result.passed else "[FAIL]"

    print(f"\n{sep}")
    print(f"  회귀 테스트 결과: {status_mark} {status_str}")
    print(f"  {result.message}")
    print(f"{sep}")

    if result.details:
        print(f"\n  {'Type':>5} | {'기준선 CER':>10} | {'현재 CER':>10} | {'차이':>7} | {'상태'}")
        print(f"  {thin}")
        for d in result.details:
            marker = ""
            if d["상태"] == "악화":
                marker = " *** "
            print(
                f"  type{d['type']:>2} | "
                f"{d['기준선_CER']:>9.2f}% | {d['현재_CER']:>9.2f}% | "
                f"{d['차이']:>+6.2f}% | {d['상태']}{marker}"
            )
    print()


# ──────────────────────────────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("\n의료 STT WER/CER 자동 평가 시스템 시작...\n")

    # 인자 파싱 (간단 모드)
    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "regression":
        # 회귀 테스트
        result = run_regression_test()
        print_regression_result(result)
        sys.exit(0 if result.passed else 1)

    elif mode == "history":
        # 이력 조회
        history = get_evaluation_history()
        if not history:
            print("저장된 평가 이력이 없습니다.")
        else:
            print(f"{'파일명':<30} | {'CER 전':>8} | {'CER 후':>8} | {'개선율':>7} | {'타입수':>5}")
            print("-" * 72)
            for h in history:
                print(
                    f"{h['파일명']:<30} | "
                    f"{h['평균_CER_교정전']:>7.2f}% | "
                    f"{h['평균_CER_교정후']:>7.2f}% | "
                    f"{h['평균_CER_개선율']:>6.1f}% | "
                    f"{h['평가_타입_수']:>5}"
                )

    elif mode.startswith("type"):
        # 단일 type 평가
        try:
            type_num = int(mode.replace("type", ""))
        except ValueError:
            print(f"잘못된 type 번호: {mode}")
            sys.exit(1)
        tr = evaluate_single_type(type_num)
        if tr.error:
            print(f"type{type_num} 평가 실패: {tr.error}")
            sys.exit(1)
        # 간이 출력
        print(f"type{type_num} 평가 결과:")
        print(f"  CER: {tr.cer_before:.2f}% → {tr.cer_after:.2f}% (개선: {tr.cer_improvement:.1f}%)")
        print(f"  WER: {tr.wer_before:.2f}% → {tr.wer_after:.2f}% (개선: {tr.wer_improvement:.1f}%)")

    else:
        # 전체 평가
        report = run_full_evaluation()
        print_report(report)

        # 회귀 테스트도 함께 수행
        print("\n회귀 테스트 수행 중...\n")
        regression = run_regression_test()
        print_regression_result(regression)
