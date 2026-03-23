"""WER/CER 평가 API 라우터.

사용법:
    main.py에서 다음과 같이 등록:

        from app.evaluation_router import router as eval_router
        app.include_router(eval_router, prefix="/api/evaluation", tags=["evaluation"])
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()


def _import_eval():
    """wer_evaluation 모듈을 지연 임포트 (순환 참조 방지)."""
    try:
        import wer_evaluation
        return wer_evaluation
    except ImportError:
        # stt-api 루트가 sys.path에 없을 수 있음
        import sys
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        import wer_evaluation
        return wer_evaluation


# ──────────────────────────────────────────────────────────────────────
# GET /api/evaluation/run — 전체 평가 실행
# ──────────────────────────────────────────────────────────────────────

@router.get("/run")
def run_evaluation(
    save: bool = Query(default=True, description="결과를 JSON 파일로 저장"),
    top_n: int = Query(default=20, description="상위 오류 개수", ge=1, le=100),
):
    """전체 type에 대한 CER/WER 평가를 실행.

    Returns:
        평가 리포트 (요약 + type별 상세)
    """
    try:
        ev = _import_eval()
        report = ev.run_full_evaluation(
            save_json=save,
            top_n_errors=top_n,
        )
        return ev._report_to_dict(report)
    except Exception as e:
        logger.exception("평가 실행 실패")
        raise HTTPException(500, f"평가 실행 중 오류가 발생했습니다: {e}") from e


# ──────────────────────────────────────────────────────────────────────
# GET /api/evaluation/type/{type_num} — 단일 type 평가
# ──────────────────────────────────────────────────────────────────────

@router.get("/type/{type_num}")
def evaluate_type(
    type_num: int,
    top_n: int = Query(default=20, description="상위 오류 개수", ge=1, le=100),
):
    """특정 type에 대한 CER/WER 평가를 실행.

    Args:
        type_num: type 번호 (1~16)
    """
    try:
        ev = _import_eval()
        result = ev.evaluate_single_type(
            type_num=type_num,
            top_n_errors=top_n,
        )
        if result.error:
            raise HTTPException(404, f"type{type_num} 평가 실패: {result.error}")

        return {
            "type": result.type_num,
            "교정전": {
                "CER": result.cer_before,
                "WER": result.wer_before,
                "CER_상세": result.cer_detail_before,
                "WER_상세": result.wer_detail_before,
            },
            "교정후": {
                "CER": result.cer_after,
                "WER": result.wer_after,
                "CER_상세": result.cer_detail_after,
                "WER_상세": result.wer_detail_after,
            },
            "개선율": {
                "CER_개선율": result.cer_improvement,
                "WER_개선율": result.wer_improvement,
            },
            "주요_오류_교정전": result.common_errors_before[:top_n],
            "주요_오류_교정후": result.common_errors_after[:top_n],
            "correction_applied": result.correction_applied,
            "정답_길이": result.ground_truth_length,
            "STT_길이": result.stt_text_length,
            "교정후_길이": result.corrected_text_length,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("type%d 평가 실패", type_num)
        raise HTTPException(500, f"평가 실행 중 오류가 발생했습니다: {e}") from e


# ──────────────────────────────────────────────────────────────────────
# GET /api/evaluation/history — 평가 이력
# ──────────────────────────────────────────────────────────────────────

@router.get("/history")
def evaluation_history(
    limit: int = Query(default=20, description="최대 반환 개수", ge=1, le=100),
):
    """저장된 평가 결과 이력을 반환 (최신순)."""
    try:
        ev = _import_eval()
        history = ev.get_evaluation_history(limit=limit)
        return {"이력": history, "총_개수": len(history)}
    except Exception as e:
        logger.exception("평가 이력 조회 실패")
        raise HTTPException(500, f"이력 조회 중 오류가 발생했습니다: {e}") from e


# ──────────────────────────────────────────────────────────────────────
# GET /api/evaluation/compare — 두 평가 결과 비교
# ──────────────────────────────────────────────────────────────────────

@router.get("/compare")
def compare_evaluations(
    file1: str = Query(..., description="첫 번째 평가 결과 파일명"),
    file2: str = Query(..., description="두 번째 평가 결과 파일명"),
):
    """두 평가 결과를 비교하여 type별 CER 변화를 보여준다.

    Args:
        file1: 기준 평가 결과 파일명 (예: eval_20250321_120000.json)
        file2: 비교 대상 파일명
    """
    try:
        ev = _import_eval()
        result = ev.compare_evaluations(file1, file2)
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("평가 비교 실패")
        raise HTTPException(500, f"비교 중 오류가 발생했습니다: {e}") from e


# ──────────────────────────────────────────────────────────────────────
# GET /api/evaluation/regression — 회귀 테스트
# ──────────────────────────────────────────────────────────────────────

@router.get("/regression")
def regression_test(
    threshold: float = Query(
        default=1.0, description="CER 악화 허용 임계값 (%)", ge=0.0, le=100.0,
    ),
):
    """기준선 대비 회귀 테스트를 수행.

    기준선(latest.json) 대비 각 type의 CER이 threshold% 이상 악화되면 실패.
    """
    try:
        ev = _import_eval()
        result = ev.run_regression_test(threshold_pct=threshold)
        return {
            "passed": result.passed,
            "message": result.message,
            "baseline_timestamp": result.baseline_timestamp,
            "current_timestamp": result.current_timestamp,
            "details": result.details,
        }
    except Exception as e:
        logger.exception("회귀 테스트 실패")
        raise HTTPException(500, f"회귀 테스트 중 오류가 발생했습니다: {e}") from e
