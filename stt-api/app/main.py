"""STT API Server - Standalone Speech-to-Text API."""

import asyncio
import logging
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings

# ──────────────────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────────────────

_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, _settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_max_concurrent_transcribe = max(1, int(_settings.max_concurrent_transcribe))
_inflight_transcribe = 0
_inflight_lock = asyncio.Lock()

# ──────────────────────────────────────────────────────────────────────
# API 버전 및 메타데이터
# ──────────────────────────────────────────────────────────────────────

API_VERSION = "0.3.0"
API_BUILD_INFO = {
    "name": "Donkey-AI STT API",
    "version": API_VERSION,
    "description": "의료 음성 전사 API (Faster-Whisper 오픈소스 기반)",
    "features": [
        "Faster-Whisper 로컬 STT",
        "의료 용어 교정",
        "화자 분리 및 교정",
        "SOAP 노트 생성",
        "STT 결과 뷰어",
    ],
}

# ──────────────────────────────────────────────────────────────────────
# FastAPI 앱 초기화
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="STT API",
    description="Standalone Speech-to-Text API (Faster-Whisper 오픈소스)",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────
# 요청 로깅 미들웨어 (타임스탬프, 엔드포인트, 소요 시간)
# ──────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """모든 요청에 대해 로깅 + 응답 헤더에 처리 시간 추가"""
    start_time = time.time()
    method = request.method
    path = request.url.path

    # 정적 파일 요청은 로깅 생략 (노이즈 방지)
    is_static = path.startswith("/static")

    if not is_static:
        logger.info("요청 시작: %s %s", method, path)

    try:
        response = await call_next(request)
    except Exception as e:
        # 미들웨어 레벨에서 잡히지 않은 예외 처리
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "처리되지 않은 오류: %s %s (%.1fms) - %s",
            method, path, duration_ms, str(e),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "서버 내부 오류가 발생했습니다",
                "detail": "요청을 처리하는 중 예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "path": path,
            },
            headers={"X-Response-Time-ms": f"{duration_ms:.1f}"},
        )

    duration_ms = (time.time() - start_time) * 1000

    # 응답 헤더에 처리 시간 추가
    response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"

    if not is_static:
        logger.info(
            "요청 완료: %s %s → %d (%.1fms)",
            method, path, response.status_code, duration_ms,
        )

    return response


# ──────────────────────────────────────────────────────────────────────
# 컴포넌트 로드 (Graceful Degradation)
# 각 컴포넌트가 실패해도 다른 컴포넌트는 정상 동작
# ──────────────────────────────────────────────────────────────────────

# 컴포넌트 상태 추적
_component_status = {
    "stt_engine": {"loaded": False, "error": None},
    "medical_dict": {"loaded": False, "error": None},
    "viewer": {"loaded": False, "error": None},
    "static_files": {"loaded": False, "error": None},
}

# 1) 의료 사전 엔진 초기화 + 라우터 등록
try:
    from app.medterm.engine import init_engine
    from app.medterm.router import router as medterm_router

    settings = get_settings()
    dict_path = Path(settings.medical_dict_path)
    # OpenAI API 키 (화자교정 GPT 검증용)
    openai_key = settings.openai_api_key if settings.enable_speaker_gpt else None
    # Tier 1 + Tier 2 (KOSTOM 자동탐지) 초기화
    ref_db_path = dict_path.parent / "kostom_reference.json"
    init_engine(dict_path, ref_db_path=ref_db_path, openai_api_key=openai_key)
    app.include_router(medterm_router, prefix="/api/medical-dict", tags=["medical-dictionary"])
    _component_status["medical_dict"]["loaded"] = True
    logger.info("의료 사전 모듈 활성화: %s", dict_path)
except Exception as exc:
    _component_status["medical_dict"]["error"] = str(exc)
    logger.warning("의료 사전 모듈 로드 실패 (폴백 모드): %s", exc)

# 2) 뷰어 라우터 등록
try:
    from app.viewer_router import router as viewer_router
    app.include_router(viewer_router, prefix="/api/viewer", tags=["viewer"])
    _component_status["viewer"]["loaded"] = True
    logger.info("STT 뷰어 모듈 활성화")
except Exception as exc:
    _component_status["viewer"]["error"] = str(exc)
    logger.warning("뷰어 모듈 로드 실패: %s", exc)

# 2-1) 평가(WER/CER) 라우터 등록
try:
    from app.evaluation_router import router as eval_router
    app.include_router(eval_router, prefix="/api/evaluation", tags=["evaluation"])
    _component_status["evaluation"] = {"loaded": True, "error": None}
    logger.info("WER/CER 평가 모듈 활성화")
except Exception as exc:
    _component_status["evaluation"] = {"loaded": False, "error": str(exc)}
    logger.warning("평가 모듈 로드 실패: %s", exc)

# 3) STT 엔진 (transcribe 관련)
try:
    from app.services.audio import download_audio, ensure_wav_16k_mono
    from app.services.pipeline import transcribe_with_diarization
    _component_status["stt_engine"]["loaded"] = True
    logger.info("STT 엔진 모듈 활성화")
except Exception as exc:
    _component_status["stt_engine"]["error"] = str(exc)
    logger.warning("STT 엔진 모듈 로드 실패: %s", exc)

# 4) 정적 파일 서빙 + 뷰어 페이지
_static_dir = Path(__file__).parent.parent / "static"
try:
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
        _component_status["static_files"]["loaded"] = True
        logger.info("정적 파일 서빙 활성화: %s", _static_dir)
    else:
        _component_status["static_files"]["error"] = f"디렉토리 없음: {_static_dir}"
        logger.warning("정적 파일 디렉토리가 존재하지 않습니다: %s", _static_dir)
except Exception as exc:
    _component_status["static_files"]["error"] = str(exc)
    logger.warning("정적 파일 서빙 설정 실패: %s", exc)


@app.get("/viewer")
def viewer_page():
    """STT 결과 뷰어 페이지."""
    try:
        html_path = _static_dir / "viewer.html"
        if html_path.exists():
            return FileResponse(str(html_path))
        raise HTTPException(404, "뷰어 페이지를 찾을 수 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("뷰어 페이지 로드 실패: %s", e)
        raise HTTPException(500, "뷰어 페이지를 로드하는 중 오류가 발생했습니다")


# ──────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ──────────────────────────────────────────────────────────────────────

class TranscribeUrlRequest(BaseModel):
    """URL로 음성 전사 요청."""
    url: str = Field(..., description="오디오 파일 URL")
    language: str = Field(default="ko", description="언어 코드 (ko, en 등)")


class SegmentResponse(BaseModel):
    """전사 세그먼트."""
    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscribeResponse(BaseModel):
    """전사 결과."""
    segments: list[SegmentResponse]
    full_text: str


# ──────────────────────────────────────────────────────────────────────
# 헬스체크 및 버전 엔드포인트
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """확장된 헬스체크 — 각 컴포넌트 상태 포함.

    Returns:
        status: "healthy" (전체 정상) / "degraded" (일부 실패) / "unhealthy" (핵심 실패)
        components: 각 컴포넌트별 로드 상태 및 오류 정보
    """
    try:
        components = {}
        for name, status in _component_status.items():
            components[name] = {
                "status": "ok" if status["loaded"] else "unavailable",
                "error": status["error"],
            }

        # 전체 상태 판단
        all_loaded = all(s["loaded"] for s in _component_status.values())
        # STT 엔진이 핵심 컴포넌트 — 이것이 실패하면 unhealthy
        stt_ok = _component_status["stt_engine"]["loaded"]

        if all_loaded:
            overall_status = "healthy"
        elif stt_ok:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "version": API_VERSION,
            "components": components,
            "runtime": {
                "inflight_transcribe": _inflight_transcribe,
                "max_concurrent_transcribe": _max_concurrent_transcribe,
            },
        }
    except Exception as e:
        logger.error("헬스체크 실패: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"헬스체크 수행 중 오류 발생: {e}",
            },
        )


@app.get("/api/version")
def api_version():
    """API 버전 및 빌드 정보 반환."""
    try:
        settings = get_settings()
        return {
            **API_BUILD_INFO,
            "config": {
                "whisper_model": settings.faster_whisper_model,
                "compute_type": settings.faster_whisper_compute_type,
                "beam_size": settings.faster_whisper_beam_size,
                "language": settings.default_language,
                "postprocessing": settings.enable_postprocessing,
                "filler_removal": settings.enable_filler_removal,
                "number_normalization": settings.enable_number_normalization,
                "hallucination_removal": settings.enable_hallucination_removal,
                "soap_include_summary": settings.soap_include_summary,
                "diarization": settings.enable_diarization,
                "speaker_gpt": settings.enable_speaker_gpt,
                "log_level": settings.log_level,
                "max_concurrent_transcribe": settings.max_concurrent_transcribe,
                "speaker_alignment_mode": settings.speaker_alignment_mode,
            },
            "components": {
                name: "loaded" if status["loaded"] else "not_loaded"
                for name, status in _component_status.items()
            },
        }
    except Exception as e:
        logger.error("버전 정보 조회 실패: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"버전 정보를 가져오는 중 오류가 발생했습니다: {e}"},
        )


# ──────────────────────────────────────────────────────────────────────
# 전사 엔드포인트
# ──────────────────────────────────────────────────────────────────────

def _check_stt_engine():
    """STT 엔진이 로드되었는지 확인. 미로드 시 적절한 오류 반환."""
    if not _component_status["stt_engine"]["loaded"]:
        raise HTTPException(
            503,
            "STT 엔진이 로드되지 않았습니다. 서버 로그를 확인하세요. "
            f"오류: {_component_status['stt_engine']['error']}",
        )


@asynccontextmanager
async def _transcribe_slot_guard():
    """동시 전사 실행 수 제한. 초과 시 429 반환."""
    global _inflight_transcribe
    async with _inflight_lock:
        if _inflight_transcribe >= _max_concurrent_transcribe:
            raise HTTPException(
                429,
                f"동시 전사 요청이 많습니다. 잠시 후 다시 시도해주세요. "
                f"(limit={_max_concurrent_transcribe})",
                headers={"Retry-After": "2"},
            )
        _inflight_transcribe += 1
    try:
        yield
    finally:
        async with _inflight_lock:
            _inflight_transcribe = max(0, _inflight_transcribe - 1)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_url(req: TranscribeUrlRequest):
    """오디오 URL을 받아 전사 결과 반환.

    지원 형식: wav, mp3, m4a 등 (ffmpeg 지원 형식)
    """
    _check_stt_engine()
    async with _transcribe_slot_guard():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "audio"

            try:
                await download_audio(req.url, src)
            except Exception as e:
                logger.warning("오디오 다운로드 실패: %s (URL: %s)", e, req.url)
                raise HTTPException(
                    400,
                    f"오디오 파일을 다운로드할 수 없습니다: {e}",
                ) from e

            try:
                wav_path = ensure_wav_16k_mono(src)
            except Exception as e:
                logger.warning("오디오 변환 실패: %s", e)
                raise HTTPException(
                    400,
                    f"오디오 파일을 변환할 수 없습니다. WAV 16kHz mono 형식이 필요합니다: {e}",
                ) from e

            try:
                segments = await transcribe_with_diarization(wav_path, language=req.language)
            except ValueError as e:
                logger.warning("전사 파라미터 오류: %s", e)
                raise HTTPException(400, f"전사 요청 파라미터 오류: {e}") from e
            except Exception as e:
                logger.error("전사 실패: %s\n%s", e, traceback.format_exc())
                raise HTTPException(
                    500,
                    f"음성 전사 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요: {e}",
                ) from e

    out = [
        SegmentResponse(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
        )
        for s in segments
    ]
    full_text = " ".join(s["text"] for s in segments).strip()
    return TranscribeResponse(segments=out, full_text=full_text)


@app.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form(default="ko"),
):
    """오디오 파일 업로드로 전사."""
    _check_stt_engine()
    async with _transcribe_slot_guard():
        suffix = Path(file.filename or "audio").suffix or ".bin"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()
                tmp_path = Path(tmp.name)
        except Exception as e:
            logger.error("업로드 파일 임시 저장 실패: %s", e)
            raise HTTPException(
                500,
                f"업로드된 파일을 처리할 수 없습니다: {e}",
            ) from e

        try:
            wav_path = ensure_wav_16k_mono(tmp_path)
            segments = await transcribe_with_diarization(wav_path, language=language)
        except ValueError as e:
            tmp_path.unlink(missing_ok=True)
            logger.warning("전사 파라미터 오류: %s", e)
            raise HTTPException(400, f"전사 요청 파라미터 오류: {e}") from e
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            logger.error("전사 실패: %s\n%s", e, traceback.format_exc())
            raise HTTPException(
                500,
                f"음성 전사 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요: {e}",
            ) from e
        finally:
            tmp_path.unlink(missing_ok=True)

    out = [
        SegmentResponse(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
        )
        for s in segments
    ]
    full_text = " ".join(s["text"] for s in segments).strip()
    return TranscribeResponse(segments=out, full_text=full_text)


# ──────────────────────────────────────────────────────────────────────
# 서버 직접 실행
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
