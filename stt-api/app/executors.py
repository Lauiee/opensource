"""전사·화자분리·Clova 등 블로킹 CPU/IO 작업용 스레드 풀.

async 엔드포인트에서 `run_in_executor(None, …)`만 쓰면 기본 스레드 풀과 경쟁해
다른 작업이 굶을 수 있어, 전사 전용 Executor로 묶는다.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None


def get_transcribe_executor() -> ThreadPoolExecutor:
    """지연 생성. `MAX_CONCURRENT_TRANSCRIBE`·동시 전사+분리 패턴을 감안해 worker 수 결정."""
    global _executor
    if _executor is None:
        raw = os.environ.get("STT_THREAD_POOL_WORKERS", "").strip()
        if raw:
            n = max(2, int(raw))
        else:
            from app.config import get_settings

            concurrent = max(1, get_settings().max_concurrent_transcribe)
            # 요청당 전사+분리 동시 2 + 여유(복구 등)
            n = max(6, min(32, concurrent * 4))
        _executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="stt-transcribe")
        logger.info("전사 전용 ThreadPoolExecutor 시작 (max_workers=%s)", n)
    return _executor


def shutdown_transcribe_executor(*, wait: bool = True) -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait, cancel_futures=False)
        _executor = None
        logger.info("전사 전용 ThreadPoolExecutor 종료")
