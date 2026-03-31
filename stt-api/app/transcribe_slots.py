"""전사 동시 실행 슬롯 — `/transcribe`, 뷰어 등 모든 진입점이 동일 세마포어(대기 큐)를 공유."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.config import get_settings

_settings = get_settings()
_max_concurrent_transcribe = max(1, int(_settings.max_concurrent_transcribe))
_transcribe_semaphore = asyncio.Semaphore(_max_concurrent_transcribe)
_inflight_transcribe = 0
_inflight_lock = asyncio.Lock()


def get_max_concurrent_transcribe() -> int:
    return _max_concurrent_transcribe


def get_inflight_transcribe() -> int:
    return _inflight_transcribe


@asynccontextmanager
async def transcribe_slot_guard():
    """동시 전사 수 제한. 슬롯이 없으면 **연결을 끊지 않고** acquire에서 대기(선입선출).

    별도의 메시지 큐(Redis 등)는 아니지만, asyncio 세마포어 대기열이 곧 큐 역할을 한다.
    """
    global _inflight_transcribe
    await _transcribe_semaphore.acquire()
    async with _inflight_lock:
        _inflight_transcribe += 1
    try:
        yield
    finally:
        async with _inflight_lock:
            _inflight_transcribe = max(0, _inflight_transcribe - 1)
        _transcribe_semaphore.release()
