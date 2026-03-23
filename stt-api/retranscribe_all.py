"""모든 test_set 타입을 v2.0 파이프라인으로 재전사.

기존 donkey_typeN.txt를 백업하고,
새로운 Whisper 파라미터(v2.0)로 재전사하여 덮어쓴다.
"""

import io
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.services.audio import ensure_wav_16k_mono
from app.services.transcription import transcribe_with_segments
from app.services.postprocessing import postprocess_segments

TEST_SET_DIR = Path(r"C:\Users\USER\Dropbox\패밀리룸\N Park\튜링\test_set")
BACKUP_SUFFIX = ".bak_v1"


def retranscribe_type(type_num: int) -> dict:
    """단일 타입 재전사."""
    type_dir = TEST_SET_DIR / f"type{type_num}"
    wav_path = type_dir / f"type{type_num}.wav"
    donkey_path = type_dir / f"donkey_type{type_num}.txt"

    if not wav_path.exists():
        return {"type": type_num, "status": "skip", "reason": "wav 없음"}

    # 백업 (이미 있으면 건너뜀)
    backup_path = donkey_path.with_suffix(donkey_path.suffix + BACKUP_SUFFIX)
    if donkey_path.exists() and not backup_path.exists():
        shutil.copy2(donkey_path, backup_path)

    start = time.time()

    # 1. WAV 변환 (16kHz mono)
    try:
        converted_wav = ensure_wav_16k_mono(wav_path)
    except Exception as e:
        return {"type": type_num, "status": "error", "reason": f"WAV 변환 실패: {e}"}

    # 2. Whisper 전사 (v2.0 파라미터 + 오디오 전처리)
    try:
        segments = transcribe_with_segments(converted_wav, language="ko")
    except Exception as e:
        return {"type": type_num, "status": "error", "reason": f"전사 실패: {e}"}

    # 3. 후처리 (환각 제거, 필러 정리 등)
    try:
        segments = postprocess_segments(segments)
    except Exception:
        pass  # 후처리 실패해도 원본 사용

    elapsed = time.time() - start

    # 4. donkey_typeN.txt 형식으로 변환 (기존 형식 유지)
    result = []
    for i, seg in enumerate(segments):
        result.append({
            "role": "원장님" if i % 2 == 0 else "환자",
            "index": i + 1,
            "content": seg.get("text", ""),
        })

    # 5. 저장
    donkey_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 전처리 임시파일 정리
    for suffix in [".converted.wav", ".preprocessed.wav"]:
        tmp = wav_path.with_suffix(suffix)
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    return {
        "type": type_num,
        "status": "ok",
        "segments": len(result),
        "elapsed": round(elapsed, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", type=int)
    args = parser.parse_args()

    type_nums = args.types or list(range(1, 22))

    print("=" * 60)
    print("  v2.0 파이프라인으로 전체 재전사")
    print("=" * 60)
    print()

    total_start = time.time()
    results = []

    for tn in type_nums:
        print(f"  Type {tn:2d} 전사 중...", end=" ", flush=True)
        r = retranscribe_type(tn)
        results.append(r)

        if r["status"] == "ok":
            print(f"완료 ({r['segments']}세그먼트, {r['elapsed']}초)")
        else:
            print(f"{r['status']}: {r.get('reason', '')}")

    total_elapsed = time.time() - total_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    total_segs = sum(r.get("segments", 0) for r in results)

    print()
    print(f"  완료: {ok_count}/{len(type_nums)}타입, {total_segs}세그먼트, {total_elapsed:.0f}초")
    print("=" * 60)


if __name__ == "__main__":
    main()
