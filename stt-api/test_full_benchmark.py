"""음성파일 → STT 전사 → 의료용어 교정: 전체 파이프라인 시간 벤치마크.

faster_whisper가 이 환경에 설치되지 않으므로,
Donkey STT 실제 처리 시간을 추정(RTF 기반)하여 교정 오버헤드를 비교.
"""

import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.medterm.store import DictionaryStore
from app.medterm.engine import MedicalCorrectionEngine

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")
OUTPUT_PATH = DATA_DIR / "full_benchmark_report.txt"

# Faster-Whisper large-v3 일반적인 RTF (GPU 기준)
# RTF = 처리시간 / 음성시간. GPU float16 기준 ~0.05~0.15
WHISPER_RTF_GPU = 0.10   # GPU (CUDA float16) 추정치
WHISPER_RTF_CPU = 0.80   # CPU (int8) 추정치


def get_wav_duration(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def load_stt(type_num: int, prefix: str) -> list[dict] | None:
    path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        bracket_end = text.rfind(']')
        if bracket_end >= 0:
            text = text[:bracket_end + 1]
        return json.loads(text)
    except Exception:
        return None


def main():
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("  음성 → STT → 교정: 전체 파이프라인 시간 벤치마크")
    log("=" * 80)

    # --- 1) 교정 엔진 초기화 ---
    dict_path = Path(__file__).parent / "data" / "medical_dict.json"
    t0 = time.perf_counter()
    store = DictionaryStore(dict_path)
    engine = MedicalCorrectionEngine(store)
    t_engine_init = time.perf_counter() - t0

    log(f"\n  교정 엔진 초기화: {t_engine_init*1000:.1f}ms (사전 {len(store.get_entries())}개)")

    # --- 2) 타입별 측정 ---
    log(f"\n{'─' * 80}")
    log(f"  {'TYPE':>6}  {'음성길이':>10}  {'세그먼트':>8}  {'STT추정(GPU)':>14}  {'교정시간':>10}  {'교정수':>6}  {'오버헤드':>10}")
    log(f"{'─' * 80}")

    grand_audio = 0.0
    grand_segments = 0
    grand_corrections = 0
    grand_corr_time = 0.0
    type_data = []

    for type_num in range(1, 11):
        wav_path = DATA_DIR / f"type{type_num}" / f"type{type_num}.wav"
        if not wav_path.exists():
            continue

        audio_sec = get_wav_duration(wav_path)
        grand_audio += audio_sec

        # STT 결과 로드 (Donkey 기준)
        data = load_stt(type_num, "donkey")
        if not data:
            continue

        seg_count = len([d for d in data if d.get("content")])
        grand_segments += seg_count

        # STT 추정 시간
        stt_est_gpu = audio_sec * WHISPER_RTF_GPU
        stt_est_cpu = audio_sec * WHISPER_RTF_CPU

        # 실제 교정 시간 측정
        corrections = 0
        t_corr_start = time.perf_counter()
        for item in data:
            content = item.get("content", "")
            if content:
                result = engine.correct(content)
                if result.logs:
                    corrections += 1
        t_corr = time.perf_counter() - t_corr_start
        grand_corr_time += t_corr
        grand_corrections += corrections

        overhead_pct = (t_corr / max(stt_est_gpu, 0.001)) * 100

        type_data.append({
            "type": type_num,
            "audio_sec": audio_sec,
            "segments": seg_count,
            "stt_gpu": stt_est_gpu,
            "stt_cpu": stt_est_cpu,
            "corr_time": t_corr,
            "corrections": corrections,
            "overhead": overhead_pct,
        })

        log(f"  type{type_num:>2}  {audio_sec:>8.1f}초  {seg_count:>6}개  {stt_est_gpu:>10.2f}초  {t_corr*1000:>8.1f}ms  {corrections:>4}개  +{overhead_pct:>7.3f}%")

    # --- 3) D-Alpha도 측정 ---
    log(f"\n{'─' * 80}")
    log(f"  D-Alpha STT 교정 시간")
    log(f"{'─' * 80}")

    dalpha_corr_time = 0.0
    dalpha_corrections = 0
    dalpha_segments = 0
    for type_num in range(1, 11):
        data = load_stt(type_num, "dalpha")
        if not data:
            continue
        seg_count = len([d for d in data if d.get("content")])
        dalpha_segments += seg_count
        corrections = 0
        t_start = time.perf_counter()
        for item in data:
            content = item.get("content", "")
            if content:
                result = engine.correct(content)
                if result.logs:
                    corrections += 1
        t_elapsed = time.perf_counter() - t_start
        dalpha_corr_time += t_elapsed
        dalpha_corrections += corrections
        if corrections > 0:
            log(f"  type{type_num:>2}  세그먼트 {seg_count}개  교정 {corrections}개  {t_elapsed*1000:.1f}ms")

    # --- 4) 전체 요약 ---
    total_stt_gpu = grand_audio * WHISPER_RTF_GPU
    total_stt_cpu = grand_audio * WHISPER_RTF_CPU
    total_all = grand_corr_time + dalpha_corr_time

    log(f"\n{'=' * 80}")
    log(f"  전체 요약")
    log(f"{'=' * 80}")
    log(f"")
    log(f"  ┌────────────────────────────────────────────────────────────────┐")
    log(f"  │  총 음성 길이              {grand_audio:>8.1f}초  ({grand_audio/60:.1f}분)          │")
    log(f"  │  총 세그먼트 (Donkey)      {grand_segments:>8d}개                          │")
    log(f"  │  총 세그먼트 (D-Alpha)     {dalpha_segments:>8d}개                          │")
    log(f"  ├────────────────────────────────────────────────────────────────┤")
    log(f"  │                      STT만           STT + 교정    교정 추가  │")
    log(f"  ├────────────────────────────────────────────────────────────────┤")
    log(f"  │  GPU (float16)    {total_stt_gpu:>8.2f}초       {total_stt_gpu + grand_corr_time:>8.2f}초    +{grand_corr_time*1000:>6.1f}ms │")
    log(f"  │  CPU (int8)       {total_stt_cpu:>8.2f}초       {total_stt_cpu + grand_corr_time:>8.2f}초    +{grand_corr_time*1000:>6.1f}ms │")
    log(f"  ├────────────────────────────────────────────────────────────────┤")
    log(f"  │  교정 오버헤드 (GPU 대비)             +{grand_corr_time/max(total_stt_gpu, 0.001)*100:.4f}%            │")
    log(f"  │  교정 오버헤드 (CPU 대비)             +{grand_corr_time/max(total_stt_cpu, 0.001)*100:.4f}%            │")
    log(f"  ├────────────────────────────────────────────────────────────────┤")
    log(f"  │  교정 엔진 초기화                     {t_engine_init*1000:>8.1f}ms              │")
    log(f"  │  Donkey 교정 건수                     {grand_corrections:>8d}개               │")
    log(f"  │  D-Alpha 교정 건수                    {dalpha_corrections:>8d}개               │")
    log(f"  │  세그먼트당 평균 교정                 {grand_corr_time/max(grand_segments,1)*1000:>8.3f}ms              │")
    log(f"  └────────────────────────────────────────────────────────────────┘")
    log(f"")
    log(f"  * STT 시간은 Faster-Whisper large-v3 기준 추정치")
    log(f"    - GPU RTF: {WHISPER_RTF_GPU} (1초 음성 → {WHISPER_RTF_GPU}초 처리)")
    log(f"    - CPU RTF: {WHISPER_RTF_CPU} (1초 음성 → {WHISPER_RTF_CPU}초 처리)")
    log(f"  * 교정 시간은 실측값 (Tier 1 사전 교정)")
    log(f"  * 모델 로드 시간은 서버 시작 시 1회만 발생 (보통 5~15초)")
    log(f"")
    log(f"  결론: 교정 모듈은 전체 STT 파이프라인에 {grand_corr_time/max(total_stt_gpu,0.001)*100:.4f}%의 시간만 추가합니다.")
    log(f"        세그먼트당 {grand_corr_time/max(grand_segments,1)*1000:.3f}ms로, 실시간 처리에 전혀 영향 없습니다.")

    # 파일 저장
    report_text = "\n".join(lines)
    OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n리포트 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
