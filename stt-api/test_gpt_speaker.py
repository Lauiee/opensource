"""GPT-4o-mini vs AB 규칙기반 화자분리 교정 비교 벤치마크."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from app.medterm.speaker_corrector import SpeakerCorrector

DATA_DIR = Path("C:/Users/USER/Dropbox/패밀리룸/N Park/튜링/woo_min/data_set")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """당신은 의료 대화 화자분리 전문가입니다.
아래는 병원에서의 진료 대화를 STT(음성→텍스트)로 변환한 결과입니다.
각 세그먼트의 화자(role)가 "원장님" 또는 "환자"로 배정되어 있지만, 일부가 잘못되어 있을 수 있습니다.

다음 기준으로 각 세그먼트의 올바른 화자를 판단하세요:
1. 진단, 치료 설명, 처방, 검사 안내 → 원장님
2. 증상 호소, 질문, 감정 표현, 호칭("선생님", "원장님") → 환자
3. "환자분", "어머님" 호칭 사용 → 원장님
4. "~해 드리겠습니다", "~드릴게요" 서비스 어투 → 원장님
5. 짧은 응답("네", "감사합니다")은 앞뒤 문맥으로 판단

JSON 배열로 응답하세요. 각 항목은 {"index": 원래index, "role": "원장님" 또는 "환자"} 형태입니다.
변경된 항목만 반환하세요. 변경 없으면 빈 배열 []을 반환하세요.
반드시 JSON만 반환하고, 다른 텍스트는 포함하지 마세요."""


def load_segments(path: Path) -> list[dict] | None:
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


def gpt_correct(segments: list[dict]) -> tuple[list[dict], float, int, int]:
    """GPT-4o-mini로 화자 교정.

    Returns: (changes, elapsed_sec, input_tokens, output_tokens)
    """
    # 세그먼트를 간결하게 포맷
    formatted = []
    for seg in segments:
        formatted.append({
            "index": seg.get("index", 0),
            "role": seg.get("role", ""),
            "content": seg.get("content", "")[:200],  # 토큰 절약
        })

    user_msg = json.dumps(formatted, ensure_ascii=False)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2000,
    )
    elapsed = time.perf_counter() - t0

    # 파싱
    raw = response.choices[0].message.content.strip()
    # JSON 블록 추출
    if "```" in raw:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        raw = raw[start:end]

    try:
        changes = json.loads(raw)
    except Exception:
        changes = []

    usage = response.usage
    return changes, elapsed, usage.prompt_tokens, usage.completion_tokens


def ab_correct(segments: list[dict]) -> tuple[list[dict], float]:
    """AB 규칙기반 교정.

    Returns: (changes, elapsed_sec)
    """
    corrector = SpeakerCorrector(strategy="ab")
    t0 = time.perf_counter()
    results = corrector.correct(segments)
    elapsed = time.perf_counter() - t0

    changes = []
    for seg, res in zip(segments, results):
        if res.changed:
            changes.append({
                "index": seg.get("index", 0),
                "role": res.corrected_role,
                "original": res.original_role,
                "signals": res.signals[:2],
            })
    return changes, elapsed


def main():
    print("=" * 80)
    print("  GPT-4o-mini vs AB 규칙기반 화자분리 교정 비교")
    print("=" * 80)

    # 테스트할 파일들 (문제가 뚜렷한 것 + 정상적인 것 혼합)
    test_cases = []
    for type_num in range(1, 11):
        for prefix, label in [("donkey", "Donkey"), ("dalpha", "D-Alpha")]:
            path = DATA_DIR / f"type{type_num}" / f"{prefix}_type{type_num}.txt"
            segments = load_segments(path)
            if segments:
                test_cases.append((f"Type{type_num}-{label}", segments))

    total_gpt_time = 0
    total_ab_time = 0
    total_gpt_changes = 0
    total_ab_changes = 0
    total_agree = 0
    total_disagree = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_segments = 0

    details = []

    for name, segments in test_cases:
        # 세그먼트가 너무 많으면 GPT 비용 고려하여 앞 30개만
        test_segs = segments[:30] if len(segments) > 30 else segments
        total_segments += len(test_segs)

        print(f"\n{'─'*60}")
        print(f"  {name} ({len(test_segs)}개 세그먼트)")
        print(f"{'─'*60}")

        # AB 교정
        ab_changes, ab_time = ab_correct(test_segs)
        total_ab_time += ab_time
        total_ab_changes += len(ab_changes)

        # GPT 교정
        try:
            gpt_changes, gpt_time, in_tok, out_tok = gpt_correct(test_segs)
            total_gpt_time += gpt_time
            total_gpt_changes += len(gpt_changes)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
        except Exception as e:
            print(f"  GPT 오류: {e}")
            gpt_changes = []
            gpt_time = 0
            in_tok = out_tok = 0

        # AB 변경 맵
        ab_map = {c["index"]: c["role"] for c in ab_changes}
        gpt_map = {c["index"]: c["role"] for c in gpt_changes}

        # 일치/불일치 비교
        all_changed_indices = set(ab_map.keys()) | set(gpt_map.keys())
        agree = 0
        disagree = 0
        only_ab = 0
        only_gpt = 0

        for idx in all_changed_indices:
            in_ab = idx in ab_map
            in_gpt = idx in gpt_map
            if in_ab and in_gpt:
                if ab_map[idx] == gpt_map[idx]:
                    agree += 1
                else:
                    disagree += 1
            elif in_ab:
                only_ab += 1
            else:
                only_gpt += 1

        total_agree += agree
        total_disagree += disagree

        print(f"  AB:  {len(ab_changes)}개 변경, {ab_time*1000:.1f}ms")
        print(f"  GPT: {len(gpt_changes)}개 변경, {gpt_time*1000:.0f}ms ({in_tok}+{out_tok} tokens)")
        print(f"  비교: 양쪽일치={agree}, 불일치={disagree}, AB만={only_ab}, GPT만={only_gpt}")

        # 불일치 상세
        if disagree > 0 or only_gpt > 0:
            for idx in sorted(all_changed_indices):
                in_ab = idx in ab_map
                in_gpt = idx in gpt_map

                # 해당 세그먼트 원문
                orig = next((s for s in test_segs if s.get("index") == idx), None)
                content = (orig.get("content", "")[:40] + "...") if orig else "?"
                orig_role = orig.get("role", "?") if orig else "?"

                if in_ab and in_gpt and ab_map[idx] != gpt_map[idx]:
                    print(f"    !! 불일치 idx={idx}: 원래={orig_role}, AB→{ab_map[idx]}, GPT→{gpt_map[idx]}")
                    print(f"       \"{content}\"")
                elif in_gpt and not in_ab:
                    print(f"    >> GPT만 idx={idx}: {orig_role}→{gpt_map[idx]}")
                    print(f"       \"{content}\"")

        details.append({
            "name": name,
            "segments": len(test_segs),
            "ab_changes": len(ab_changes),
            "gpt_changes": len(gpt_changes),
            "ab_time": ab_time,
            "gpt_time": gpt_time,
            "agree": agree,
            "disagree": disagree,
            "only_ab": only_ab,
            "only_gpt": only_gpt,
        })

    # ─────── 종합 결과 ─────────
    print("\n" + "=" * 80)
    print("  종합 비교 결과")
    print("=" * 80)

    gpt_cost_input = total_input_tokens / 1_000_000 * 0.15
    gpt_cost_output = total_output_tokens / 1_000_000 * 0.60
    gpt_cost = gpt_cost_input + gpt_cost_output

    print(f"""
  총 세그먼트: {total_segments}개 ({len(test_cases)}개 파일)

  ┌──────────────────────────────────────────────────────────┐
  │  항목                │  AB 규칙기반    │  GPT-4o-mini     │
  ├──────────────────────────────────────────────────────────┤
  │  총 변경 수          │  {total_ab_changes:>6}개        │  {total_gpt_changes:>6}개         │
  │  총 소요 시간        │  {total_ab_time*1000:>8.1f}ms    │  {total_gpt_time*1000:>8.0f}ms      │
  │  세그먼트당 평균     │  {total_ab_time/max(total_segments,1)*1000:>8.3f}ms    │  {total_gpt_time/max(len(test_cases),1)*1000:>8.0f}ms/파일   │
  │  비용               │  무료           │  ${gpt_cost:.4f}       │
  │  토큰 사용           │  -             │  {total_input_tokens:,}+{total_output_tokens:,}  │
  ├──────────────────────────────────────────────────────────┤
  │  양쪽 일치 변경      │  {total_agree:>6}건                                │
  │  불일치 변경         │  {total_disagree:>6}건                                │
  │  속도 차이           │  GPT가 AB보다 {total_gpt_time/max(total_ab_time,0.0001):.0f}배 느림                    │
  └──────────────────────────────────────────────────────────┘
""")

    # 결론
    if total_disagree == 0:
        print("  결론: GPT와 AB가 완전 일치! AB만으로 충분합니다.")
    elif total_disagree < total_agree * 0.1:
        print("  결론: GPT가 약간 더 많이 잡지만, AB의 속도 우위가 압도적.")
        print("        → AB 기본 + confidence 낮은 것만 GPT 2차 검증 추천")
    else:
        print("  결론: GPT가 더 많은 오류를 발견. 정확도 우선이면 GPT 고려.")

    # GPT만 잡은 것 분석
    gpt_only_total = sum(d["only_gpt"] for d in details)
    if gpt_only_total > 0:
        print(f"\n  GPT만 잡은 교정: {gpt_only_total}건 (AB가 놓친 것)")
        print("  → AB 패턴 보강 후보:")


if __name__ == "__main__":
    main()
