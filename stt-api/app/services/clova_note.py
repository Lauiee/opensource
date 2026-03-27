"""medical_stt.py 기반 Clova Speech + Whisper 보완 + 전처리."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from app.config import get_settings
from app.services.transcription import transcribe_with_segments

logger = logging.getLogger(__name__)

TYPE_SPEC: dict[int, str] = {
    1: "내과", 2: "내분비내과", 3: "간담도외과", 4: "안과",
    5: "정형외과", 6: "간담도외과", 7: "정형외과", 8: "비뇨기과",
    9: "내과", 10: "정형외과", 11: "내과", 12: "감염내과",
    13: "정형외과", 14: "호흡기내과", 15: "호흡기내과",
    16: "정형외과", 17: "정형외과", 18: "정형외과",
    19: "정형외과", 20: "신장내과", 21: "내과",
}

COMMON_BOOSTINGS: list[dict[str, str]] = [
    {"words": "수술"}, {"words": "처방"}, {"words": "검사"}, {"words": "진단"},
    {"words": "합병증"}, {"words": "전신 마취"}, {"words": "국소 마취"},
    {"words": "항생제"}, {"words": "진통제"}, {"words": "퇴원"}, {"words": "입원"},
    {"words": "외래"}, {"words": "재발"}, {"words": "경과 관찰"}, {"words": "예후"},
    {"words": "CT"}, {"words": "MRI"}, {"words": "초음파"}, {"words": "엑스레이"},
    # 당뇨/투약 관련(진료과 힌트가 없어도 자주 등장)
    {"words": "인슐린"}, {"words": "인슐린 펜"}, {"words": "인슐린 주사"}, {"words": "단위"},
    {"words": "퇴행성"}, {"words": "퇴행성 변화"}
]

SPECIALTY_BOOSTINGS: dict[str, list[dict[str, str]]] = {
    "정형외과": [{"words": "고관절"}, {"words": "무릎"}, {"words": "디스크"}, {"words": "관절염"}],
    "안과": [{"words": "백내장"}, {"words": "녹내장"}, {"words": "안압"}, {"words": "망막"}],
    "간담도외과": [{"words": "담즙"}, {"words": "총담관"}, {"words": "담석"}, {"words": "빌리루빈"}],
    "비뇨기과": [{"words": "방광"}, {"words": "요도"}, {"words": "전립선"}, {"words": "카테터"}],
    "내과": [{"words": "혈압"}, {"words": "혈당"}, {"words": "콜레스테롤"}, {"words": "당화혈색소"}],
    "내분비내과": [{"words": "쿠싱 증후군"}, {"words": "부신"}, {"words": "코르티솔"}, {"words": "인슐린"}],
    "호흡기내과": [{"words": "호흡 곤란"}, {"words": "천식"}, {"words": "폐렴"}, {"words": "산소포화도"}],
    "감염내과": [{"words": "발열"}, {"words": "고열"}, {"words": "항바이러스제"}, {"words": "CRP"}],
    "신장내과": [{"words": "신장 기능"}, {"words": "사구체 여과율"}, {"words": "단백뇨"}, {"words": "크레아티닌"}],
}

SPECIALTY_TERMS_FOR_GPT: dict[str, str] = {
    "정형외과": "골절, 디스크, 관절염, 인대, 연골, 척추, 무릎, 고관절, 인공관절, 대퇴골, 경골, 반월상연골, 십자인대",
    "안과": "백내장, 녹내장, 비문증, 안압, 수정체, 망막, 시신경, 황반, 각막, 인공수정체",
    "간담도외과": "담즙, 총담관, 담석, 담관암, 담도 재건, 빌리루빈, 루와이, 공장 문합술",
    "비뇨기과": "방광, 요도, 요관, 전립선, 카테터, 스텐트, 배뇨장애, 혈뇨, 잔뇨감",
    "내과": "혈압, 혈당, 콜레스테롤, 당화혈색소, 빈혈, 간수치, 신장 기능, 인슐린, 단백뇨",
    "내분비내과": "쿠싱 증후군, 부신, 호르몬, 코르티솔, 당뇨, 인슐린, 갑상선, 골다공증",
    "호흡기내과": "호흡 곤란, 기침, 가래, 천식, 폐렴, 산소포화도, 폐기능, 기관지",
    "감염내과": "발열, 고열, 인후통, 항바이러스제, 격리, 백혈구, CRP, 대증 치료",
    "신장내과": "신장 기능, 사구체 여과율, 단백뇨, 크레아티닌, 수신증, 투석, 콩팥",
}

DOCTOR_SIGNALS: list[tuple[str, float]] = [
    (r"수술|처방|검사|진단|합병증|마취", 2.0),
    (r"하셔야|됩니다|거예요|설명드리", 1.5),
    (r"결과가|수치|정상|이상|나왔", 1.2),
]
PATIENT_SIGNALS: list[tuple[str, float]] = [
    (r"아파요|아프|불편|통증|걱정", 1.8),
    (r"인가요|되나요|있나요|어떻게|\?$", 1.2),
]

_HALLUCINATION_RE = [
    re.compile(p) for p in [
        r"시청해\s*주셔서\s*감사",
        r"구독과?\s*좋아요",
        r"좋아요와?\s*구독",
        r"MBC\s*뉴스|KBS\s*뉴스|SBS\s*뉴스",
        r"\[음악\]|\[박수\]|♪|🎵",
        r"1[3-9]월부터|[2-9]\d월부터",
        r"(\d\s+){5,}",
        r"(.{2,10})\1{4,}",
    ]
]


def _build_boostings(type_num: int | None) -> list[dict[str, str]]:
    specialty = TYPE_SPEC.get(type_num or 0, "")
    merged = list(COMMON_BOOSTINGS) + SPECIALTY_BOOSTINGS.get(specialty, [])
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in merged:
        word = item.get("words", "")
        if word and word not in seen:
            seen.add(word)
            unique.append({"words": word})
    return unique


def _preprocess_clova_segments(segments: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if any(p.search(text) for p in _HALLUCINATION_RE):
            continue
        filtered.append(seg)

    merged: list[dict] = []
    for seg in filtered:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            gap = seg["start"] - merged[-1]["end"]
            if gap < 0.2:
                merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}".strip()
                merged[-1]["end"] = seg["end"]
                merged[-1]["confidence"] = min(
                    float(merged[-1].get("confidence", 1.0)),
                    float(seg.get("confidence", 1.0)),
                )
                continue
        merged.append(dict(seg))
    return merged


def _slice_and_whisper(wav_path: Path, start_sec: float, end_sec: float, language: str) -> str:
    dur = max(0.0, end_sec - start_sec)
    if dur < 0.3:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", f"{start_sec:.3f}",
            "-t", f"{dur:.3f}",
            "-ar", "16000", "-ac", "1",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        wsegs = transcribe_with_segments(tmp_path, language=language)
        return " ".join((s.get("text") or "").strip() for s in wsegs).strip()
    except Exception:
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


def _whisper_supplement(segments: list[dict], wav_path: Path, language: str) -> list[dict]:
    for seg in segments:
        confidence = float(seg.get("confidence", 1.0) or 1.0)
        if confidence >= 0.75:
            continue
        wt = _slice_and_whisper(wav_path, float(seg["start"]), float(seg["end"]), language)
        if wt and len(wt) > 2:
            seg["text"] = wt
    return segments


def _gpt_verify_roles(speakers: dict[str, dict]) -> dict[str, str] | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
    except Exception:
        return None

    prompt_parts: list[str] = []
    for spk, data in speakers.items():
        samples = sorted(data["texts"], key=len, reverse=True)[:5]
        bullet = "\n".join(f"- {t}" for t in samples)
        prompt_parts.append(f"Speaker {spk}\n{bullet}")

    prompt = (
        "다음은 의료 상담 음성의 화자별 발화 예시입니다.\n"
        "각 speaker가 원장님인지 환자인지 JSON으로만 반환하세요.\n"
        "예: {\"1\":\"원장님\", \"2\":\"환자\"}\n\n"
        + "\n\n".join(prompt_parts)
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "화자 역할 판별기. JSON만 출력."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = _strip_code_fence(resp.choices[0].message.content)
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        return None
    return None


def _map_speaker_roles(segments: list[dict]) -> list[dict]:
    """medical_stt.py Stage3: 화자 역할 매핑."""
    if not segments:
        return segments

    speakers: dict[str, dict] = {}
    for seg in segments:
        spk = str(seg.get("speaker", "unknown"))
        speakers.setdefault(spk, {"texts": [], "total_len": 0, "count": 0})
        txt = seg.get("text", "")
        speakers[spk]["texts"].append(txt)
        speakers[spk]["total_len"] += len(txt)
        speakers[spk]["count"] += 1

    if len(speakers) < 2:
        for seg in segments:
            seg["role"] = "원장님"
        return segments

    scores: dict[str, dict[str, float]] = {}
    for spk, data in speakers.items():
        full_text = " ".join(data["texts"])
        doc_score = 0.0
        pat_score = 0.0
        for pattern, weight in DOCTOR_SIGNALS:
            if re.search(pattern, full_text):
                doc_score += weight
        for pattern, weight in PATIENT_SIGNALS:
            if re.search(pattern, full_text):
                pat_score += weight
        short_count = sum(1 for t in data["texts"] if len(t.split()) <= 2)
        if data["count"] > 0:
            pat_score += (short_count / data["count"]) * 4.0
        avg_len = data["total_len"] / max(data["count"], 1)
        scores[spk] = {"doc": doc_score, "pat": pat_score, "avg_len": avg_len}

    role_map: dict[str, str]
    spk_list = sorted(scores.keys())
    if len(spk_list) == 2:
        s1, s2 = spk_list
        net1 = scores[s1]["doc"] - scores[s1]["pat"]
        net2 = scores[s2]["doc"] - scores[s2]["pat"]
        if abs(net1 - net2) < 1.0:
            if scores[s1]["avg_len"] >= scores[s2]["avg_len"]:
                net1 += 1.0
            else:
                net2 += 1.0
        role_map = {s1: "원장님", s2: "환자"} if net1 >= net2 else {s1: "환자", s2: "원장님"}
        if abs(net1 - net2) < 2.0:
            gpt_map = _gpt_verify_roles(speakers)
            if gpt_map:
                role_map = {
                    s1: gpt_map.get(s1, role_map[s1]),
                    s2: gpt_map.get(s2, role_map[s2]),
                }
    else:
        gpt_map = _gpt_verify_roles(speakers)
        if gpt_map:
            role_map = {spk: gpt_map.get(spk, "환자") for spk in speakers.keys()}
        else:
            longest = max(speakers.keys(), key=lambda s: speakers[s]["total_len"])
            role_map = {s: ("원장님" if s == longest else "환자") for s in speakers.keys()}

    for seg in segments:
        seg["role"] = role_map.get(str(seg.get("speaker", "unknown")), "환자")
    return segments


def _get_specialty(type_num: int | None) -> str:
    return TYPE_SPEC.get(type_num or 0, "내과")


def _strip_code_fence(text: str) -> str:
    out = (text or "").strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[1] if "\n" in out else out[3:]
        if out.endswith("```"):
            out = out[:-3]
    return out.strip()


def _gpt_postprocess_chunk(segments: list[dict], specialty: str) -> list[dict]:
    """medical_stt.py Stage5 유사 로직.

    - 입력은 role/index/content 기반
    - 세그먼트 수/순서를 보존하도록 index 고정 교정
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return segments
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
    except Exception:
        return segments

    input_lines: list[str] = []
    for i, seg in enumerate(segments):
        role = str(seg.get("role", "환자"))
        text = seg.get("text", "")
        input_lines.append(f"[{role}, index={i+1}]: {text}")
    stt_text = "\n".join(input_lines)
    terms = SPECIALTY_TERMS_FOR_GPT.get(specialty, "")

    system_prompt = (
        "당신은 한국어 의료 음성 전사(STT) 교정 전문가입니다. "
        f"전문 분야: {specialty}. "
        "JSON만 출력하세요."
    )
    user_prompt = f"""아래는 의료 상담 STT 결과입니다.
role/index는 변경하지 말고 content만 교정하세요.

규칙:
1) 세그먼트 개수와 index를 반드시 유지
2) 의료 용어 오인식만 교정, 의미 추가/요약 금지
3) 환각 문구 제거(시청해주셔서 감사합니다 등)
4) 표현은 자연스럽게 정리 가능
5) 출력은 JSON 배열만

의료 용어 힌트: {terms}

출력 형식:
[
  {{"role":"원장님|환자", "index":1, "content":"..."}},
  ...
]

입력:
{stt_text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=8192,
        )
        raw = _strip_code_fence(resp.choices[0].message.content)
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return segments

        by_index: dict[int, dict[str, str]] = {}
        for item in arr:
            try:
                idx = int(item.get("index", 0))
                txt = (item.get("content") or "").strip()
                role = str(item.get("role") or "").strip()
                if idx > 0 and txt:
                    by_index[idx] = {"text": txt, "role": role}
            except Exception:
                continue

        if len(by_index) < max(1, int(len(segments) * 0.6)):
            return segments

        out: list[dict] = []
        for i, seg in enumerate(segments, start=1):
            nseg = dict(seg)
            picked = by_index.get(i)
            if picked:
                nseg["text"] = picked["text"]
                if picked.get("role") in ("원장님", "환자"):
                    nseg["role"] = picked["role"]
            out.append(nseg)
        return out
    except Exception:
        return segments


def _gpt_postprocess_chunked(segments: list[dict], specialty: str, chunk_size: int = 80) -> list[dict]:
    """medical_stt.py Stage7 유사 청크 처리."""
    if len(segments) <= 120:
        return _gpt_postprocess_chunk(segments, specialty)

    out: list[dict] = []
    for i in range(0, len(segments), chunk_size):
        chunk = segments[i:i + chunk_size]
        out.extend(_gpt_postprocess_chunk(chunk, specialty))
        time.sleep(0.2)
    return out


def validate_result(segments: list[dict]) -> list[str]:
    """medical_stt.py Stage6 일부 검증."""
    if not segments:
        return ["Empty result"]
    issues: list[str] = []
    consecutive = 1
    for i in range(1, len(segments)):
        if str(segments[i].get("speaker")) == str(segments[i - 1].get("speaker")):
            consecutive += 1
            if consecutive >= 15:
                issues.append("long consecutive same-speaker turns")
                break
        else:
            consecutive = 1
    return issues


async def transcribe_with_clova_note(
    audio_path: str | Path,
    language: str = "ko",
    type_num: int | None = None,
) -> list[dict]:
    """medical_stt.py 파이프라인을 직접 호출해 동일 동작 보장."""
    settings = get_settings()
    api_url = (settings.clova_speech_invoke_url or settings.clova_note_api_url or "").strip()
    api_key = (settings.clova_speech_api_key or settings.clova_note_api_token or "").strip()

    if not api_url:
        raise ValueError("CLOVA_SPEECH_INVOKE_URL (또는 CLOVA_NOTE_API_URL) 이 설정되지 않았습니다.")
    if not api_key:
        raise ValueError("CLOVA_SPEECH_API_KEY (또는 CLOVA_NOTE_API_TOKEN) 이 설정되지 않았습니다.")

    path = Path(audio_path)

    # 1) medical_stt.py 직접 로드 — Docker 이미지에는 /app/medical_stt.py 로 포함됨
    candidate_paths = [
        Path("/app/medical_stt.py"),
        Path(__file__).resolve().parents[2] / "medical_stt.py",
        Path("/app/stt-api/medical_stt.py"),
    ]
    legacy_path = next((p for p in candidate_paths if p.exists()), None)

    if legacy_path is not None:
        spec = importlib.util.spec_from_file_location("medical_stt_runtime", str(legacy_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("medical_stt.py 로더를 생성할 수 없습니다.")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

        legacy.CLOVA_URL = api_url.rstrip("/")
        legacy.CLOVA_KEY = api_key

        # .env OPENAI_API_KEY가 있으면 medical_stt 런타임 client를 덮어씀(없으면 GPT 단계 스킵).
        api_openai = (settings.openai_api_key or "").strip()
        try:
            from openai import OpenAI
            if api_openai:
                legacy.OPENAI_KEY = api_openai
                legacy.client = OpenAI(api_key=api_openai)
        except Exception as exc:
            logger.warning("OpenAI 클라이언트 재설정 실패, medical_stt 기본 client 사용: %s", exc)

        def _run_legacy() -> list[dict]:
            # CLI와 완전 동일: medical_stt.process_wav 단일 진입점 (+ 바이트 해시 캐시 공유)
            fd, out_json = tempfile.mkstemp(suffix=".json", prefix="med_")
            os.close(fd)
            try:
                result = legacy.process_wav(str(path), output_path=out_json, type_num=type_num)
            finally:
                try:
                    os.unlink(out_json)
                except OSError:
                    pass
            if not result:
                return []
            out: list[dict] = []
            for item in result:
                role = str(item.get("role") or "환자")
                text = str(item.get("content") or "").strip()
                if not text:
                    continue
                idx = int(item.get("index") or (len(out) + 1))
                out.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": text,
                    "speaker": role,
                    "role": role,
                    "index": idx,
                })
            return out

        loop = asyncio.get_running_loop()
        segments = await loop.run_in_executor(None, _run_legacy)
    elif os.environ.get("STT_ALLOW_MEDICAL_FALLBACK", "").strip().lower() in ("1", "true", "yes"):
        logger.warning(
            "medical_stt.py 없음 — STT_ALLOW_MEDICAL_FALLBACK 로 내부 근사 파이프라인 사용 중 "
            "(CLI와 결과가 다를 수 있음). 배포에는 Dockerfile COPY medical_stt.py 권장."
        )
        # 2) 개발 전용: medical_stt.py 없을 때만 내부 구현
        headers = {"X-CLOVASPEECH-API-KEY": api_key}
        params = {
            "language": "ko-KR" if language.startswith("ko") else language,
            "completion": "sync",
            "diarization": {"enable": True, "speakerCountMin": 2, "speakerCountMax": 4},
            "boostings": _build_boostings(type_num),
            "wordAlignment": True,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            with path.open("rb") as f:
                files = {
                    "media": (path.name, f, "application/octet-stream"),
                    "params": (None, json.dumps(params), "application/json"),
                }
                response = await client.post(
                    f"{api_url.rstrip('/')}/recognizer/upload",
                    headers=headers,
                    files=files,
                )
            response.raise_for_status()
            payload = response.json()

        raw_segments: list[dict] = []
        for seg in payload.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start_ms = float(seg.get("start", 0))
            end_ms = float(seg.get("end", 0))
            speaker = str(seg.get("diarization", {}).get("label", "unknown"))
            raw_segments.append({
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "text": text,
                "speaker": speaker,
                "confidence": float(seg.get("confidence", 0.0) or 0.0),
            })

        raw_segments = _preprocess_clova_segments(raw_segments)
        raw_segments = _map_speaker_roles(raw_segments)
        raw_segments = _whisper_supplement(raw_segments, path, language)
        raw_segments = _gpt_postprocess_chunked(raw_segments, specialty=_get_specialty(type_num))
        segments = []
        for idx, seg in enumerate(raw_segments, start=1):
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            role = str(seg.get("role") or seg.get("speaker") or "환자")
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
                "speaker": role,
                "role": role,
                "index": idx,
            })
    else:
        in_docker = Path("/.dockerenv").exists()
        hint = (
            "Docker 이미지를 다시 빌드하세요: cd stt-api && docker compose build --no-cache && docker compose up -d"
            if in_docker
            else "저장소 루트에 medical_stt.py 가 있는지 확인하거나 STT_ALLOW_MEDICAL_FALLBACK=1 로 근사 파이프라인을 켜세요."
        )
        raise RuntimeError(
            f"medical_stt.py 를 찾을 수 없습니다. "
            f"API는 CLI와 동일 결과를 위해 해당 파일이 필요합니다. {hint}"
        )

    if not segments:
        logger.warning("medical_stt 파이프라인 결과가 비어 있습니다.")
    return segments
