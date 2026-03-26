"""
의료 음성 STT 파이프라인 v2
- Stage 1: CLOVA Speech API (진료과별 boosting + diarization)
- Stage 2: 세그먼트 전처리 (환각필터, 병합, 짧은발화 보존)
- Stage 3: 화자 역할 매핑 (규칙기반 점수 + GPT 검증)
- Stage 4: Whisper 보완 전사 (저신뢰 구간만)
- Stage 5: GPT-4o 후처리 (강화된 프롬프트 + 진료과별 사전)
- Stage 6: 후처리 검증
- Stage 7: 2-Pass 청크 처리 (긴 대화용)

사용법:
  python medical_stt.py type8/type8.wav                    # 단일 파일
  python medical_stt.py --all                               # data_set 내 전체
  python medical_stt.py --compare                           # --all + answer 비교
  python medical_stt.py --compare --force                   # 기존 결과 무시 재처리
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import requests
from openai import OpenAI

# ─── API 설정 (비밀은 저장소에 넣지 말 것: 환경 변수) ───────────
# CLOVA: CLOVA_SPEECH_INVOKE_URL + CLOVA_SPEECH_API_KEY (또는 CLOVA_URL + CLOVA_KEY)
# OpenAI: OPENAI_API_KEY — API의 app/services/clova_note.py 가 실행 시 덮어쓸 수 있음
CLOVA_URL = (
    os.environ.get("CLOVA_SPEECH_INVOKE_URL", "").strip().rstrip("/")
    or os.environ.get("CLOVA_URL", "").strip().rstrip("/")
)
CLOVA_KEY = (
    os.environ.get("CLOVA_SPEECH_API_KEY", "").strip()
    or os.environ.get("CLOVA_KEY", "").strip()
)
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


def _medical_stt_cache_file_for_wav(wav_path: str) -> str | None:
    """동일 WAV 바이트 → 동일 결과 JSON. API·CLI가 같은 캐시를 쓰면 출력이 일치한다.

    - MEDICAL_STT_CACHE_DIR: 캐시 디렉터리 (미설정 시 {BASE_DIR}/.cache/medical_stt)
    - MEDICAL_STT_DISABLE_CACHE=1: 캐시 비활성화
    """
    if os.environ.get("MEDICAL_STT_DISABLE_CACHE", "").strip().lower() in ("1", "true", "yes"):
        return None
    base = os.environ.get("MEDICAL_STT_CACHE_DIR", "").strip()
    if not base:
        base = os.path.join(BASE_DIR, ".cache", "medical_stt")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        return None
    digest = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return os.path.join(base, digest.hexdigest() + ".json")


# ─── Type → 진료과 매핑 ────────────────────────────────────
TYPE_SPEC = {
    1: "내과", 2: "내분비내과", 3: "간담도외과", 4: "안과",
    5: "정형외과", 6: "간담도외과", 7: "정형외과", 8: "비뇨기과",
    9: "내과", 10: "정형외과", 11: "내과", 12: "감염내과",
    13: "정형외과", 14: "호흡기내과", 15: "호흡기내과",
    16: "정형외과", 17: "정형외과", 18: "정형외과",
    19: "정형외과", 20: "신장내과", 21: "내과",
}

# ─── 공통 Boosting ──────────────────────────────────────────
COMMON_BOOSTINGS = [
    {"words": "수술"}, {"words": "처방"}, {"words": "검사"}, {"words": "진단"},
    {"words": "합병증"}, {"words": "전신 마취"}, {"words": "국소 마취"},
    {"words": "항생제"}, {"words": "진통제"}, {"words": "퇴원"}, {"words": "입원"},
    {"words": "외래"}, {"words": "재발"}, {"words": "경과 관찰"}, {"words": "예후"},
    {"words": "CT"}, {"words": "MRI"}, {"words": "초음파"}, {"words": "엑스레이"},
    {"words": "내시경"}, {"words": "혈액검사"}, {"words": "혈압"}, {"words": "혈당"},
    {"words": "수혈"}, {"words": "감염"}, {"words": "종양"}, {"words": "재활"},
]

# ─── 진료과별 Boosting + 용어 사전 ─────────────────────────
SPECIALTY_DATA = {
    "정형외과": {
        "boostings": [
            {"words": "골절"}, {"words": "디스크"}, {"words": "관절염"},
            {"words": "물리치료"}, {"words": "인대"}, {"words": "연골"},
            {"words": "척추"}, {"words": "무릎"}, {"words": "고관절"},
            {"words": "발목"}, {"words": "부종"}, {"words": "염증"},
            {"words": "근육"}, {"words": "힘줄"}, {"words": "뼈"},
            {"words": "통증"}, {"words": "저림"}, {"words": "인공관절"},
            {"words": "대퇴골"}, {"words": "경골"}, {"words": "비골"},
            {"words": "슬개골"}, {"words": "반월상연골"}, {"words": "십자인대"},
            {"words": "이형성증"}, {"words": "사타구니"}, {"words": "둔부"},
            {"words": "체외충격파"}, {"words": "석회성 건염"}, {"words": "지간신경종"},
            {"words": "부정유합"}, {"words": "신경염"}, {"words": "좌골 신경통"},
            {"words": "요추 염좌"}, {"words": "관절경"}, {"words": "DNA 주사"},
            {"words": "포도당"}, {"words": "신경 주사"}, {"words": "재생 치료"},
        ],
        "terms_for_gpt": """[정형외과 용어]
골절, 디스크, 관절염, 물리치료, 인대, 연골, 척추, 무릎, 고관절, 발목, 부종,
염증, 근육, 힘줄, 뼈, 통증, 저림, 인공관절, 대퇴골, 경골, 비골, 슬개골,
반월상연골, 십자인대, 이형성증, 체외충격파, 석회성 건염, 지간신경종,
부정유합, 신경염, 좌골 신경통, 요추 염좌, 관절경, 활액막, 둔부, 외전근
[영어 의료 용어 음역]
Dark Disk Disease → 다크 디스크 디지즈
Hip dysplasia → 힙 디스플레이시아 / 고관절 이형성증
Coronal view → 코로날 뷰
Spinal canal → 척수강
Subchondral bone sclerosis → 서브콘드랄 본 스클레로시스
Sacrococcygeal ligament → 세이크로콕시지알 리가멘트
Gluteus medius → 글루테우스 메디우스
Thomas test → 토마스 테스트
Range of motion → 레인지 오브 모션
Effusion → 에퓨전
Coccygeal tip → 콕시지얼 팁
Tenderness → 텐더니스
[약품명]
조인스 정, 세레브렉스(celecoxib), 리리카(Lyrica/pregabalin),
파라마셋(paracetamol), 트리돌(Tridol/tramadol), 솔레톤(Soleton),
나프메드(Napmed), 나프록센(Naproxen)""",
    },
    "안과": {
        "boostings": [
            {"words": "백내장"}, {"words": "녹내장"}, {"words": "비문증"},
            {"words": "안압"}, {"words": "시력"}, {"words": "안약"},
            {"words": "점안제"}, {"words": "수정체"}, {"words": "망막"},
            {"words": "시신경"}, {"words": "황반"}, {"words": "각막"},
            {"words": "결막"}, {"words": "인공수정체"}, {"words": "산동 검사"},
        ],
        "terms_for_gpt": """[안과 용어]
백내장, 녹내장, 비문증, 안압, 시력, 안약, 점안제, 수정체, 망막, 시신경,
황반, 각막, 결막, 인공수정체, 산동 검사, 안저 검사, 시야 검사,
초음파 유화술, 레이저, 백내장 진행 억제, 백내장 지연제
[영어 의료 용어 음역]
OU(오유) = 양안, OD = 우안, OS = 좌안
No change → 노 체인지
Vertical opacity → 버티컬 오페시티
Posterior subcapsular opacity → 피에시오페시티 / 후낭하 혼탁
Anterior subcapsular → 전낭하""",
    },
    "간담도외과": {
        "boostings": [
            {"words": "담즙"}, {"words": "총담관"}, {"words": "낭종"},
            {"words": "담석"}, {"words": "담석증"}, {"words": "담관암"},
            {"words": "절제"}, {"words": "공장 문합술"}, {"words": "담도 재건"},
            {"words": "소화액"}, {"words": "빌리루빈"}, {"words": "간기능"},
            {"words": "소장"}, {"words": "루와이"}, {"words": "낭성"},
        ],
        "terms_for_gpt": """[간담도외과 용어]
담즙(Bile), 총담관(Common bile duct), 낭종, 담석, 담석증, 담관암,
절제, 담관 공장 문합술, 담도 재건, 소화액, 빌리루빈(Bilirubin),
간기능 검사, 소장(small intestine), 루와이(Roux-en-Y), 낭성 확장,
합병증(complication), 간, 담도(bile duct)""",
    },
    "비뇨기과": {
        "boostings": [
            {"words": "방광"}, {"words": "요도"}, {"words": "요관"},
            {"words": "스텐트"}, {"words": "소변줄"}, {"words": "배뇨장애"},
            {"words": "전립선"}, {"words": "신장"}, {"words": "카테터"},
            {"words": "배액관"}, {"words": "혈전"}, {"words": "무기폐"},
            {"words": "폐렴"}, {"words": "심호흡"}, {"words": "종양"},
        ],
        "terms_for_gpt": """[비뇨기과 용어]
방광, 요도, 요관, 스텐트, 소변줄, 배뇨장애, 전립선, 신장, 카테터,
배액관, 혈전, 무기폐, 폐렴, 심호흡, 종양, 악성, 양성, 재발,
전이, 림프절, 조직검사, 생검, 병리, 봉합, 지혈, 수혈,
전립선 비대증, 전립선암, 요로 감염, 요로 결석, 혈뇨, 단백뇨,
빈뇨, 야간뇨, 잔뇨감, PSA""",
    },
    "내과": {
        "boostings": [
            {"words": "혈압"}, {"words": "혈당"}, {"words": "콜레스테롤"},
            {"words": "당화혈색소"}, {"words": "빈혈"}, {"words": "간수치"},
            {"words": "신장 기능"}, {"words": "요산"}, {"words": "골다공증"},
            {"words": "기억력"}, {"words": "일상생활"}, {"words": "진료 의뢰서"},
            {"words": "비타민D"}, {"words": "칼슘"}, {"words": "인슐린"},
        ],
        "terms_for_gpt": """[내과 용어]
혈압, 혈당, 콜레스테롤, 당화혈색소, 빈혈, 간수치, 간 기능,
신장 기능, 콩팥 기능, 요산 수치, 골다공증, 기억력, 일상생활,
단기 기억, 장기 기억, 비타민D(Vitamin D), 칼슘, 인슐린(insulin),
단백뇨, 공복 혈당, 식후 혈당, 당뇨약, 혈압약, 췌장, 의뢰서,
양성 종변, 빌리루빈(Bilirubin), 요양보호사, 전원""",
    },
    "내분비내과": {
        "boostings": [
            {"words": "쿠싱 증후군"}, {"words": "부신"}, {"words": "호르몬"},
            {"words": "코르티솔"}, {"words": "당뇨"}, {"words": "인슐린"},
            {"words": "갑상선"}, {"words": "골다공증"}, {"words": "비타민D"},
        ],
        "terms_for_gpt": """[내분비내과 용어]
쿠싱 증후군, 부신, 호르몬, 코르티솔, 당뇨, 혈당, 인슐린,
갑상선, 갑상선 기능, 갑상선 항진증, 갑상선 저하증,
비타민D, 골다공증, 칼슘""",
    },
    "호흡기내과": {
        "boostings": [
            {"words": "흉부"}, {"words": "호흡 곤란"}, {"words": "기침"},
            {"words": "가래"}, {"words": "천식"}, {"words": "폐렴"},
            {"words": "산소포화도"}, {"words": "폐기능"}, {"words": "흡입기"},
        ],
        "terms_for_gpt": """[호흡기내과 용어]
흉부, 호흡 곤란, 숨, 기침, 가래, 천식, 폐렴,
흉부 엑스레이(X-ray), 산소포화도, 폐기능 검사, 기관지 내시경,
흡입기, 네뷸라이저, 폐, 기관지, 흉막, 늑골,
심전도 검사, 근육통""",
    },
    "감염내과": {
        "boostings": [
            {"words": "발열"}, {"words": "고열"}, {"words": "권태감"},
            {"words": "인후통"}, {"words": "해열제"}, {"words": "해열진통제"},
            {"words": "항바이러스제"}, {"words": "격리"}, {"words": "백혈구"},
            {"words": "CRP"}, {"words": "대증 치료"},
        ],
        "terms_for_gpt": """[감염내과 용어]
발열, 고열, 전신 권태감, 인후통, 콧물, 기침, 가래, 호흡 곤란, 흉통,
해열제, 해열진통제, 진통제, 항생제, 항바이러스제,
대증 치료, 충분한 휴식, 수분 섭취, 경과 관찰,
백혈구, CRP, 혈액 배양, 격리, 전파, 접촉, 재 내원""",
    },
    "신장내과": {
        "boostings": [
            {"words": "신장 기능"}, {"words": "사구체 여과율"}, {"words": "콩팥"},
            {"words": "신우"}, {"words": "단백뇨"}, {"words": "크레아티닌"},
            {"words": "수신증"}, {"words": "투석"}, {"words": "핵의학"},
        ],
        "terms_for_gpt": """[신장내과 용어]
신장 기능, 사구체 여과율(GFR/eGFR), 콩팥, 신우,
단백뇨, 혈뇨, 부종, 투석, 신장 이식, 크레아티닌,
요소질소(BUN), 소변 검사, 초음파, 핵의학 검사,
수신증(Hydronephrosis), 비뇨의학과""",
    },
}

# ─── 환각 패턴 (postprocessing.py에서 이식) ─────────────────
HALLUCINATION_PATTERNS = [
    r"시청해\s*주셔서\s*감사",
    r"구독과?\s*좋아요",
    r"좋아요와?\s*구독",
    r"채널에\s*가입",
    r"알림\s*설정",
    r"다음\s*(영상|동영상|시간|편)에",
    r"MBC\s*뉴스", r"KBS\s*뉴스", r"SBS\s*뉴스",
    r"\[음악\]", r"\[박수\]", r"♪", r"🎵",
    r"^(음악|박수|웃음)$",
    r"1[3-9]월부터", r"[2-9]\d월부터",
    r"(\d\s+){5,}",  # 숫자 나열 1 2 3 4 5 ...
    r"(.{2,10})\1{4,}",  # 같은 구절 5회 이상 반복
]
HALLUCINATION_RE = [re.compile(p) for p in HALLUCINATION_PATTERNS]

# ─── 화자 역할 판별 패턴 ────────────────────────────────────
DOCTOR_SIGNALS = [
    # 의료 행위/설명
    (r"수술", 2), (r"처방", 2), (r"검사", 1.5), (r"진단", 2),
    (r"합병증", 2), (r"마취", 2), (r"재발", 1.5), (r"퇴원", 1),
    # 지시/설명 어미
    (r"하셔야", 3), (r"되고요", 2), (r"입니다", 1), (r"거예요", 1),
    (r"거고요", 1.5), (r"드릴게요", 2), (r"설명드리", 3),
    (r"해주셔야", 2), (r"안\s*하셔도", 1.5), (r"될\s*것\s*같아요", 1.5),
    # 검사 결과 설명
    (r"나왔는데", 2), (r"나왔고", 1.5), (r"결과가", 1.5),
    (r"정상", 1), (r"이상", 1), (r"수치", 1.5),
]
PATIENT_SIGNALS = [
    # 증상 호소
    (r"아파요", 2), (r"아프", 1), (r"불편", 1.5), (r"통증", 1),
    (r"걱정", 1.5),
    # 질문
    (r"\?$", 1), (r"인가요", 1.5), (r"되나요", 1.5), (r"있나요", 1.5),
    (r"어떻게", 1),
]
# 짧은 응답은 환자 가능성 높음 (2어절 이하)
SHORT_RESPONSE_WEIGHT = 1.5


# ═══════════════════════════════════════════════════════════
# Stage 1: CLOVA Speech API (진료과별 boosting)
# ═══════════════════════════════════════════════════════════
def clova_stt(wav_path, type_num=None):
    """CLOVA Speech API로 전사. 진료과별 boosting 자동 적용."""
    if not CLOVA_URL or not CLOVA_KEY:
        raise RuntimeError(
            "CLOVA가 설정되지 않았습니다. "
            "CLOVA_SPEECH_INVOKE_URL(또는 CLOVA_URL)과 "
            "CLOVA_SPEECH_API_KEY(또는 CLOVA_KEY)를 환경 변수로 설정하세요."
        )
    headers = {"X-CLOVASPEECH-API-KEY": CLOVA_KEY}

    # 진료과별 boosting 병합
    boostings = list(COMMON_BOOSTINGS)
    if type_num and type_num in TYPE_SPEC:
        spec = TYPE_SPEC[type_num]
        spec_data = SPECIALTY_DATA.get(spec, {})
        boostings.extend(spec_data.get("boostings", []))

    # 중복 제거
    seen = set()
    unique_boostings = []
    for b in boostings:
        w = b["words"]
        if w not in seen:
            seen.add(w)
            unique_boostings.append(b)

    params = {
        "language": "ko-KR",
        "completion": "sync",
        "diarization": {"enable": True, "speakerCountMin": 2, "speakerCountMax": 4},
        "boostings": unique_boostings,
        "wordAlignment": True,
    }

    filename = os.path.basename(wav_path)
    with open(wav_path, "rb") as f:
        files = {
            "media": (filename, f, "application/octet-stream"),
            "params": (None, json.dumps(params), "application/json"),
        }
        resp = requests.post(
            f"{CLOVA_URL}/recognizer/upload",
            headers=headers,
            files=files,
            timeout=600,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"CLOVA API error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()
    segments = []
    for seg in raw.get("segments", []):
        text = seg.get("text", "").strip()
        speaker = seg.get("diarization", {}).get("label", "unknown")
        confidence = seg.get("confidence", 0.0)
        start_ms = seg.get("start", 0)
        end_ms = seg.get("end", 0)
        words = seg.get("words", [])
        if text:
            segments.append({
                "speaker": str(speaker),
                "text": text,
                "start": start_ms,
                "end": end_ms,
                "confidence": confidence,
                "words": words,
            })
    return raw, segments


# ═══════════════════════════════════════════════════════════
# Stage 2: 세그먼트 전처리
# ═══════════════════════════════════════════════════════════
def preprocess_clova_segments(segments):
    """환각 필터링 + 같은 화자 연속 병합 + 짧은 발화 보존."""
    if not segments:
        return segments

    # 2a. 환각 필터링
    filtered = []
    for seg in segments:
        text = seg["text"].strip()
        is_hallucination = False
        for pat in HALLUCINATION_RE:
            if pat.search(text):
                is_hallucination = True
                break
        if not is_hallucination and len(text) > 0:
            filtered.append(seg)

    # 2b. 같은 화자 연속 세그먼트 → 병합하지 않음
    # GPT가 턴 구조를 판단하도록 원본 세그먼트를 최대한 유지
    # 단, 매우 짧은 간격(200ms 미만)의 동일 화자 세그먼트만 병합
    merged = []
    for seg in filtered:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            gap = seg["start"] - merged[-1]["end"]
            if gap < 200:
                merged[-1]["text"] += " " + seg["text"]
                merged[-1]["end"] = seg["end"]
                merged[-1]["confidence"] = min(merged[-1]["confidence"], seg["confidence"])
                merged[-1]["words"].extend(seg.get("words", []))
                continue
        merged.append(dict(seg))

    return merged


# ═══════════════════════════════════════════════════════════
# Stage 3: 화자 역할 매핑
# ═══════════════════════════════════════════════════════════
def map_speaker_roles(segments):
    """규칙 기반 점수 + GPT 검증으로 화자 역할 매핑."""
    if not segments:
        return segments

    # 화자별 통계 수집
    speakers = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in speakers:
            speakers[spk] = {"texts": [], "total_len": 0, "count": 0}
        speakers[spk]["texts"].append(seg["text"])
        speakers[spk]["total_len"] += len(seg["text"])
        speakers[spk]["count"] += 1

    if len(speakers) < 2:
        # 화자가 1명뿐이면 모두 "원장님"으로
        for seg in segments:
            seg["role"] = "원장님"
        return segments

    # 각 화자별 의사/환자 점수 산출
    scores = {}
    for spk, data in speakers.items():
        doc_score = 0
        pat_score = 0
        full_text = " ".join(data["texts"])

        for pattern, weight in DOCTOR_SIGNALS:
            if re.search(pattern, full_text):
                doc_score += weight

        for pattern, weight in PATIENT_SIGNALS:
            if re.search(pattern, full_text):
                pat_score += weight

        # 짧은 응답 비율
        short_count = sum(1 for t in data["texts"] if len(t.split()) <= 2)
        if data["count"] > 0:
            short_ratio = short_count / data["count"]
            pat_score += short_ratio * SHORT_RESPONSE_WEIGHT * 3

        # 평균 발화 길이 (의사가 더 긺)
        avg_len = data["total_len"] / max(data["count"], 1)
        scores[spk] = {"doc": doc_score, "pat": pat_score, "avg_len": avg_len}

    # 점수로 역할 배정
    spk_list = sorted(scores.keys())
    if len(spk_list) == 2:
        s1, s2 = spk_list
        # 의사 점수가 더 높은 쪽 = 의사
        net1 = scores[s1]["doc"] - scores[s1]["pat"]
        net2 = scores[s2]["doc"] - scores[s2]["pat"]

        # 애매하면 평균 발화 길이로 보조 판단
        if abs(net1 - net2) < 1.0:
            if scores[s1]["avg_len"] > scores[s2]["avg_len"]:
                net1 += 1.0
            else:
                net2 += 1.0

        if net1 >= net2:
            role_map = {s1: "원장님", s2: "환자"}
        else:
            role_map = {s1: "환자", s2: "원장님"}

        # 점수 차이가 매우 작으면 GPT 검증
        if abs(net1 - net2) < 2.0:
            gpt_map = _gpt_verify_roles(speakers)
            if gpt_map:
                role_map = gpt_map
    else:
        # 3명 이상 → GPT에 위임
        role_map = _gpt_verify_roles(speakers)
        if not role_map:
            # fallback: 가장 긴 발화 화자 = 의사
            longest = max(speakers.keys(), key=lambda s: speakers[s]["total_len"])
            role_map = {s: ("원장님" if s == longest else "환자") for s in speakers}

    # 역할 적용
    for seg in segments:
        seg["role"] = role_map.get(seg["speaker"], "환자")

    return segments


def _gpt_verify_roles(speakers):
    """GPT로 화자 역할 검증 (대표 발화 5개만 전송)."""
    if not client:
        return None
    try:
        prompt_parts = []
        for spk, data in speakers.items():
            # 가장 긴 발화 5개 선택 (의미 있는 내용)
            sorted_texts = sorted(data["texts"], key=len, reverse=True)[:5]
            samples = "\n".join(f"  - {t}" for t in sorted_texts)
            prompt_parts.append(f"Speaker {spk}:\n{samples}")

        prompt = _openai_safe_text(
            """다음은 의료 상담 음성에서 추출한 화자별 대표 발화입니다.
각 Speaker가 의사(원장님)인지 환자인지 판별해주세요.

"""
            + "\n\n".join(_openai_safe_text(p) for p in prompt_parts)
            + """

JSON으로만 답변하세요. 예: {"1": "원장님", "2": "환자"}
- 의사: 진단, 설명, 치료 계획, 지시
- 환자: 증상 호소, 질문, 짧은 응답"""
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "화자 역할 판별기. JSON만 출력."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"    GPT role verify failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# Stage 4: Whisper 보완 전사 (저신뢰 구간)
# ═══════════════════════════════════════════════════════════
ENGLISH_MEDICAL_HINTS = re.compile(
    r"(시스|로시스|오파시티|패시티|스클레로|디지즈|에퓨전|텐더니스|"
    r"모션|리가멘트|메디우스|테스트|디스플레이|콕시지|크로콕시|세이크로)"
)

# 짧은 구간 Whisper 재전사는 환각(무관 단어 나열)이 잦아 최소 길이·품질 검사 후에만 GPT에 넘김
_WHISPER_MIN_SLICE_MS = int(os.environ.get("MEDICAL_STT_WHISPER_MIN_MS", "800"))


def _korean_char_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.replace(" ", ""))
    sb = set(b.replace(" ", ""))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _whisper_slice_looks_hallucinated(
    clova_text: str, whisper_text: str, whisper_segments: list
) -> bool:
    """짧은 슬라이스·저신뢰 구간에서 Whisper가 낸 잡음/환각이면 True → 보조문 미사용."""
    wt = (whisper_text or "").strip()
    if len(wt) < 3:
        return True
    ct = (clova_text or "").strip()

    for bad in (
        "시청해",
        "구독",
        "좋아요",
        "알림 설정",
        "자막",
        "유튜브",
    ):
        if bad in wt:
            return True

    probs = []
    nsp0 = None
    for s in whisper_segments or []:
        p = getattr(s, "avg_logprob", None)
        if p is not None and not (isinstance(p, float) and math.isnan(p)):
            probs.append(float(p))
        if nsp0 is None:
            nsp0 = getattr(s, "no_speech_prob", None)
    if probs:
        mean_p = sum(probs) / len(probs)
        if mean_p < -0.9:
            return True
    if nsp0 is not None and nsp0 > 0.5 and len(wt) < 50:
        return True

    parts = [p.strip() for p in wt.split(",") if p.strip()]
    if (
        len(parts) >= 3
        and len(wt) <= 48
        and all(2 <= len(p) <= 14 for p in parts)
        and (not ct or _korean_char_jaccard(ct, wt) < 0.2)
    ):
        return True

    if len(ct) >= 6 and _korean_char_jaccard(ct, wt) < 0.06 and len(wt) >= 8:
        return True

    return False


def whisper_supplement(segments, wav_path, type_num=None):
    """CLOVA 저신뢰 구간만 Whisper로 재전사하여 보완."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("    [Whisper] faster-whisper not installed, skipping supplement")
        return segments

    # 저신뢰/영어용어 의심 구간 식별
    targets = []
    for i, seg in enumerate(segments):
        need_whisper = False
        # 신뢰도 낮은 구간
        if seg.get("confidence", 1.0) < 0.75:
            need_whisper = True
        # 영어 의료 용어 의심
        if ENGLISH_MEDICAL_HINTS.search(seg["text"]):
            need_whisper = True
        if need_whisper:
            targets.append(i)

    if not targets:
        print("    [Whisper] No low-confidence segments, skipping")
        return segments

    print(f"    [Whisper] {len(targets)} segments to re-transcribe")

    # Whisper 모델 로드
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    except Exception:
        try:
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"    [Whisper] Model load failed: {e}")
            return segments

    # 진료과별 initial prompt
    specialty = TYPE_SPEC.get(type_num, "내과") if type_num else "내과"
    spec_data = SPECIALTY_DATA.get(specialty, {})
    terms = spec_data.get("terms_for_gpt", "")
    # 프롬프트에서 핵심 용어만 추출 (첫 줄)
    initial_prompt = f"한국어 의료 상담. {specialty}. " + ", ".join(
        [w["words"] for w in spec_data.get("boostings", [])[:15]]
    )

    for idx in targets:
        seg = segments[idx]
        start_ms = seg["start"]
        end_ms = seg["end"]

        # 오디오 슬라이스 추출
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            duration_ms = end_ms - start_ms
            if duration_ms < _WHISPER_MIN_SLICE_MS:
                continue

            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", f"{start_ms / 1000:.3f}",
                "-t", f"{duration_ms / 1000:.3f}",
                "-ar", "16000", "-ac", "1",
                tmp.name,
            ], capture_output=True, timeout=30)

            # Whisper 전사 (한국어) — 아주 짧은 슬라이스는 무음에 가깝다가 무작위 단어를 내는 경우가 많음
            whisper_segs_ko, _ = model.transcribe(
                tmp.name,
                language="ko",
                beam_size=5,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                no_speech_threshold=0.65,
            )
            whisper_list_ko = list(whisper_segs_ko)
            whisper_text = " ".join(s.text.strip() for s in whisper_list_ko).strip()
            clova_here = seg.get("text") or ""

            if _whisper_slice_looks_hallucinated(clova_here, whisper_text, whisper_list_ko):
                continue

            # 영어 자동감지 모드도 시도
            whisper_segs_auto, _ = model.transcribe(
                tmp.name,
                beam_size=5,
                vad_filter=True,
                no_speech_threshold=0.65,
            )
            whisper_auto = " ".join(s.text.strip() for s in whisper_segs_auto).strip()

            # 영어 단어 추출
            eng_words = re.findall(r"[A-Za-z][A-Za-z\s\-]{2,}", whisper_auto)

            if whisper_text and len(whisper_text) > 2:
                # CLOVA 텍스트에 Whisper 영어 용어 주입
                if eng_words:
                    seg["whisper_english"] = " ".join(eng_words)
                seg["whisper_text"] = whisper_text
                seg["whisper_auto"] = whisper_auto

        except Exception as e:
            pass
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # Whisper 모델 해제
    del model

    return segments


# ═══════════════════════════════════════════════════════════
# Stage 5: GPT-4o 후처리 (강화된 프롬프트)
# ═══════════════════════════════════════════════════════════
def _openai_safe_text(obj) -> str:
    """OpenAI HTTP JSON 본문 직렬화에 안전한 문자열로 정리.

    STT 결과에 NUL(0x00)이나 잘못된 서로게이트가 섞이면 SDK가 보내는 JSON이 깨져
    'could not parse the JSON body of your request' 400이 날 수 있음.
    """
    if obj is None:
        return ""
    s = obj if isinstance(obj, str) else str(obj)
    s = s.replace("\x00", "")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16", errors="replace")
    return s


def gpt_postprocess(segments, specialty="내과", max_tokens=8192):
    """역할 매핑 완료된 세그먼트를 GPT가 교정."""
    if not client:
        print("    GPT postprocess skipped: OPENAI_API_KEY not set")
        return segments
    specialty = _openai_safe_text(specialty)
    # 입력 텍스트 구성
    input_lines = []
    for i, seg in enumerate(segments):
        role = _openai_safe_text(seg.get("role", "unknown"))
        text = _openai_safe_text(seg.get("text", ""))
        extras = []
        if seg.get("whisper_text"):
            extras.append(f"[Whisper: {_openai_safe_text(seg['whisper_text'])}]")
        if seg.get("whisper_english"):
            extras.append(f"[English: {_openai_safe_text(seg['whisper_english'])}]")
        extra_str = " ".join(extras)
        input_lines.append(f"[{role}, seg{i+1}]: {text} {extra_str}".strip())

    clova_text = "\n".join(input_lines)

    # 진료과별 용어 사전
    spec_data = SPECIALTY_DATA.get(specialty, {})
    terms = _openai_safe_text(spec_data.get("terms_for_gpt", ""))

    system_prompt = _openai_safe_text(f"""당신은 한국어 의료 음성 전사(STT) 교정 전문가입니다.
전문 분야: {specialty}
역할: CLOVA STT + Whisper 보완 결과를 교정하여 정확한 전사문을 만듭니다.""")

    user_prompt = _openai_safe_text(f"""아래는 {specialty} 진료 상담 음성의 STT 결과입니다.
화자 역할(원장님/환자)은 이미 판별되어 있습니다.

## 교정 규칙

### 1. 화자 역할
- 각 세그먼트의 [원장님] 또는 [환자] 태그를 그대로 유지하세요.
- 역할을 변경하지 마세요.

### 2. 턴 구성 규칙 (매우 중요!)
- 화자가 바뀌면 반드시 새로운 턴
- 환자의 짧은 응답("네", "예", "아", "알겠습니다", "걸어다니고" 등)은 반드시 독립 턴으로 유지
- 의사의 긴 설명은 **주제가 전환될 때** 별도 턴으로 분리하세요:
  예) 수술 설명 → (새 턴) 합병증 설명 → (새 턴) 재발 설명 → (새 턴) 서명 안내
- 같은 화자의 연속 세그먼트라도 CLOVA가 별도 세그먼트로 분리한 것은 가능한 유지
- 각 턴은 하나의 완결된 의미 단위여야 합니다
- 너무 긴 턴(200자 이상)은 자연스러운 끊김 지점에서 분리 가능

### 3. 의료 용어 교정
{terms}
- [Whisper: ...]는 참고용입니다. 앞 문장의 CLOVA 원문과 의료 상담 맥락 모두에 어울릴 때만 반영하세요.
- Whisper가 쉼표로만 이어진 짧은 단어 나열·상식 밖 조합(예: 일상·예술 단어가 잇달아 나옴)처럼 보이면 **환각**으로 보고 CLOVA 원문을 유지하세요.
- [English: ...] 태그의 영어 의료 용어는 한국어 음역과 함께 표기
  예: "버티컬 오페시티(vertical opacity)"
- STT가 잘못 인식한 의료 용어를 위 사전을 참고하여 교정

### 4. 절대 하지 말 것
- 원본에 없는 발화를 추가하지 마세요
- 내용을 요약하거나 줄이지 마세요
- 의료 설명을 임의로 넣지 마세요
- "시청해주셔서 감사합니다" 같은 STT 환각은 제거

### 5. 출력 형식
반드시 JSON 배열만 출력. 설명 없이:
[
  {{"role": "원장님", "index": 1, "content": "..."}},
  {{"role": "환자", "index": 2, "content": "..."}},
  ...
]

=== STT 결과 ===
{clova_text}""")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )

    result_text = response.choices[0].message.content.strip()
    # Remove markdown code fences
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1] if "\n" in result_text else result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

    return json.loads(result_text)


# ═══════════════════════════════════════════════════════════
# Stage 7: 2-Pass 청크 처리 (긴 대화용)
# ═══════════════════════════════════════════════════════════
def gpt_postprocess_chunked(segments, specialty="내과", chunk_size=80):
    """긴 대화를 화자 전환 기준으로 자연스럽게 분할 처리."""
    if len(segments) <= 120:
        return gpt_postprocess(segments, specialty)

    # 화자 전환 지점에서 청크 분할
    chunks = []
    current_chunk = []
    for seg in segments:
        current_chunk.append(seg)
        if len(current_chunk) >= chunk_size:
            # 화자가 바뀌는 지점에서 끊기
            if len(current_chunk) > chunk_size - 10:
                # 현재 화자와 다른 화자가 나올 때까지 계속 추가
                chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    all_results = []
    current_index = 1

    for ci, chunk in enumerate(chunks):
        print(f"    chunk {ci+1}/{len(chunks)} ({len(chunk)} segs)...")

        # 이전 청크의 마지막 2턴을 context로 포함
        context_segs = []
        if ci > 0 and all_results:
            context_segs = all_results[-2:]

        try:
            chunk_result = gpt_postprocess(chunk, specialty)
            for item in chunk_result:
                item["index"] = current_index
                current_index += 1
            all_results.extend(chunk_result)
        except Exception as e:
            print(f"    chunk {ci+1} error: {e}")
            for seg in chunk:
                all_results.append({
                    "role": seg.get("role", "원장님"),
                    "index": current_index,
                    "content": seg["text"],
                })
                current_index += 1

        time.sleep(1)

    return all_results


# ═══════════════════════════════════════════════════════════
# Stage 6: 후처리 검증
# ═══════════════════════════════════════════════════════════
def validate_result(result, audio_duration_sec=None):
    """결과의 구조적 합리성 검증."""
    issues = []
    if not result:
        return ["Empty result"]

    # 역할 일관성: 10회 이상 연속 같은 역할은 의심
    consecutive = 1
    for i in range(1, len(result)):
        if result[i].get("role") == result[i-1].get("role"):
            consecutive += 1
            if consecutive >= 10:
                issues.append(f"10+ consecutive same-role turns at index {result[i]['index']}")
                break
        else:
            consecutive = 1

    # 인덱스 연속성
    for i, item in enumerate(result):
        if item.get("index") != i + 1:
            issues.append(f"Index discontinuity at position {i}")
            break

    # 턴 수 합리성 (음성 길이 대비)
    if audio_duration_sec and audio_duration_sec > 0:
        expected_min = max(3, audio_duration_sec / 15)  # 최소 15초당 1턴
        expected_max = audio_duration_sec / 1.5  # 최대 1.5초당 1턴
        if len(result) < expected_min:
            issues.append(f"Too few turns: {len(result)} (expected {expected_min:.0f}+)")
        if len(result) > expected_max:
            issues.append(f"Too many turns: {len(result)} (expected <{expected_max:.0f})")

    return issues


# ═══════════════════════════════════════════════════════════
# 메인 처리 함수
# ═══════════════════════════════════════════════════════════
def process_wav(wav_path, output_path=None, type_num=None):
    """WAV 파일 하나를 7-Stage 파이프라인으로 처리."""
    if not os.path.exists(wav_path):
        print(f"  ERROR: file not found: {wav_path}")
        return None

    cache_path = _medical_stt_cache_file_for_wav(wav_path)
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) > 0:
                print(f"  [cache] HIT -> {len(cached)} turns ({cache_path})")
                return cached
        except Exception as exc:
            print(f"  [cache] read failed ({exc}), running pipeline...")

    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
    if output_path is None:
        output_path = os.path.join(os.path.dirname(wav_path), "..", f"stt_result_{wav_name}.json")
        output_path = os.path.normpath(output_path)

    # type 번호 추출
    if type_num is None:
        m = re.search(r"type(\d+)", wav_name)
        if m:
            type_num = int(m.group(1))

    specialty = TYPE_SPEC.get(type_num, "내과") if type_num else "내과"
    print(f"  진료과: {specialty} (type{type_num})")

    # Stage 1: CLOVA STT
    print("  [1/6] CLOVA Speech API (진료과별 boosting)...")
    raw_result, segments = clova_stt(wav_path, type_num)

    # Raw 저장
    raw_path = os.path.join(os.path.dirname(output_path), f"clova_raw_{wav_name}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)
    print(f"         {len(segments)} segments")

    if not segments:
        print("  WARNING: no segments from CLOVA")
        return []

    # Stage 2: 전처리
    print("  [2/6] 세그먼트 전처리...")
    segments = preprocess_clova_segments(segments)
    print(f"         {len(segments)} segments after preprocessing")

    # Stage 3: 화자 역할 매핑
    print("  [3/6] 화자 역할 매핑...")
    segments = map_speaker_roles(segments)
    roles = set(seg.get("role", "?") for seg in segments)
    print(f"         역할: {roles}")

    # Stage 4: Whisper 보완
    print("  [4/6] Whisper 보완 전사...")
    segments = whisper_supplement(segments, wav_path, type_num)

    # Stage 5: GPT 후처리
    print("  [5/6] GPT-4o 후처리...")
    result = gpt_postprocess_chunked(segments, specialty)

    # Normalize keys
    for item in result:
        if "text" in item and "content" not in item:
            item["content"] = item.pop("text")

    # Stage 6: 검증
    print("  [6/6] 검증...")
    issues = validate_result(result)
    if issues:
        for issue in issues:
            print(f"    WARNING: {issue}")
    else:
        print("         OK")

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"         {len(result)} utterances -> {output_path}")

    if cache_path and result:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  [cache] saved -> {cache_path}")
        except Exception as exc:
            print(f"  [cache] save failed: {exc}")

    return result


# ═══════════════════════════════════════════════════════════
# 비교 함수
# ═══════════════════════════════════════════════════════════
def compare_with_answer(result, answer_path):
    """결과를 정답과 비교 (CER + 구조 매칭)."""
    if not os.path.exists(answer_path):
        return None

    with open(answer_path, "r", encoding="utf-8") as f:
        answer = json.load(f)

    # Index-based 매칭
    match = 0
    diffs = []
    for ans in answer:
        idx = ans["index"]
        res_list = [r for r in result if r.get("index") == idx]
        if res_list:
            res = res_list[0]
            if res.get("content", "") == ans["content"] and res.get("role", "") == ans["role"]:
                match += 1
            else:
                diffs.append({
                    "index": idx,
                    "answer_role": ans["role"],
                    "answer_content": ans["content"][:80],
                    "result_role": res.get("role", ""),
                    "result_content": res.get("content", "")[:80],
                })
        else:
            diffs.append({
                "index": idx,
                "answer_role": ans["role"],
                "answer_content": ans["content"][:80],
                "result_role": "MISSING",
                "result_content": "",
            })

    # CER 계산 (전체 텍스트 기준)
    import unicodedata
    def normalize(text):
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\w가-힣]", "", text)
        return text.lower()

    ref_text = normalize("".join(a["content"] for a in answer))
    hyp_text = normalize("".join(r.get("content", "") for r in result))

    # 간단한 CER (Levenshtein)
    cer = _levenshtein(ref_text, hyp_text) / max(len(ref_text), 1)

    return {"match": match, "total": len(answer), "diffs": diffs, "cer": cer}


def _levenshtein(s1, s2):
    """편집 거리 계산."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def find_all_wavs():
    """data_set 내 모든 type 폴더에서 WAV를 찾습니다."""
    wavs = []
    for entry in sorted(os.listdir(BASE_DIR)):
        m = re.match(r"^type(\d+)$", entry)
        if m:
            num = int(m.group(1))
            wav_path = os.path.join(BASE_DIR, entry, f"{entry}.wav")
            if os.path.exists(wav_path):
                wavs.append((num, wav_path))
    return wavs


def main():
    parser = argparse.ArgumentParser(description="의료 음성 STT 파이프라인 v2")
    parser.add_argument("files", nargs="*", help="WAV 파일 경로(들)")
    parser.add_argument("--all", action="store_true", help="전체 type 처리")
    parser.add_argument("--compare", action="store_true", help="answer 비교")
    parser.add_argument("--force", action="store_true", help="재처리")
    parser.add_argument("--no-whisper", action="store_true", help="Whisper 보완 생략")
    args = parser.parse_args()

    if args.compare:
        args.all = True

    if args.all:
        wavs = find_all_wavs()
        if not wavs:
            print(f"No WAV files found in {BASE_DIR}")
            sys.exit(1)

        summary = {}
        for num, wav_path in wavs:
            output_path = os.path.join(BASE_DIR, f"stt_result_type{num}.json")

            if os.path.exists(output_path) and not args.force:
                print(f"[type{num}] Already processed, loading...")
                with open(output_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            else:
                print(f"\n{'='*60}")
                print(f"[type{num}] Processing...")
                print("=" * 60)
                try:
                    result = process_wav(wav_path, output_path, type_num=num)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    summary[num] = {"error": str(e)}
                    continue
                time.sleep(1)

            if args.compare and result:
                answer_path = os.path.join(BASE_DIR, f"answer{num}.txt")
                cmp = compare_with_answer(result, answer_path)
                if cmp:
                    rate = cmp["match"] / cmp["total"] * 100 if cmp["total"] > 0 else 0
                    summary[num] = cmp
                    print(f"  match: {cmp['match']}/{cmp['total']} ({rate:.1f}%)  CER: {cmp['cer']:.1%}")
                    for d in cmp["diffs"][:3]:
                        print(f"    [idx {d['index']}] ans: [{d['answer_role']}] {d['answer_content']}")
                        print(f"             res: [{d['result_role']}] {d['result_content']}")
                else:
                    print("  (no answer file)")

        # Summary
        if args.compare and summary:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print("=" * 60)
            print(f"{'Type':<8} {'Match':<8} {'Total':<8} {'Rate':<8} {'CER':<8}")
            print("-" * 48)
            total_m, total_t = 0, 0
            cer_sum, cer_count = 0, 0
            for num in sorted(summary.keys()):
                s = summary[num]
                if "error" in s:
                    print(f"type{num:<4} ERROR: {s['error']}")
                    continue
                rate = s["match"] / s["total"] * 100 if s["total"] > 0 else 0
                cer = s.get("cer", 0)
                print(f"type{num:<4} {s['match']:<8} {s['total']:<8} {rate:<7.1f}% {cer:<7.1%}")
                total_m += s["match"]
                total_t += s["total"]
                cer_sum += cer
                cer_count += 1
            print("-" * 48)
            if total_t > 0:
                avg_cer = cer_sum / max(cer_count, 1)
                print(f"TOTAL    {total_m:<8} {total_t:<8} {total_m/total_t*100:<7.1f}% {avg_cer:<7.1%}")

    elif args.files:
        for wav_path in args.files:
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(BASE_DIR, wav_path)
            wav_path = os.path.normpath(wav_path)

            print(f"\n{'='*60}")
            print(f"Processing: {wav_path}")
            print("=" * 60)
            try:
                process_wav(wav_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
