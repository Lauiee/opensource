"""Tier 3: GPT 기반 문장 수준 의료 용어 교정.

저신뢰도 세그먼트에 대해 GPT를 사용하여 문맥 기반 교정을 수행한다.
- 단어 수준이 아닌 문장/문맥 수준 교정
- 주변 세그먼트 정보를 컨텍스트로 제공
- 비용 제어: 파일당 최대 교정 횟수 제한
- 캐싱: 동일 오류 반복 시 사전 등록
"""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# GPT 교정 프롬프트
_SYSTEM_PROMPT = """당신은 한국어 의료 음성인식(STT) 전문 교정기입니다.
아래 텍스트는 병원 진료 대화를 음성인식한 결과입니다.
음성인식 오류를 교정해주세요.

규칙:
1. 의료 용어가 잘못 인식된 부분만 교정하세요
2. 문맥상 의미가 통하도록 최소한의 수정만 하세요
3. 원래 의미를 추정할 수 없으면 원본을 그대로 유지하세요
4. 일반 대화체(환자 말투)는 수정하지 마세요
5. 교정된 텍스트만 반환하세요 (설명 없이)
6. 의료 용어 예시: 배뇨장애, 담즙, 총담관 낭종, 루-엔-Y 담관 공장 문합술, 백내장, DNA 등"""


class GPTSentenceCorrector:
    """GPT 기반 문장 수준 의료 용어 교정기."""

    def __init__(self, max_corrections_per_file: int = 15):
        self.max_corrections = max_corrections_per_file
        self._correction_count = 0
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 (지연 초기화)."""
        if self._client is not None:
            return self._client

        settings = get_settings()
        api_key = settings.openai_api_key
        if not api_key:
            logger.debug("OpenAI API 키 없음, GPT 교정 비활성화")
            return None

        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
            return self._client
        except ImportError:
            logger.warning("openai 패키지 미설치")
            return None
        except Exception as e:
            logger.warning("OpenAI 클라이언트 초기화 실패: %s", e)
            return None

    def reset(self):
        """파일 단위 카운터 리셋."""
        self._correction_count = 0

    def correct_segment(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
        specialty: str = "",
    ) -> Optional[str]:
        """단일 세그먼트 GPT 교정.

        Args:
            text: 교정할 텍스트
            context_before: 이전 2개 세그먼트 텍스트
            context_after: 이후 2개 세그먼트 텍스트
            specialty: 진료과 힌트

        Returns:
            교정된 텍스트 (변경 없으면 None)
        """
        if self._correction_count >= self.max_corrections:
            logger.debug("GPT 교정 횟수 제한 도달 (%d)", self.max_corrections)
            return None

        client = self._get_client()
        if client is None:
            return None

        # 컨텍스트 구성
        user_msg = f"진료과: {specialty or '일반'}\n"
        if context_before:
            user_msg += f"[이전 대화] {context_before}\n"
        user_msg += f"[교정 대상] {text}\n"
        if context_after:
            user_msg += f"[이후 대화] {context_after}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            corrected = response.choices[0].message.content.strip()
            self._correction_count += 1

            # 변경이 있으면 반환
            if corrected and corrected != text:
                logger.info(
                    "GPT Tier3 교정: '%s' → '%s'",
                    text[:40], corrected[:40],
                )
                return corrected

        except Exception as e:
            logger.warning("GPT 교정 실패: %s", e)

        return None

    def correct_low_confidence_segments(
        self,
        segments: list[dict],
        confidence_threshold: float = 0.45,
        specialty: str = "",
    ) -> list[dict]:
        """저신뢰도 세그먼트들에 대해 GPT 교정 수행.

        Args:
            segments: 전사 세그먼트 목록 (confidence 필드 포함)
            confidence_threshold: 이 값 이하인 세그먼트만 교정
            specialty: 진료과 힌트

        Returns:
            교정된 세그먼트 목록
        """
        self.reset()
        result = []

        for i, seg in enumerate(segments):
            confidence = seg.get("confidence", 1.0)

            if confidence <= confidence_threshold:
                # 주변 컨텍스트 수집
                ctx_before = " ".join(
                    s.get("text", "") for s in segments[max(0, i - 2):i]
                )
                ctx_after = " ".join(
                    s.get("text", "") for s in segments[i + 1:i + 3]
                )

                corrected = self.correct_segment(
                    seg["text"],
                    context_before=ctx_before,
                    context_after=ctx_after,
                    specialty=specialty,
                )

                if corrected:
                    new_seg = dict(seg)
                    new_seg["text"] = corrected
                    new_seg["_gpt_corrected"] = True
                    new_seg["_original_text"] = seg["text"]
                    result.append(new_seg)
                    continue

            result.append(seg)

        corrected_count = self._correction_count
        if corrected_count > 0:
            logger.info("GPT Tier3: %d개 세그먼트 교정 완료", corrected_count)

        return result


# 모듈 레벨 인스턴스
_default_corrector: Optional[GPTSentenceCorrector] = None


def get_gpt_corrector() -> GPTSentenceCorrector:
    """기본 GPT 교정기 인스턴스 반환."""
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = GPTSentenceCorrector()
    return _default_corrector
