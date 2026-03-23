"""STT 설정. Faster-Whisper 오픈소스 전용."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Faster-Whisper 로컬 STT
    faster_whisper_model: str = Field(
        default="large-v3", validation_alias="FASTER_WHISPER_MODEL",
    )
    faster_whisper_beam_size: int = Field(
        default=5, validation_alias="FASTER_WHISPER_BEAM_SIZE",
    )
    faster_whisper_compute_type: str = Field(
        default="float16", validation_alias="FASTER_WHISPER_COMPUTE_TYPE",
    )

    # 공통
    default_language: str = Field(default="ko", validation_alias="DEFAULT_LANGUAGE")

    # 후처리 (의료 용어 교정, 환각 제거) 적용 여부
    enable_postprocessing: bool = Field(default=True, validation_alias="ENABLE_POSTPROCESSING")

    # 필러(추임새) 제거 활성화 여부
    enable_filler_removal: bool = Field(default=True, validation_alias="ENABLE_FILLER_REMOVAL")

    # 숫자 정규화 활성화 여부 (예: "삼십이" → "32")
    enable_number_normalization: bool = Field(default=True, validation_alias="ENABLE_NUMBER_NORMALIZATION")

    # 환각(hallucination) 제거 활성화 여부
    enable_hallucination_removal: bool = Field(default=True, validation_alias="ENABLE_HALLUCINATION_REMOVAL")

    # SOAP 노트에 핵심 요약 포함 여부
    soap_include_summary: bool = Field(default=True, validation_alias="SOAP_INCLUDE_SUMMARY")

    # 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # 의료 사전 JSON 경로
    medical_dict_path: str = Field(
        default=str(Path(__file__).resolve().parent.parent / "data" / "medical_dict.json"),
        validation_alias="MEDICAL_DICT_PATH",
    )

    # OpenAI API 키 (화자교정 GPT 2차 검증용)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # 화자교정 GPT 사용 여부
    enable_speaker_gpt: bool = Field(default=True, validation_alias="ENABLE_SPEAKER_GPT")

    # 세그먼트 병합 (인접 세그먼트 묶어 후처리/GPT 호출 감소 → 속도 향상)
    enable_segment_merge: bool = Field(default=True, validation_alias="ENABLE_SEGMENT_MERGE")
    segment_merge_max_gap_s: float = Field(default=1.0, validation_alias="SEGMENT_MERGE_MAX_GAP_S")
    segment_merge_max_duration_s: float = Field(default=15.0, validation_alias="SEGMENT_MERGE_MAX_DURATION_S")

    # 화자 분리 (pyannote) - HUGGINGFACE_TOKEN 필요 (또는 huggingface-cli login)
    enable_diarization: bool = Field(default=True, validation_alias="ENABLE_DIARIZATION")
    huggingface_token: str = Field(default="", validation_alias="HUGGINGFACE_TOKEN")
    # 화자 수. None이면 min/max만 적용(자동). 2 등 지정 시 해당 인원으로 고정
    diarization_num_speakers: int | None = Field(default=None, validation_alias="DIARIZATION_NUM_SPEAKERS")
    # 자동 탐지 시 최소 화자 수. 1=기본(완전 자동). 2로 두면 2명 대화에서 1명으로 나오는 것 방지
    diarization_min_speakers: int = Field(default=1, validation_alias="DIARIZATION_MIN_SPEAKERS")
    diarization_max_speakers: int | None = Field(default=None, validation_alias="DIARIZATION_MAX_SPEAKERS")

    @field_validator("diarization_num_speakers", "diarization_max_speakers", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v

    @field_validator("diarization_min_speakers", mode="before")
    @classmethod
    def min_speakers_default(cls, v):
        if v == "" or v is None:
            return 1
        return int(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """로그 레벨 유효성 검사"""
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if isinstance(v, str):
            v = v.upper()
            if v not in valid_levels:
                return "INFO"
        return v

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
