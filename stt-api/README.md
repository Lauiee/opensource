# STT API Server

Donkey 프로젝트의 STT(Speech-to-Text) + **화자 분리**를 **오픈소스**로 독립 실행 가능한 API 서버입니다.

- **Faster-Whisper** (전사)
- **pyannote** (화자 분리)
- 전사 + 화자분리 **병렬 처리**로 빠른 응답

## 요구사항

- Python 3.10+
- **ffmpeg** (시스템에 설치 필요: `brew install ffmpeg`)
- **화자 분리 사용 시**: [Hugging Face](https://huggingface.co) 토큰 + [pyannote 모델 약관](https://huggingface.co/pyannote/speaker-diarization-3.1) 동의

## 설치

```bash
cd stt-api
pip install -e .
# 또는 uv 사용 시: uv sync
```

## 설정

`.env` 파일을 생성하고 변수를 설정합니다. `.env.example`을 참고하세요.

```bash
cp .env.example .env
# .env 편집
```

- **화자 분리 끄기**: `ENABLE_DIARIZATION=false` → 전사만 수행, HUGGINGFACE_TOKEN 불필요

## 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# 또는
python -m app.main
```

## Docker 배포 (온프레미스 GPU 서버)

서버에서 **git clone → .env 설정 → docker 실행**만 하면 됩니다.

```bash
# 1. 클론
git clone <repo-url>
cd Donkey-Ai/stt-api

# 2. .env 설정 (필수)
cp .env.example .env
# .env 편집: HUGGINGFACE_TOKEN 등

# 3. 빌드 & 실행
docker compose up -d --build
```

- **접속**: `http://localhost:8000`
- **GPU**: docker-compose에 GPU 설정 포함. 서버에 `nvidia-container-toolkit` 설치 필요
  - [NVIDIA 공식 설치 가이드](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - 설치 후 `nvidia-smi` 로 GPU 확인, `sudo systemctl restart docker` 실행
- **재시작**: `docker compose restart`

## API 엔드포인트

### `GET /health`

헬스 체크.

### `POST /transcribe`

오디오 URL로 전사.

```json
{
  "url": "https://example.com/audio.mp3",
  "language": "ko"
}
```

**응답:**

```json
{
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "안녕하세요", "speaker": "SPEAKER_00" }
  ],
  "full_text": "안녕하세요 ..."
}
```

(speaker: 화자 분리 시 SPEAKER_00, SPEAKER_01 등)

### `POST /transcribe/file`

오디오 파일 업로드로 전사.

- `file`: 오디오 파일 (multipart/form-data)
- `language`: (선택) 언어 코드, 기본 `ko`

## 다른 레포로 분리 시

이 `stt-api` 폴더를 그대로 복사하여 새 저장소로 푸시하면 됩니다.

원본 Donkey 프로젝트와 완전히 독립적으로 동작합니다. Redis, MySQL, S3, 요약 등은 포함되지 않습니다.
