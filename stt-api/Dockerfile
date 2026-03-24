# Donkey STT API — 운영 배포는 SERVER_DEPLOY.md 와 docker-compose.yml 을 기준으로 함.
#
# full 이미지에 포함: pyproject.toml, app/, data/, static/
# 이미지에 넣지 않음: models/ (호스트 ./models → /app/models 볼륨), .env (compose env_file)
#
# 로컬 헬스만 확인 (ML 없음, 1~2분):
#   docker build --target test -t stt-api:test .
#   docker run -p 8000:8000 stt-api:test
#   curl http://localhost:8000/health
#
# GPU 서버(compose): 호스트 8080 → 컨테이너 8000, NVIDIA 1장
#   docker compose up -d --build

# ---- test: Docker 동작 확인용 (ML 없음) ----
FROM python:3.12-slim AS test
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY docker_test_main.py ./main.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- full: 실제 STT API (SERVER_DEPLOY.md 의 운영 이미지) ----
FROM python:3.12-slim AS full
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models

COPY pyproject.toml ./
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
