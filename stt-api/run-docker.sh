#!/bin/bash
# docker-compose v1 (ContainerConfig 오류) 대신 docker run으로 실행
# 사용: ./run-docker.sh [build|up|down|restart]

set -e
cd "$(dirname "$0")"
# -p stt-opensource 에 맞춘 프로젝트명 (다른 이름: PROJECT=xxx ./run-docker.sh up)
PROJECT="${PROJECT:-stt-opensource}"
IMAGE="stt-api:local"
CONTAINER="${PROJECT}_stt-api_1"

case "${1:-up}" in
  build)
    docker build --target full -t "$IMAGE" .
    echo "빌드 완료: $IMAGE"
    ;;
  down)
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
    echo "컨테이너 중지/제거: $CONTAINER"
    ;;
  up)
    # 기존 컨테이너 정리 (recreate 대신 수동)
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true

    if ! docker image inspect "$IMAGE" &>/dev/null; then
      echo "이미지 없음. 먼저 ./run-docker.sh build 실행"
      exit 1
    fi

    [ -f .env ] || { echo ".env 파일 필요"; exit 1; }
    [ -d models ] || mkdir -p models

    docker run -d \
      --name "$CONTAINER" \
      --restart unless-stopped \
      -p 8080:8000 \
      --env-file .env \
      -v "$(pwd)/models:/app/models" \
      --gpus all \
      "$IMAGE"

    echo "실행 중: $CONTAINER (http://localhost:8080/health)"
    ;;
  restart)
    $0 down
    $0 up
    ;;
  *)
    echo "사용법: $0 {build|up|down|restart}"
    exit 1
    ;;
esac
