#!/bin/bash
# docker-compose v1 ContainerConfig 오류 회피용 재시작
set -e
cd "$(dirname "$0")"
docker-compose down --remove-orphans 2>/dev/null || true
docker rm -f stt-api_stt-api_1 2>/dev/null || true
docker rmi stt-api_stt-api 2>/dev/null || true
docker-compose up -d --build
