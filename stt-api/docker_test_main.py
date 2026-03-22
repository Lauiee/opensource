"""Docker 테스트용 - /health만 (ML 의존성 없음)."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}
