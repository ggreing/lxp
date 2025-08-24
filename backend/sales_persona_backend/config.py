from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트에 위치한 .env 로드 (상위 두 단계 경로까지 탐색)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)

# ▶︎ API / 모델 설정 ---------------------------------------------------------
API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.0-flash")
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

# 필수 키 유효성 검사 – 존재하지 않으면 실행 중단
if not API_KEY:
    raise EnvironmentError("[config] GOOGLE_API_KEY 가 설정되지 않았습니다 (.env 혹은 환경변수).")

# ▶︎ 하이브리드 메모리 파라미터 -------------------------------------------
MAX_RECENT_MESSAGES = 10
VECTOR_DIM = 384
SIMILARITY_THRESHOLD = 0.7

# ▶︎ 대화 오토‑스탑 기준 -----------------------------------------------------
MIN_DIALOGUE_LENGTH = 12