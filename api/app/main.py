from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import galaxy, coach, translate, files, events, sim, misc, sales
from . import rabbitmq
from .db import ensure_indexes
from .config import settings

app = FastAPI(title="LXP PoC API (v2 fixed)")

# ✅ 프론트 개발 서버(8080)에서 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    # RabbitMQ 토폴로지(큐/익스체인지) 보장
    await rabbitmq.ensure_topology()
    # Mongo 인덱스 보장(중복 정리 포함)
    await ensure_indexes(settings.app_org_id)

# 라우터 등록
app.include_router(coach.router, prefix="/coach", tags=["coach"])
app.include_router(galaxy.router, prefix="/galaxy", tags=["galaxy"])
app.include_router(translate.router, prefix="/translate", tags=["translate"])
app.include_router(files.router, prefix="/files", tags=["files"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(sim.router, prefix="/sim", tags=["sim"])
app.include_router(sales.router, prefix="/sales", tags=["sales"])
app.include_router(misc.router, tags=["misc"])
