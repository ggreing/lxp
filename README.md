# LXP PoC (FastAPI + RabbitMQ + MongoDB + MinIO + Qdrant)

> 인증은 PoC 단계에서 **없음**으로 구성했습니다. Docker만 있으면 바로 실행됩니다.

## 구성요소
- **API (FastAPI)**: 파일 업로드, 벡터스토어 관리, 작업 생성, **SSE 이벤트 스트림**
- **Worker (Python)**: 큐 소비 → AI 모듈 실행 → Mongo 로그 기록 / MinIO 산출물 저장 / 결과 이벤트 publish
- **RabbitMQ**: `ai.tasks` / `ai.results` / `ai.dlq`
- **MongoDB**: `og_keys`, `institution_{org}/(user_thread, threads, ai_log, vectorstore)`
- **MinIO**: 업로드/산출물 저장(버킷: `lxp-artifacts`)
- **Qdrant**: 임베딩 저장/검색(컬렉션: `vs_{vectorstore_id}`), **벡터 차원 256**

## 빠른 시작
```bash
cp .env.example .env
docker compose up -d --build

# API 문서
open http://localhost:8000/docs
# RabbitMQ
open http://localhost:15672  # guest/guest
# MinIO 콘솔
open http://localhost:9001   # minioadmin/minioadmin
# Qdrant 대시보드
open http://localhost:6333/dashboard
```

## 엔드포인트 요약

### Galaxy Picks (문서 및 추천)
- `POST /galaxy/documents` → 새 문서 컬렉션 생성 → `{ id }` 반환
- `POST /files/upload?user_id=U&vectorstore_id=VS` (multipart) → MinIO 저장 + `vectorstore.files` 메타 업데이트
- `POST /galaxy/documents/{id}/index` → MinIO 파일을 읽어 **청크→임베딩→Qdrant 업서트** (PoC: `.txt/.md` 지원)
- `POST /galaxy/recommendations` → RAG 기반 추천 작업 생성

### 기타 작업 생성
- `POST /translate`
  → `{ job_id, thread_id, status_url }` 반환 (메시지는 `ai.tasks`로 발행)

### 이벤트 스트림 (SSE)
- `GET /events/jobs/{job_id}`  
  → **RabbitMQ `ai.results`**를 임시 큐로 구독하여 해당 `job_id` 이벤트만 **SSE**로 푸시

예시 (브라우저/Node)
```js
const es = new EventSource("http://localhost:8000/events/jobs/<job_id>");
es.onmessage = e => console.log("event", JSON.parse(e.data));
es.onerror = e => es.close();
```

### 실시간 시뮬레이션
- `WS /sim/session` (PoC echo) – 음성 프레임/partial 결과는 실제 구현 시 확장

## RAG 파이프라인(간단 구현)
- **임베딩**: 외부 모델 없이 **토큰 해시 기반 256차원** 임베딩(결정적, 경량)  
- **색인**: 업로드 텍스트를 **중첩 오버랩 청크**로 분할 → Qdrant(`vs_{vectorstore_id}`)에 업서트  
- **검색**: 쿼리 임베딩 → Qdrant top-k 검색 → 상위 청크를 근거로 요약(단순 추출/템플릿)  
  > 실제 모델 연동 시, `worker/worker/ai_assist.py`의 로직을 대체하면 됩니다.

## RabbitMQ 토폴로지
- Exchanges: `ai.tasks`(topic), `ai.results`(topic), `ai.dlq`(fanout)
- Queues: `q.assist`, `q.galaxy`, `q.translate`, `q.sim.control`, `q.dlq`
- 라우팅키: `assist.* | galaxy.* | translate.* | sim.control.*`

## Mongo 인덱스
부팅 시 자동 보장(API `ensure_indexes`).

## 시나리오(샘플)

1) **문서 컬렉션 생성**
```bash
curl.exe -sX POST "http://localhost:8000/galaxy/documents"

# -> { "id": "<vs_id>" }
```

2) **파일 업로드 + 메타기록**
```bash
curl.exe -s -F "file=@README.md" "http://localhost:8000/files/upload?user_id=u1&vectorstore_id=689d7670ec5d54b6610cf3a5"

```

3) **인덱싱(Qdrant 업서트)**
```bash
curl.exe -sX POST "http://localhost:8000/galaxy/documents/689d7670ec5d54b6610cf3a5/index"

```

4) **Galaxy Picks(추천) 작업 생성(+RAG)**
```bash
$JOB = (curl.exe -sL -X POST "http://localhost:8000/galaxy/recommendations" `
  -H "Content-Type: application/json" `
  -d '{"user_id":"u1","prompt":"What course should I take?","vectorstore_id":"689d7670ec5d54b6610cf3a5","sub_function":"recommendation"}' `
  | ConvertFrom-Json).job_id
Write-Output $JOB

```

5) **SSE로 결과 받기**
```bash
curl.exe -N "http://localhost:8000/events/jobs/$JOB"
```

## 주의
- 임베딩은 PoC용(해시 기반). 실제 환경에선 모델 임베딩으로 교체하세요.
- 인덱싱은 텍스트 파일 중심(`.txt/.md`). PDF/문서는 추후 파서 연결 필요.
- 인증/권한은 PoC에서 비활성. 실서비스 전 OIDC/JWT·미들웨어 적용 권장.

---

### 디렉토리 트리
```
api/
  app/
    routes/ (galaxy, translate, files, events, sim, misc, sales)
    rabbitmq.py, db.py, storage.py, vector.py, config.py, schemas.py, rag_utils.py
backend/
  sales_persona_backend/
    worker/
      main.py, ai_assist.py, ...
    fastapi_app.py, rabbitmq.py, ...
```
