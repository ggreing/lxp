# Google Cloud TTS 서비스 계정 키 설정 가이드

## 1. Google Cloud Console에서 서비스 계정 생성

### 1) Google Cloud Console 접속
- https://console.cloud.google.com/ 접속
- 프로젝트 선택

### 2) 서비스 계정 생성
- 왼쪽 메뉴 → "IAM 및 관리" → "서비스 계정"
- "서비스 계정 만들기" 클릭
- 서비스 계정 이름: `tts-service-account`
- 설명: `TTS API 사용을 위한 서비스 계정`

### 3) 권한 설정
- "역할 선택" → "Cloud Platform" → "Cloud Platform" → "편집자"
- 또는 더 구체적으로: "Cloud Text-to-Speech API" → "Cloud Text-to-Speech 사용자"

### 4) 키 생성
- 서비스 계정 생성 후 → "키" 탭
- "키 추가" → "새 키 만들기" → "JSON"
- JSON 키 파일 다운로드

## 2. 환경변수 설정 방법

### 방법 1: JSON 키를 환경변수로 설정
```bash
# .env 파일에 추가
GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account","project_id":"your-project-id",...}'
```

### 방법 2: JSON 키 파일 경로 설정
```bash
# .env 파일에 추가
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/your-service-account-key.json
```

### 방법 3: gcloud CLI 사용 (개발 환경)
```bash
# gcloud CLI 설치 후
gcloud auth application-default login
```

## 3. API 활성화

### Text-to-Speech API 활성화
- Google Cloud Console → "API 및 서비스" → "라이브러리"
- "Cloud Text-to-Speech API" 검색
- "사용" 버튼 클릭

## 4. 테스트

### 백엔드 상태 확인
```bash
curl http://localhost:8000/tts/status
```

### TTS 테스트
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"안녕하세요! TTS 테스트입니다."}'
```

## 5. 보안 주의사항

### 프로덕션 환경
- 서비스 계정 키는 절대 Git에 커밋하지 마세요
- 환경변수나 시크릿 관리 서비스 사용
- 최소 권한 원칙 적용

### 개발 환경
- `.env` 파일을 `.gitignore`에 추가
- 로컬에서만 키 파일 사용

## 6. 문제 해결

### 권한 오류
- 서비스 계정에 올바른 역할 부여
- API 활성화 확인
- 프로젝트 ID 확인

### 인증 오류
- JSON 키 형식 확인
- 환경변수 이름 확인
- 파일 경로 확인

### 네트워크 오류
- 방화벽 설정 확인
- 프록시 설정 확인 