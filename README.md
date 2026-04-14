# SwingIQ ⚾
> AI 기반 야구 타격폼 분석 서비스

스마트폰으로 촬영한 타격 영상을 업로드하면 MediaPipe로 관절을 추출하고,
Claude AI가 선수 개인 맞춤형 코칭 피드백을 제공합니다.

## 스크린샷

### 선수 정보 입력 & 영상 업로드
<img width="993" height="1014" alt="Screenshot 2026-04-14 at 3 35 25 AM" src="https://github.com/user-attachments/assets/a34157b8-dece-4ed8-87c0-9d91d67feefe" />
<img width="1077" height="1057" alt="Screenshot 2026-04-14 at 3 07 24 AM" src="https://github.com/user-attachments/assets/6f9f3077-89ba-40d5-970d-ad1660634cae" />

### 스윙 분석 결과
<img width="1075" height="1219" alt="Screenshot 2026-04-14 at 3 09 08 AM" src="https://github.com/user-attachments/assets/0cb0af21-2de3-449f-99fd-197139fe3248" />

### AI 코칭 피드백
<img width="616" height="912" alt="Screenshot 2026-04-14 at 3 36 09 AM" src="https://github.com/user-attachments/assets/ddc8f40e-7ca7-47e0-9b31-8f155f0fcf65" />

## 시스템 구조

영상 업로드 → MediaPipe (관절 추출) → utils.py (7개 지표 계산) → Claude API (AI 피드백) → Streamlit (웹 UI)

## 분석 지표
- 힙/어깨 회전각도
- Kinetic Chain Gap (상하체 분리)
- 머리 고정 여부
- 뒷팔꿈치 위치
- 뒷무릎 각도
- Z값 기반 힙 회전 (depth 추정)
- 좌타/우타 자동 분기 처리

## 기술 스택
- Python 3.10
- MediaPipe 0.10.9
- OpenCV
- Claude API (Anthropic)
- Streamlit

## 실행 방법
```bash
conda activate baseball-analyzer
streamlit run app.py
```

## 환경 설정
`.streamlit/secrets.toml` 파일 생성 후 API 키 입력
```toml
ANTHROPIC_API_KEY = "your-api-key"
```

## 개발 기간
2026년 2월 ~ 진행 중

## 로드맵
- Phase 1: 타격폼 분석 고도화 + KBSA 성적 연동
- Phase 2: 인바디 데이터 연동, 스윙 히스토리 트래킹
- Phase 3: 모바일 앱, Back View 분석, FastAPI 전환
- Phase 4: Spring Boot 연동, 임베디드 센서 연동
