# SwingIQ ⚾
> AI 기반 야구 타격폼 분석 서비스

기존 타격 분석 앱들은 "체중이동이 부족합니다" 수준의 표면적인 피드백만 제공하는 반면,
SwingIQ는 **왜** 문제인지, **어떻게** 고쳐야 하는지를 선수 개인 맞춤형으로 설명합니다.

초등학교부터 고등학교까지 엘리트 야구 선수로 뛰었던 개발자가 직접 설계한
biomechanics 기반 분석 엔진과 Claude AI를 결합해, 실제 코치 수준의 피드백을 제공합니다.

궁극적인 목표는 아마추어부터 국가대표까지 — **대한민국 야구 경쟁력 강화**입니다.

## 비전
- 사회인/아마추어 선수 맞춤형 AI 코칭
- KBSA 성적 데이터와 타격폼 연결 분석
- 임베디드 센서 + 영상 분석 결합
- 전 포지션 확장 및 국가대표급 데이터 플랫폼

## 스크린샷

### 선수 정보 입력 & 영상 업로드
<img width="1069" height="1070" alt="Screenshot 2026-04-14 at 3 43 03 AM" src="https://github.com/user-attachments/assets/d14038d3-a6c3-4996-9372-fedaa8f86243" />
<img width="1072" height="1084" alt="Screenshot 2026-04-14 at 3 44 06 AM" src="https://github.com/user-attachments/assets/f7d572ce-399b-4206-9b4c-bd022a3afd72" />

### 스윙 분석 결과
<img width="1058" height="1379" alt="Screenshot 2026-04-14 at 3 44 28 AM" src="https://github.com/user-attachments/assets/a12c4c42-efe5-4cd8-a2c7-c3610f88f8c5" />

### AI 코칭 피드백
<img width="1056" height="1616" alt="Screenshot 2026-04-14 at 3 44 43 AM" src="https://github.com/user-attachments/assets/ceebcffb-f2d6-4c78-90ec-0ecc410f40a1" />

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
