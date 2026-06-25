[English](README_EN.md) | 한국어
# SwingIQ ⚾
> AI 기반 야구 타격폼 분석 서비스

🔗 **[라이브 데모 보기](https://swingiq.streamlit.app)**

기존 타격 분석 앱들은 "체중이동이 부족합니다" 수준의 표면적인 피드백만 제공하는 반면,
SwingIQ는 **왜** 문제인지, **어떻게** 고쳐야 하는지를 선수 개인 맞춤형으로 설명합니다.

초등학교부터 고등학교까지 엘리트 야구 선수로 뛰었던 개발자가 직접 설계한
biomechanics 기반 분석 엔진과 Claude AI를 결합해, 실제 코치 수준의 피드백을 제공합니다.

## 비전
- 사회인/아마추어 선수 맞춤형 AI 코칭
- 개인 성적 데이터와 타격폼 연결 분석
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
- MediaPipe 0.10.21
- OpenCV
- Claude API (Anthropic)
- Streamlit
- scikit-learn (Random Forest)
- joblib

## AI 모델
- Random Forest Classifier (교차검증 정확도 94.6%)
- 학습 데이터: 프로/아마추어 선수 스윙 39개
- 주요 피처: 머리 고정, 어깨 회전, 팔꿈치 거리, Kinetic Chain Gap, 손목 위치
- 스윙 점수 (0~100점) + Claude AI 피드백 결합

## 프로젝트 구조
```
SwingIQ/
├── app.py                  # 메인 앱 (Streamlit)
├── utils.py                 # 핵심 분석 로직 (포즈 추출, 지표 계산)
├── requirements.txt
├── packages.txt
├── models/
│   └── swing_model.pkl      # 학습된 Random Forest 모델
├── data/
│   └── swing_dataset.csv    # 학습 데이터셋
└── dev/                      # 개발용 스크립트
    ├── extract_data.py       # 영상 → 데이터셋 추출
    ├── pose_test.py          # 실시간 포즈 분석 디버깅용
    ├── feedback.py           # AI 피드백 프롬프트 테스트용
    └── train_model.ipynb     # 모델 학습 노트북
```

## 실행 방법

1. 레포지토리를 클론하고 폴더로 이동합니다.
2. (Optional, but recommended) 가상환경을 만듭니다:
```bash
conda create -n baseball-analyzer python=3.10
conda activate baseball-analyzer
```
3. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```
4. 앱을 실행합니다:
```bash
streamlit run app.py
```

## 환경 설정
`.streamlit/secrets.toml` 파일을 만들고 아래와 같이 API 키를 입력하세요.
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
