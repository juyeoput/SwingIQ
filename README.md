# SwingIQ ⚾
> AI 기반 야구 타격폼 분석 서비스

스마트폰으로 촬영한 타격 영상을 업로드하면 MediaPipe로 관절을 추출하고,
Claude AI가 선수 개인 맞춤형 코칭 피드백을 제공합니다.

## 주요 기능
- 힙/어깨 회전각도 분석
- Kinetic Chain Gap (상하체 분리) 측정
- 머리 고정 여부 감지
- 뒷팔꿈치 위치 분석
- 뒷무릎 각도 분석
- Claude API 기반 개인화 AI 피드백

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

## 로드맵
- Phase 1: 타격폼 분석 고도화 + KBSA 성적 연동
- Phase 2: 인바디 데이터 연동, 스윙 히스토리 트래킹
- Phase 3: 모바일 앱, Back View 분석
- Phase 4: 임베디드 센서 연동
