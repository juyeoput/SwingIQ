import streamlit as st
import matplotlib.pyplot as plt
import anthropic
import json
import tempfile
import os
import joblib

model = joblib.load("swing_model.pkl")

from utils import analyze_video

def get_ai_feedback(data, profile, swing_score):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    prompt = f"""
당신은 전문 야구 타격 코치입니다.
아래는 타자의 스윙 분석 데이터와 선수 정보입니다.

[선수 프로필]
- 이름: {profile['name']}
- 키: {profile['height']}cm
- 몸무게: {profile['weight']}kg
- 경력: {profile['experience']}년
- 주타: {profile['batting']}
- 유연성: {profile['flexibility']}

[AI 스윙 점수]
- 종합 점수: {swing_score}점 / 100점
  (100점에 가까울수록 기본기가 잘 잡힌 스윙)
- Feature importance
  (avg_head_move, max_shoulder, avg_elbow_dist, min_gap,
   avg_wrist_y, avg_gap, avg_hip, max_gap, max_hip, avg_hip_z, avg_knee_angle 순)

[스윙 데이터]
- 힙 최대 회전각도: {data['hip']['max']}도
- 어깨 최대 회전각도: {data['shoulder']['max']}도
- Kinetic Chain Gap 최대값: {data['kinetic_chain']['max_gap']}도
- Kinetic Chain Gap 최솟값: {data['kinetic_chain']['min_gap']}도
- 평균 Gap: {data['kinetic_chain']['avg_gap']}도
- 머리 평균 움직임: {data['head']['avg_movement']} (0.02 이하 안정)
- 불안정 프레임 수: {data['head']['unstable_frames']}
- 뒷팔꿈치 평균 거리: {data['elbow']['avg_distance']}
- 뒷무릎 평균 각도: {data['knee']['avg_angle']}도

[분석 기준]
- Gap 60도 이상: 좋은 kinetic chain
- Gap 0도 이하: 어깨가 먼저 열리는 문제
- 머리 움직임 0.02 이하: 안정적
- 팔꿈치 거리 0.15 이하: 이상적인 폼

선수 프로필을 반드시!!! 고려해서 개인화된 피드백을 작성해주세요.
수치 위주의 답변은 이해하기 어려우니, 야구 입문자 입장에서 이해하기 쉽게 작성해주세요.

1. 전체 평가 (2-3문장)
2. 잘하고 있는 점
3. 개선이 필요한 점
4. 이 선수에게 맞는 구체적인 훈련 방법 (1가지만)

코치가 선수에게 직접 말하는 톤으로 각 항목 2-3줄 이내로 간결하게 작성해주세요.
"""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


st.title("⚾ SwingIQ")
st.subheader("AI 타격폼 분석 서비스")

st.header("선수 정보")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("이름")
    height = st.number_input("키 (cm)", 150, 200, 175)
    weight = st.number_input("몸무게 (kg)", 50, 120, 75)
with col2:
    experience = st.number_input("야구 경력 (년)", 0, 30, 5)
    batting = st.selectbox("주타", ["우타", "좌타", "양타"])
    flexibility = st.selectbox("유연성", ["낮음", "보통", "높음"])

st.header("스윙 영상 업로드")
st.info("💡 스윙 한 번만 포함된 영상을 올려주세요. 준비 루틴 없이 스윙 동작만 찍으면 분석이 더 정확해요.")
uploaded_file = st.file_uploader("MP4 파일을 업로드하세요", type=["mp4"])

if uploaded_file and name:
    if st.button("분석 시작", type="primary"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("영상 분석 중..."):
            data = analyze_video(tmp_path, batting=batting)
        os.unlink(tmp_path)

        if not data:
            st.error("포즈를 감지하지 못했어요. 다른 영상을 시도해보세요.")
        else:
            profile = {
                "name": name, "height": height, "weight": weight,
                "experience": experience, "batting": batting, "flexibility": flexibility,
            }

            st.header("📊 스윙 분석 결과")
            frames = list(range(data["total_frames"]))
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            axes[0,0].plot(frames, data["hip_angles"], color='green', label='Hip Angle', linewidth=1.5)
            axes[0,0].plot(frames, data["shoulder_angles"], color='orange', label='Shoulder Angle', linewidth=1.5)
            axes[0,0].axhline(y=140, color='red', linestyle='--', alpha=0.5)
            axes[0,0].set_title('Hip & Shoulder Rotation')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            axes[0,1].plot(frames, data["gap_angles"], color='blue', linewidth=1.5)
            axes[0,1].axhline(y=0, color='black', alpha=0.3)
            axes[0,1].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Good gap')
            axes[0,1].fill_between(frames, data["gap_angles"], 0,
                where=[g > 0 for g in data["gap_angles"]], alpha=0.2, color='blue', label='Hip leading')
            axes[0,1].fill_between(frames, data["gap_angles"], 0,
                where=[g < 0 for g in data["gap_angles"]], alpha=0.2, color='red', label='Shoulder leading')
            axes[0,1].set_title('Kinetic Chain Gap')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            axes[1,0].plot(frames, data["head_stability"], color='purple', linewidth=1.5)
            axes[1,0].axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='Unstable threshold')
            axes[1,0].set_title('Head Stability (lower = better)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

            axes[1,1].plot(frames, data["elbow_distances"], color='brown', linewidth=1.5)
            axes[1,1].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Threshold')
            axes[1,1].set_title('Rear Elbow Distance (lower = better)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            axes[2,0].plot(frames, data["knee_angles"], color='teal', linewidth=1.5)
            axes[2,0].set_title('Rear Knee Angle')
            axes[2,0].grid(True, alpha=0.3)

            axes[2,1].plot(frames, data["wrist_y_positions"], color='red', linewidth=1.5)
            axes[2,1].set_title('Wrist Y Position (bat head drop)')
            axes[2,1].invert_yaxis()
            axes[2,1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("힙 최대 회전", f"{data['hip']['max']}°")
            col2.metric("최대 Gap", f"{data['kinetic_chain']['max_gap']}°")
            col3.metric("평균 Gap", f"{data['kinetic_chain']['avg_gap']}°")
            col4.metric("불안정 프레임", f"{data['head']['unstable_frames']}")

            #모델 점수 계산
            X_input = [[data['hip']['max'], data['hip']['avg'], data['shoulder']['max'],
                        data['kinetic_chain']['max_gap'], data['kinetic_chain']['min_gap'],
                        data['kinetic_chain']['avg_gap'], data['head']['avg_movement'],
                        data['elbow']['avg_distance'], data['knee']['avg_angle'],
                        data['bat_head']['avg_wrist_y'], data['hip_z']['avg']]]
            
            #좋은 스윙일 확률
            swing_score = round(model.predict_proba(X_input)[0][1] * 100, 1)

            st.header("🤖 AI 코칭 피드백")
            with st.spinner("AI 피드백 생성 중..."):
                feedback = get_ai_feedback(data, profile, swing_score)
            st.markdown(feedback)

            with open("swing_data.json", "w") as f:
                json.dump({k: v for k, v in data.items()
                           if k not in ["hip_angles", "shoulder_angles", "gap_angles",
                                        "head_stability", "elbow_distances",
                                        "knee_angles", "wrist_y_positions"]}, f, indent=2)