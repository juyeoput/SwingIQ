import streamlit as st
import matplotlib.pyplot as plt
import anthropic
import json
import tempfile
import os
import joblib

model = joblib.load("models/swing_model.pkl")

from utils import analyze_video

TEXTS = {
    "한국어": {
        "title": "⚾ SwingIQ",
        "subtitle": "AI 타격폼 분석 서비스",
        "player_info_header": "선수 정보",
        "name_label": "이름",
        "height_label": "키 (cm)",
        "weight_label": "몸무게 (kg)",
        "experience_label": "야구 경력 (년)",
        "batting_label": "주타",
        "batting_options": ["우타", "좌타", "양타"],
        "flexibility_label": "유연성",
        "flexibility_options": ["낮음", "보통", "높음"],
        "upload_header": "스윙 영상 업로드",
        "upload_info": "스윙 한 번만 포함된 영상을 올려주세요. 준비 루틴 없이 스윙 동작만 찍으면 분석이 더 정확해요.",
        "upload_label": "MP4 파일을 업로드하세요",
        "analyze_btn": "분석 시작",
        "analyzing_spinner": "영상 분석 중...",
        "no_pose_error": "포즈를 감지하지 못했어요. 다른 영상을 시도해보세요.",
        "result_header": "스윙 분석 결과",
        "metric_hip": "힙 최대 회전",
        "metric_max_gap": "최대 Gap",
        "metric_avg_gap": "평균 Gap",
        "metric_unstable": "불안정 프레임",
        "feedback_header": "AI 코칭 피드백",
        "feedback_spinner": "AI 피드백 생성 중...",
    },
    "English": {
        "title": "⚾ SwingIQ",
        "subtitle": "AI-Powered Swing Analysis",
        "player_info_header": "Player Info",
        "name_label": "Name",
        "height_label": "Height (cm)",
        "weight_label": "Weight (kg)",
        "experience_label": "Years of Experience",
        "batting_label": "Batting Stance",
        "batting_options": ["Right", "Left", "Switch"],
        "flexibility_label": "Flexibility",
        "flexibility_options": ["Low", "Average", "High"],
        "upload_header": "Upload Swing Video",
        "upload_info": "Upload a video containing a single swing. For best accuracy, capture just the swing motion without a pre-swing routine.",
        "upload_label": "Upload an MP4 file",
        "analyze_btn": "Start Analysis",
        "analyzing_spinner": "Analyzing video...",
        "no_pose_error": "Couldn't detect a pose. Please try a different video.",
        "result_header": "Swing Analysis Results",
        "metric_hip": "Max Hip Rotation",
        "metric_max_gap": "Max Gap",
        "metric_avg_gap": "Avg Gap",
        "metric_unstable": "Unstable Frames",
        "feedback_header": "AI Coaching Feedback",
        "feedback_spinner": "Generating AI feedback...",
    },
}

BATTING_MAP = {
    "한국어": {"우타": "우타", "좌타": "좌타", "양타": "양타"},
    "English": {"Right": "우타", "Left": "좌타", "Switch": "양타"},
}
FLEX_MAP = {
    "한국어": {"낮음": "낮음", "보통": "보통", "높음": "높음"},
    "English": {"Low": "낮음", "Average": "보통", "High": "높음"},
}

_title_col, _lang_col = st.columns([4, 1])
with _lang_col:
    lang = st.selectbox("🌐 Language", ["한국어", "English"])
t = TEXTS[lang]

def get_ai_feedback(data, profile, swing_score, lang):
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
    if lang == "English":
        prompt += "\n\n위 내용을 전부 영어로 작성해주세요. (Please write the entire feedback above in English.)"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


with _title_col:
    st.title(t["title"])
st.subheader(t["subtitle"])

st.header(t["player_info_header"])
col1, col2 = st.columns(2)
with col1:
    name = st.text_input(t["name_label"])
    height = st.number_input(t["height_label"], 150, 200, 175)
    weight = st.number_input(t["weight_label"], 50, 120, 75)
with col2:
    experience = st.number_input(t["experience_label"], 0, 30, 5)
    batting_display = st.selectbox(t["batting_label"], t["batting_options"])
    flexibility_display = st.selectbox(t["flexibility_label"], t["flexibility_options"])

batting = BATTING_MAP[lang][batting_display]
flexibility = FLEX_MAP[lang][flexibility_display]

st.header(t["upload_header"])
st.info(t["upload_info"])
uploaded_file = st.file_uploader(t["upload_label"], type=["mp4"])

if uploaded_file and name:
    if st.button(t["analyze_btn"], type="primary"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(t["analyzing_spinner"]):
            data = analyze_video(tmp_path, batting=batting)
        os.unlink(tmp_path)

        if not data:
            st.error(t["no_pose_error"])
        else:
            profile = {
                "name": name, "height": height, "weight": weight,
                "experience": experience, "batting": batting_display, "flexibility": flexibility_display,
            }

            st.header(t["result_header"])
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
            col1.metric(t["metric_hip"], f"{data['hip']['max']}°")
            col2.metric(t["metric_max_gap"], f"{data['kinetic_chain']['max_gap']}°")
            col3.metric(t["metric_avg_gap"], f"{data['kinetic_chain']['avg_gap']}°")
            col4.metric(t["metric_unstable"], f"{data['head']['unstable_frames']}")

            #모델 점수 계산
            X_input = [[data['hip']['max'], data['hip']['avg'], data['shoulder']['max'],
                        data['kinetic_chain']['max_gap'], data['kinetic_chain']['min_gap'],
                        data['kinetic_chain']['avg_gap'], data['head']['avg_movement'],
                        data['elbow']['avg_distance'], data['knee']['avg_angle'],
                        data['bat_head']['avg_wrist_y'], data['hip_z']['avg']]]
            
            #좋은 스윙일 확률
            swing_score = round(model.predict_proba(X_input)[0][1] * 100, 1)

            st.header(t["feedback_header"])
            with st.spinner(t["feedback_spinner"]):
                feedback = get_ai_feedback(data, profile, swing_score, lang)
            st.markdown(feedback)

            with open("swing_data.json", "w") as f:
                json.dump({k: v for k, v in data.items()
                           if k not in ["hip_angles", "shoulder_angles", "gap_angles",
                                        "head_stability", "elbow_distances",
                                        "knee_angles", "wrist_y_positions"]}, f, indent=2)