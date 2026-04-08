import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import anthropic
import json
import tempfile
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

def analyze_video(video_path):
    hip_angles, shoulder_angles, gap_angles = [], [], []
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip_angle = calculate_angle(left_shoulder, left_hip, right_hip)
                shoulder_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
                gap = round(hip_angle - shoulder_angle, 1)
                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)
                gap_angles.append(gap)
    cap.release()
    return hip_angles, shoulder_angles, gap_angles

def get_ai_feedback(data, profile):
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

[스윙 데이터]
- 힙 최대 회전각도: {data['max_hip_angle']}도
- 어깨 최대 회전각도: {data['max_shoulder_angle']}도
- Kinetic Chain Gap 최대값: {data['max_gap']}도
- Kinetic Chain Gap 최솟값: {data['min_gap']}도
- 평균 Gap: {data['avg_gap']}도

[분석 기준]
- Gap 60도 이상: 좋은 kinetic chain
- Gap 0도 이하: 어깨가 먼저 열리는 문제

선수 프로필을 반드시 고려해서 개인화된 피드백을 작성해주세요.

1. 전체 평가 (2-3문장)
2. 잘하고 있는 점
3. 개선이 필요한 점
4. 이 선수에게 맞는 구체적인 훈련 방법 (1-2가지)

코치가 선수에게 직접 말하는 톤으로 작성해주세요.
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
uploaded_file = st.file_uploader("MP4 파일을 업로드하세요", type=["mp4"])

if uploaded_file and name:
    if st.button("분석 시작", type="primary"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("영상 분석 중..."):
            hip_angles, shoulder_angles, gap_angles = analyze_video(tmp_path)
        os.unlink(tmp_path)

        if not hip_angles:
            st.error("포즈를 감지하지 못했어요. 다른 영상을 시도해보세요.")
        else:
            data = {
                "total_frames": len(hip_angles),
                "max_hip_angle": max(hip_angles),
                "max_shoulder_angle": max(shoulder_angles),
                "max_gap": max(gap_angles),
                "min_gap": min(gap_angles),
                "avg_gap": round(sum(gap_angles) / len(gap_angles), 1),
            }
            profile = {
                "name": name,
                "height": height,
                "weight": weight,
                "experience": experience,
                "batting": batting,
                "flexibility": flexibility,
            }

            st.header("📊 스윙 분석 결과")
            frames = list(range(len(hip_angles)))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

            ax1.plot(frames, hip_angles, color='green', label='Hip Angle', linewidth=1.5)
            ax1.plot(frames, shoulder_angles, color='orange', label='Shoulder Angle', linewidth=1.5)
            ax1.axhline(y=140, color='red', linestyle='--', alpha=0.5)
            ax1.set_title('Hip & Shoulder Rotation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames, gap_angles, color='blue', label='Kinetic Chain Gap', linewidth=1.5)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axhline(y=60, color='red', linestyle='--', alpha=0.5)
            ax2.fill_between(frames, gap_angles, 0,
                             where=[g > 0 for g in gap_angles], alpha=0.2, color='blue')
            ax2.fill_between(frames, gap_angles, 0,
                             where=[g < 0 for g in gap_angles], alpha=0.2, color='red')
            ax2.set_title('Kinetic Chain Gap')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            col1, col2, col3 = st.columns(3)
            col1.metric("힙 최대 회전", f"{data['max_hip_angle']}°")
            col2.metric("최대 Gap", f"{data['max_gap']}°")
            col3.metric("평균 Gap", f"{data['avg_gap']}°")

            st.header("🤖 AI 코칭 피드백")
            with st.spinner("AI 피드백 생성 중..."):
                feedback = get_ai_feedback(data, profile)
            st.markdown(feedback)