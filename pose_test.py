import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json

from utils import analyze_video, extract_metrics, calculate_distance

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

VIDEO_PATH = "choo_swing.mp4"


def draw_overlay(frame, metrics):
    """영상 위에 실시간 수치 오버레이"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    items = [
        ("Hip",      f"{metrics['hip_angle']}°",      metrics['hip_angle'] > 100),
        ("Shoulder", f"{metrics['shoulder_angle']}°",  True),
        ("Gap",      f"{metrics['gap']}°",             metrics['gap'] > 0),
        ("Head",     f"{metrics['head_move']:.4f}",    metrics['head_move'] < 0.02),
        ("Elbow",    f"{metrics['elbow_dist']:.3f}",   metrics['elbow_dist'] < 0.15),
        ("Knee",     f"{metrics['knee_angle']}°",      True),
        ("Wrist Y",  f"{metrics['wrist_y']:.3f}",      True),
        ("Hip Z",   f"{metrics['hip_rotation_z']:.4f}",   True),
        ("Sep Z",   f"{metrics['separation_z']:.4f}",      True),
    ]

    for i, (label, value, is_good) in enumerate(items):
        color = (100, 255, 100) if is_good else (80, 80, 255)
        cv2.putText(frame, f"{label}: {value}",
                    (20, 42 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return frame


def process_frame(frame, pose, prev_nose_x=None):
    """프레임 포즈 추출 후 오버레이 적용"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    metrics = None
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        metrics = extract_metrics(results.pose_landmarks.landmark, prev_nose_x, batting="좌타")
        frame = draw_overlay(frame, metrics)
    return frame, metrics


def draw_frame_counter(frame, current, total):
    """프레임 번호 표시"""
    cv2.putText(frame, f"Frame: {current}/{total}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


print("영상 분석 중...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_nose_x = None

hip_angles, shoulder_angles, gap_angles = [], [], []
head_stability, elbow_distances, knee_angles, wrist_y_positions = [], [], [], []

hip_rotation_z_list = []
separation_z_list = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame, metrics = process_frame(frame, pose, prev_nose_x)
        if metrics:
            prev_nose_x = metrics["nose_x"]
            hip_angles.append(metrics["hip_angle"])
            shoulder_angles.append(metrics["shoulder_angle"])
            gap_angles.append(metrics["gap"])
            head_stability.append(metrics["head_move"])
            elbow_distances.append(metrics["elbow_dist"])
            knee_angles.append(metrics["knee_angle"])
            wrist_y_positions.append(metrics["wrist_y"])

            hip_rotation_z_list.append(metrics["hip_rotation_z"])
            separation_z_list.append(metrics["separation_z"])

        cv2.imshow("SwingIQ — 실시간 분석", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = True
            while paused:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' '):
                    paused = False
                elif key2 == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                elif key2 in (2, 3):
                    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if key2 == 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min(current + 5, total_frames - 1))
                    elif key2 == 2:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(current - 6, 0))
                    ret2, frame2 = cap.read()
                    if ret2 and frame2 is not None:
                        frame2, _ = process_frame(frame2, pose)
                        draw_frame_counter(frame2, int(cap.get(cv2.CAP_PROP_POS_FRAMES)), total_frames)
                        cv2.imshow("SwingIQ — 실시간 분석", frame2)

cap.release()
cv2.destroyAllWindows()
print(f"분석 완료 — 총 {len(hip_angles)} 프레임")

frames = list(range(len(hip_angles)))
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('SwingIQ — Swing Analysis', fontsize=16, fontweight='bold')

axes[0,0].plot(frames, hip_angles, color='green', label='Hip', linewidth=1.5)
axes[0,0].plot(frames, shoulder_angles, color='orange', label='Shoulder', linewidth=1.5)
axes[0,0].axhline(y=140, color='red', linestyle='--', alpha=0.5, label='Swing threshold')
axes[0,0].set_title('Hip & Shoulder Rotation')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(frames, gap_angles, color='blue', linewidth=1.5)
axes[0,1].axhline(y=0, color='black', alpha=0.3)
axes[0,1].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Good gap (60+)')
axes[0,1].fill_between(frames, gap_angles, 0,
    where=[g > 0 for g in gap_angles], alpha=0.2, color='blue', label='Hip leading')
axes[0,1].fill_between(frames, gap_angles, 0,
    where=[g < 0 for g in gap_angles], alpha=0.2, color='red', label='Shoulder leading')
axes[0,1].set_title('Kinetic Chain Gap')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(range(len(head_stability)), head_stability, color='purple', linewidth=1.5)
axes[1,0].axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='Unstable threshold')
axes[1,0].set_title('Head Stability (lower = better)')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(frames, elbow_distances, color='brown', linewidth=1.5)
axes[1,1].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Threshold')
axes[1,1].set_title('Rear Elbow Distance (lower = better)')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

axes[2,0].plot(frames, knee_angles, color='teal', linewidth=1.5)
axes[2,0].set_title('Rear Knee Angle')
axes[2,0].grid(True, alpha=0.3)

axes[2,1].plot(frames, wrist_y_positions, color='red', linewidth=1.5)
axes[2,1].set_title('Wrist Y Position (bat head drop)')
axes[2,1].invert_yaxis()
axes[2,1].grid(True, alpha=0.3)


fig2, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.plot(frames, hip_rotation_z_list, color='navy', label='Hip Z', linewidth=1.5)
ax.plot(frames, separation_z_list, color='crimson', label='Separation Z', linewidth=1.5)
ax.axhline(y=0, color='black', alpha=0.3)
ax.set_title('Hip & Separation Z (depth)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('swing_z_analysis.png', dpi=150)
plt.show()
print("그래프 저장됨 → swing_analysis.png")

try:
    data = analyze_video(VIDEO_PATH)
    if data:
        save_data = {k: v for k, v in data.items()
                     if k not in ["hip_angles", "shoulder_angles", "gap_angles",
                                  "head_stability", "elbow_distances",
                                  "knee_angles", "wrist_y_positions"]}
        with open("swing_data.json", "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print("데이터 저장됨 → swing_data.json")
except Exception as e:
    print(f"에러: {e}")