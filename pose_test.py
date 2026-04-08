import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

VIDEO_PATH = "video1.mp4"


def calculate_angle(a, b, c):
    """세 관절 좌표로 각도 계산 (도 단위)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)


def calculate_distance(a, b):
    """두 관절 사이 거리 계산"""
    return round(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2), 4)


def extract_metrics(lm, prev_nose_x=None):
    """랜드마크에서 스윙 지표 추출"""
    left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip       = lm[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow    = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_knee     = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
    right_ankle    = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
    right_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose           = lm[mp_pose.PoseLandmark.NOSE]

    hip_angle      = calculate_angle(left_shoulder, left_hip, right_hip)
    shoulder_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
    gap            = round(hip_angle - shoulder_angle, 1)
    elbow_dist     = calculate_distance(right_elbow, right_hip)
    knee_angle     = calculate_angle(right_hip, right_knee, right_ankle)
    head_move      = abs(nose.x - prev_nose_x) if prev_nose_x is not None else 0.0

    return {
        "hip_angle":      hip_angle,
        "shoulder_angle": shoulder_angle,
        "gap":            gap,
        "head_move":      round(head_move, 4),
        "elbow_dist":     elbow_dist,
        "knee_angle":     knee_angle,
        "wrist_y":        round(right_wrist.y, 4),
        "nose_x":         nose.x,
    }


def draw_overlay(frame, metrics):
    """영상 위에 실시간 수치 오버레이"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    items = [
        ("힙 회전",    f"{metrics['hip_angle']}°",      metrics['hip_angle'] > 100),
        ("어깨 회전",  f"{metrics['shoulder_angle']}°",  True),
        ("상하체 분리", f"{metrics['gap']}°",            metrics['gap'] > 0),
        ("헤드 고정",  f"{metrics['head_move']:.4f}",    metrics['head_move'] < 0.02),
        ("팔꿈치 거리", f"{metrics['elbow_dist']:.3f}",  metrics['elbow_dist'] < 0.15),
        ("무릎 각도",  f"{metrics['knee_angle']}°",      True),
        ("손목 Y",    f"{metrics['wrist_y']:.3f}",       True),
    ]

    for i, (label, value, is_good) in enumerate(items):
        color = (100, 255, 100) if is_good else (80, 80, 255)
        cv2.putText(frame, f"{label}: {value}",
                    (20, 42 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return frame


def process_frame(frame, pose):
    """프레임에서 포즈 추출 후 오버레이 적용, metrics 반환"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    metrics = None
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        metrics = extract_metrics(results.pose_landmarks.landmark)
        frame = draw_overlay(frame, metrics)
    return frame, metrics


def draw_frame_counter(frame, current, total):
    """프레임 번호 표시"""
    cv2.putText(frame, f"Frame: {current}/{total}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


hip_angles, shoulder_angles, gap_angles = [], [], []
head_stability, elbow_distances, knee_angles, wrist_y_positions = [], [], [], []

print("영상 분석 중...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_nose_x = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            metrics = extract_metrics(results.pose_landmarks.landmark, prev_nose_x)
            prev_nose_x = metrics["nose_x"]

            hip_angles.append(metrics["hip_angle"])
            shoulder_angles.append(metrics["shoulder_angle"])
            gap_angles.append(metrics["gap"])
            head_stability.append(metrics["head_move"])
            elbow_distances.append(metrics["elbow_dist"])
            knee_angles.append(metrics["knee_angle"])
            wrist_y_positions.append(metrics["wrist_y"])

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = draw_overlay(frame, metrics)

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
                elif key2 in (ord('d'), ord('a')):
                    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if key2 == ord('d'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min(current + 5, total_frames - 1))
                    else:
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

plt.tight_layout()
plt.savefig('swing_analysis.png', dpi=150)
plt.show()
print("그래프 저장됨 → swing_analysis.png")

try:
    swing_data = {
        "total_frames": len(hip_angles),
        "hip": {
            "max": max(hip_angles),
            "avg": round(sum(hip_angles) / len(hip_angles), 1)
        },
        "shoulder": {
            "max": max(shoulder_angles),
            "avg": round(sum(shoulder_angles) / len(shoulder_angles), 1)
        },
        "kinetic_chain": {
            "max_gap": max(gap_angles),
            "min_gap": min(gap_angles),
            "avg_gap": round(sum(gap_angles) / len(gap_angles), 1)
        },
        "head_stability": {
            "avg_movement": round(sum(head_stability) / len(head_stability), 4),
            "max_movement": round(max(head_stability), 4),
            "unstable_frames": len([h for h in head_stability if h > 0.02])
        },
        "elbow": {
            "avg_distance": round(sum(elbow_distances) / len(elbow_distances), 4),
            "min_distance": round(min(elbow_distances), 4)
        },
        "knee": {
            "avg_angle": round(sum(knee_angles) / len(knee_angles), 1),
            "min_angle": round(min(knee_angles), 1)
        },
        "bat_head": {
            "avg_wrist_y": round(sum(wrist_y_positions) / len(wrist_y_positions), 4),
            "max_drop": round(max(wrist_y_positions), 4)
        }
    }

    with open("swing_data.json", "w") as f:
        json.dump(swing_data, f, indent=2, ensure_ascii=False)

    print("데이터 저장됨 → swing_data.json")

except Exception as e:
    print(f"에러: {e}")