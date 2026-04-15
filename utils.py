import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """아크탄 관절 각도 계산, 세 관절 좌표로 각도 계산 (도 단위)"""
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
    """피타고리안으로 두 관절 사이 거리 계산"""
    return round(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2), 4)


def extract_metrics(lm, prev_nose_x=None, batting="우타"):
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

    if batting == "우타":
        hip_rotation_z      = round(lm[mp_pose.PoseLandmark.RIGHT_HIP].z - lm[mp_pose.PoseLandmark.LEFT_HIP].z, 4)
        shoulder_rotation_z = round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z, 4)
    else:
        hip_rotation_z      = round(lm[mp_pose.PoseLandmark.LEFT_HIP].z - lm[mp_pose.PoseLandmark.RIGHT_HIP].z, 4)
        shoulder_rotation_z = round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z, 4)

    separation_z = round(hip_rotation_z - shoulder_rotation_z, 4)

    return {
        "hip_angle":      hip_angle,
        "shoulder_angle": shoulder_angle,
        "gap":            gap,
        "head_move":      round(head_move, 4),
        "elbow_dist":     elbow_dist,
        "knee_angle":     knee_angle,
        "wrist_y":        round(right_wrist.y, 4),
        "nose_x":         nose.x,
        # Z값으로 힙/어깨 회전 추정
        "hip_rotation_z":      hip_rotation_z,
        "shoulder_rotation_z":      shoulder_rotation_z,
        "separation_z":      separation_z
    }


def analyze_video(video_path, batting="우타"):
    """영상 전체 분석 후 지표 딕셔너리 반환"""
    hip_angles, shoulder_angles, gap_angles = [], [], []
    head_stability, elbow_distances, knee_angles, wrist_y_positions = [], [], [], []
    hip_z_list = []
    prev_nose_x = None

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                metrics = extract_metrics(results.pose_landmarks.landmark, prev_nose_x, batting=batting)
                prev_nose_x = metrics["nose_x"]
                hip_angles.append(metrics["hip_angle"])
                shoulder_angles.append(metrics["shoulder_angle"])
                gap_angles.append(metrics["gap"])
                head_stability.append(metrics["head_move"])
                elbow_distances.append(metrics["elbow_dist"])
                knee_angles.append(metrics["knee_angle"])
                wrist_y_positions.append(metrics["wrist_y"])
                hip_z_list.append(metrics["hip_rotation_z"])

    cap.release()

    if not hip_angles:
        return None

    return {
        "total_frames": len(hip_angles),
        "hip_angles":        hip_angles,
        "shoulder_angles":   shoulder_angles,
        "gap_angles":        gap_angles,
        "head_stability":    head_stability,
        "elbow_distances":   elbow_distances,
        "knee_angles":       knee_angles,
        "wrist_y_positions": wrist_y_positions,
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
        "head": {
            "avg_movement":    round(sum(head_stability) / len(head_stability), 4),
            "max_movement":    round(max(head_stability), 4),
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
            "max_drop":    round(max(wrist_y_positions), 4)
        },
        "hip_z": {
            "avg": round(sum(hip_z_list) / len(hip_z_list), 4)
        }
    }