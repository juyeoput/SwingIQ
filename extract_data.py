import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys

mp_pose = mp.solutions.pose

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

def calculate_distance(a, b):
    return round(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2), 4)

def extract_from_video(video_path, player_name, batting="우타", label=None):
    """영상에서 평균 수치 추출 후 CSV에 저장"""
    hip_angles, shoulder_angles, gap_angles = [], [], []
    head_stability, elbow_distances, knee_angles, wrist_y_positions = [], [], [], []
    hip_z_list, sep_z_list = [], []
    prev_nose_x = None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"분석 중: {video_path} ({total_frames} 프레임)")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

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
                prev_nose_x    = nose.x

                if batting == "우타":
                    hip_z = round(lm[mp_pose.PoseLandmark.RIGHT_HIP].z - lm[mp_pose.PoseLandmark.LEFT_HIP].z, 4)
                    sh_z  = round(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z, 4)
                else:
                    hip_z = round(lm[mp_pose.PoseLandmark.LEFT_HIP].z - lm[mp_pose.PoseLandmark.RIGHT_HIP].z, 4)
                    sh_z  = round(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z, 4)

                sep_z = round(hip_z - sh_z, 4)

                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)
                gap_angles.append(gap)
                head_stability.append(head_move)
                elbow_distances.append(elbow_dist)
                knee_angles.append(knee_angle)
                wrist_y_positions.append(right_wrist.y)
                hip_z_list.append(hip_z)
                sep_z_list.append(sep_z)

    cap.release()

    if not hip_angles:
        print(f"포즈 감지 실패: {video_path}")
        return False

    row = {
        "player":           player_name,
        "batting":          batting,
        "label":            label,
        "video":            os.path.basename(video_path),
        "max_hip":          max(hip_angles),
        "avg_hip":          round(sum(hip_angles) / len(hip_angles), 1),
        "max_shoulder":     max(shoulder_angles),
        "max_gap":          max(gap_angles),
        "min_gap":          min(gap_angles),
        "avg_gap":          round(sum(gap_angles) / len(gap_angles), 1),
        "avg_head_move":    round(sum(head_stability) / len(head_stability), 4),
        "avg_elbow_dist":   round(sum(elbow_distances) / len(elbow_distances), 4),
        "avg_knee_angle":   round(sum(knee_angles) / len(knee_angles), 1),
        "avg_wrist_y":      round(sum(wrist_y_positions) / len(wrist_y_positions), 4),
        "avg_hip_z":        round(sum(hip_z_list) / len(hip_z_list), 4),
        "avg_sep_z":        round(sum(sep_z_list) / len(sep_z_list), 4),
        "max_sep_z":        round(max(sep_z_list), 4),
        "total_frames":     len(hip_angles),
    }

    csv_path = "swing_dataset.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"저장 완료 → swing_dataset.csv")
    return True


if __name__ == "__main__":
    # 여기서 영상 정보 입력
    VIDEO_PATH  = "swing_01.mp4"   # 영상 파일명
    PLAYER_NAME = "홍길동"          # 선수 이름
    BATTING     = "우타"            # 우타 / 좌타
    LABEL       = None             # 나중에 라벨링할 때 입력 (1=좋음, 0=나쁨)

    success = extract_from_video(VIDEO_PATH, PLAYER_NAME, BATTING, LABEL)

    if success:
        print("영상 삭제할까요? (y/n)")
        answer = input()
        if answer.lower() == "y":
            os.remove(VIDEO_PATH)
            print(f"{VIDEO_PATH} 삭제됨")