import cv2
import mediapipe as mp
import numpy as np
import csv
import os

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
    """영상에서 임팩트 이전 구간 수치만 추출 후 CSV 저장"""

    # 프레임별 데이터 수집
    all_frames = []
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

                all_frames.append({
                    "hip_angle":      hip_angle,
                    "shoulder_angle": shoulder_angle,
                    "gap":            gap,
                    "head_move":      round(head_move, 4),
                    "elbow_dist":     elbow_dist,
                    "knee_angle":     knee_angle,
                    "wrist_y":        round(right_wrist.y, 4),
                    "hip_z":          hip_z,
                    "sep_z":          sep_z,
                })

    cap.release()

    if not all_frames:
        print(f"포즈 감지 실패: {video_path}")
        return False

    # 손목 Y 최저점 = 임팩트 시점
    wrist_y_list = [f["wrist_y"] for f in all_frames]
    impact_frame = wrist_y_list.index(min(wrist_y_list))

    # 임팩트 이전 구간만 사용
    pre_impact = all_frames[:impact_frame]

    if not pre_impact:
        print(f"임팩트 이전 구간 없음: {video_path}")
        return False

    print(f"전체 {len(all_frames)}프레임 중 임팩트 이전 {len(pre_impact)}프레임 사용")

    # 수치 계산
    def avg(key): return round(sum(f[key] for f in pre_impact) / len(pre_impact), 4)
    def mx(key):  return round(max(f[key] for f in pre_impact), 4)
    def mn(key):  return round(min(f[key] for f in pre_impact), 4)

    row = {
        "player":         player_name,
        "batting":        batting,
        "label":          label,
        "video":          os.path.basename(video_path),
        "max_hip":        mx("hip_angle"),
        "avg_hip":        avg("hip_angle"),
        "max_shoulder":   mx("shoulder_angle"),
        "max_gap":        mx("gap"),
        "min_gap":        mn("gap"),
        "avg_gap":        avg("gap"),
        "avg_head_move":  avg("head_move"),
        "avg_elbow_dist": avg("elbow_dist"),
        "avg_knee_angle": avg("knee_angle"),
        "avg_wrist_y":    avg("wrist_y"),
        "avg_hip_z":      avg("hip_z"),
        "max_sep_z":      mx("sep_z"),
        "avg_sep_z":      avg("sep_z"),
        "impact_frame":   impact_frame,
        "total_frames":   len(all_frames),
    }

    csv_path = "swing_dataset.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"저장 완료 → swing_dataset.csv")
    return True


if __name__ == "__main__":
    # swings 폴더 전체 처리
    swings_dir = "swings"
    file_info = {
        # 파일명: (선수이름, 우타/좌타, 라벨)
        # 예시: "swing_01.mp4": ("Player", "우타", 1),
    }

    for filename in os.listdir(swings_dir):
        if not filename.endswith(".mp4"):
            continue

        if filename not in file_info:
            print(f"정보 없음, 건너뜀: {filename}")
            continue

        player_name, batting, label = file_info[filename]
        extract_from_video(
            f"{swings_dir}/{filename}",
            player_name,
            batting,
            label
        )