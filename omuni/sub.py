import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from scr.mouse_traking import MouseTracker  # 軌跡データ取得
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, motor_m4, stop_all  # モーター制御

# カメラ設定
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 60
DT = 1.0 / FPS
PIXEL = 63.0 / 480.0
WAIT = 1

# PIDゲイン
K_P = 0.7
K_I = 0.5
K_D = 0.03

# マウス移動のスケーリング
DEVICE_PATH = '/dev/input/event5'
SCALING = 0.0172  # [mm]

# クロップ範囲
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
frame_center = (HEIGHT // 2, HEIGHT // 2)

# グローバル変数
mouse_x, mouse_y = 0.0, 0.0
prev_offset_x = 0.0
prev_offset_y = 0.0

# CSVログ設定
LOG_INTERVAL = 0.1  # [秒]

def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    #cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    return cap if cap.read()[0] else None

def gray_binary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return max_contour, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return max_contour, (cx, cy)

def fit_ellipse_if_possible(contour):
    if contour is not None and len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]:
            angle += 90
        angle %= 180
        angle_rad = math.radians(angle)
        return ellipse, angle_rad
    return None, None

def calculate_offset(center, frame_center, fps, angle_rad):
    global prev_offset_x, prev_offset_y, integral_p, integral_n

    if angle_rad is None:
        angle_rad = 9.0

    offset_x = (center[0] - frame_center[0]) * PIXEL
    offset_y = (-center[1] + frame_center[1]) * PIXEL
    offset_z = 300 * np.cos(angle_rad) / 4

    position_p = offset_x + offset_y
    position_n = offset_x - offset_y

    integral_p += position_p * DT
    integral_n += position_n * DT

    prev_position_p = ((offset_x - prev_offset_x) + (offset_y - prev_offset_y)) / DT
    prev_position_n = ((offset_x - prev_offset_x) - (offset_y - prev_offset_y)) / DT

    drive_m1 = position_p 
    drive_m2 = position_n 
    drive_m3 = - position_n 
    drive_m4 = - position_p 

    speed_m1 = K_P * drive_m1 + K_I * integral_p + K_D * prev_position_p + offset_z
    speed_m2 = K_P * drive_m2 + K_I * integral_n + K_D * prev_position_n + offset_z
    speed_m3 = K_P * drive_m3 - K_I * integral_n - K_D * prev_position_n + offset_z
    speed_m4 = K_P * drive_m4 - K_I * integral_p - K_D * prev_position_p + offset_z

    prev_offset_x = offset_x
    prev_offset_y = offset_y

    return offset_x, offset_y, speed_m1, speed_m2, speed_m3, speed_m4

def draw_overlay(frame, center, offset_x, offset_y, ellipse, fps, angle_rad):
    cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)
    if center:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        offset_text = f"(x,y)=({offset_x:.2f},{offset_y:.2f})"
        cv2.putText(frame, offset_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    if ellipse is not None:
        degrees = math.degrees(angle_rad)
        angle_text = f"Angle: {degrees:.2f} rad"
        cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    mouse_pos_text = f"trajectory_data (x,y): ({mouse_x:.2f}, {mouse_y:.2f})"
    cv2.putText(frame, mouse_pos_text, (100, 400), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

def move_motors(speed_m1, speed_m2, speed_m3, speed_m4):
    motor_m1(int(speed_m1))
    motor_m2(int(speed_m2))
    motor_m3(int(speed_m3))
    motor_m4(int(speed_m4))

def initialize_csv_logger(filename="log.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "mouse_x[mm]", "mouse_y[mm]", "angle_rad"])

def log_to_csv(filename, elapsed_time, mouse_x, mouse_y, angle_rad):
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{elapsed_time:.1f}",
            f"{mouse_x:.2f}",
            f"{mouse_y:.2f}",
            f"{angle_rad:.2f}" if angle_rad is not None else "-1.00"
        ])

# --- メイン処理 ---
if __name__ == "__main__":
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger("log.csv")

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない")
        mouse_tracker.stop()
        exit()

    prev_time = time.time()
    integral_p = 0.0
    integral_n = 0.0
    is_logging = False
    start_log_time = None
    last_log_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            if center is not None:
                offset_x, offset_y, speed_m1, speed_m2, speed_m3, speed_m4 = calculate_offset(
                    center, frame_center, fps, angle_rad
                )
                draw_overlay(frame_cropped, center, offset_x, offset_y, ellipse, fps, angle_rad)
                move_motors(speed_m1, speed_m2, speed_m3, speed_m4)
            else:
                draw_overlay(frame_cropped, None, 0.0, 0.0, ellipse, fps, angle_rad)
                stop_all()

            # ロギング処理
            if is_logging and (current_time - last_log_time >= LOG_INTERVAL):
                elapsed_time = current_time - start_log_time
                log_to_csv("log.csv", elapsed_time, mouse_x, mouse_y, angle_rad if angle_rad is not None else -1.0)
                last_log_time = current_time

            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not is_logging:
                print("Logging started.")
                is_logging = True
                mouse_callback(0, 0)
                start_log_time = current_time
                last_log_time = current_time

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()