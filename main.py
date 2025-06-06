import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker  # マウスの軌跡データ取得
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, motor_m4, stop_all  # モーター制御関数

# カメラ設定
CAMERA_INDEX = 1
WIDTH = 640
HEIGHT = 480
FPS = 30
DT = 1.0 / FPS
PIXEL = 63.0 / 480.0
WAIT = 1

# PID制御のゲイン
K_P = 0.65   # 比例ゲイン
K_I = 0.35   # 積分ゲイン
K_D = 0.06  # PIDゲイン

# マウス移動スケーリング（デバイスパスとスケール係数）
DEVICE_PATH = '/dev/input/event9'
SCALING = 0.0172  # mm単位

# クロップ範囲（中央の正方領域を抽出）
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
frame_center = (HEIGHT // 2, HEIGHT // 2)

# グローバル変数（位置と積分用）
mouse_x, mouse_y = 0.0, 0.0
prev_offset_x = 0.0
prev_offset_y = 0.0

# ログ設定
LOG_INTERVAL = 0.1  # 秒間隔
MAX_DURATION = 30 * 60  # 最大測定時間：30分
IDLE_THRESHOLD = 30.0   # 停止時間閾値：30秒
MOVEMENT_EPSILON = 0.5  # 停止と見なす移動距離閾値（mm）

# マウス位置記録用
prev_mouse_x, prev_mouse_y = 0.0, 0.0
last_movement_time = time.time()

# ログ一時保存リスト（後でCSV出力）
log_entries = []

# マウス位置の更新用コールバック関数
def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

# カメラ初期化関数
def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    return cap if cap.read()[0] else None

# グレースケール変換＋2値化処理
def gray_binary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return binary

# 最大輪郭とその中心点を取得
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

# 楕円フィッティング（最小5点必要）
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

# オフセットとPIDによる速度計算
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

# オーバーレイ描画関数（オフセット、角度、FPS、マウス位置）
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

# モーター速度送信
def move_motors(speed_m1, speed_m2, speed_m3, speed_m4):
    motor_m1(int(speed_m1))
    motor_m2(int(speed_m2))
    motor_m3(int(speed_m3))
    motor_m4(int(speed_m4))

# CSV初期化関数
def initialize_csv_logger(filename="log9.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "mouse_x[mm]", "mouse_y[mm]", "angle_rad"])

# 一時ログ保存（CSVに書く前のメモリ保持）
def log_to_csv(filename, elapsed_time, mouse_x, mouse_y, angle_rad):
    log_entries.append([
        f"{elapsed_time:.1f}",
        f"{mouse_x:.2f}",
        f"{mouse_y:.2f}",
        f"{angle_rad:.2f}" if angle_rad is not None else "-1.00"
    ])

# ログをファイルに書き出し
def flush_log_entries(filename):
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_entries)

# メイン処理開始
if __name__ == "__main__":
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger("log9.csv")

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

            # 停止状態の検出
            movement = math.hypot(mouse_x - prev_mouse_x, mouse_y - prev_mouse_y)
            if movement > MOVEMENT_EPSILON:
                last_movement_time = current_time
            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            # ログ記録の管理
            if is_logging:
                elapsed_time = current_time - start_log_time
                if elapsed_time >= MAX_DURATION:
                    print("30分経過のためログ停止")
                    flush_log_entries("log9.csv")
                    break
                elif current_time - last_movement_time >= IDLE_THRESHOLD:
                    print("30秒以上停止状態のためログ停止")
                    flush_log_entries("log9.csv")
                    break
                elif current_time - last_log_time >= LOG_INTERVAL:
                    log_to_csv("log9.csv", elapsed_time, mouse_x, mouse_y, angle_rad if angle_rad is not None else -1.0)
                    last_log_time = current_time

            # 画面表示とキー操作
            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                if is_logging:
                    flush_log_entries("log9.csv")
                break
            elif key == ord('s') and not is_logging:
                print("ロギング開始")
                is_logging = True
                mouse_callback(0, 0)
                start_log_time = current_time
                last_log_time = current_time
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                last_movement_time = current_time

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        if is_logging:
            flush_log_entries("log9.csv")
        print("プログラム終了")