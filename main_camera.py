import time
import cv2
import numpy as np
import math
import csv
from datetime import datetime
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, motor_m4, stop_all

# ----------------------- 설정 -----------------------
CAMERA_INDEX = 1
WIDTH, HEIGHT = 640, 480
FPS = 30
DT = 1.0 / FPS
PIXEL = 63.0 / 480.0
WAIT = 1

K_P, K_I, K_D = 0.65, 0.35, 0.06

DEVICE_PATH = '/dev/input/event9'
SCALING = 0.0172

CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
frame_center = (HEIGHT // 2, HEIGHT // 2)

prev_offset_x = 0.0
prev_offset_y = 0.0
integral_p = 0.0
integral_n = 0.0
mouse_x, mouse_y = 0.0, 0.0

# -------------------- CSV 설정 ----------------------
LOG_INTERVAL = 0.1
log_entries = []
last_log_time = 0

def initialize_csv_logger(filename="mouse_log.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "mouse_x[mm]", "mouse_y[mm]"])

def log_to_csv(filename, x, y):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entries.append([now, f"{x:.2f}", f"{y:.2f}"])

def flush_log_entries(filename="mouse_log.csv"):
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_entries)

# -------------------- 마우스 콜백 -------------------
def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

# -------------------- 카메라 ------------------------
def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
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
        return ellipse, math.radians(angle)
    return None, None

def pid_controller(Kp, Ki, Kd, position, prev_position, integral):
    integral += position * DT
    derivative = (position - prev_position) / DT
    output = Kp * position + Ki * integral + Kd * derivative
    return output, integral

def calculate_offset(center, angle_rad):
    global prev_offset_x, prev_offset_y, integral_p, integral_n

    if angle_rad is None:
        angle_rad = 0.0

    offset_x = (center[0] - frame_center[0]) * PIXEL
    offset_y = (frame_center[1] - center[1]) * PIXEL
    offset_z = 300 * math.cos(angle_rad) / 4

    position_p = offset_x + offset_y
    position_n = offset_x - offset_y
    prev_position_p = (prev_offset_x + prev_offset_y)
    prev_position_n = (prev_offset_x - prev_offset_y)

    output_p, integral_p = pid_controller(K_P, K_I, K_D, position_p, prev_position_p, integral_p)
    output_n, integral_n = pid_controller(K_P, K_I, K_D, position_n, prev_position_n, integral_n)

    speed_m1 = output_p + offset_z
    speed_m2 = output_n + offset_z
    speed_m3 = -output_n + offset_z
    speed_m4 = -output_p + offset_z

    prev_offset_x = offset_x
    prev_offset_y = offset_y

    return offset_x, offset_y, speed_m1, speed_m2, speed_m3, speed_m4

# ------------------- 메인 루프 ----------------------
if __name__ == "__main__":
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger("mouse_log.csv")

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない")
        mouse_tracker.stop()
        exit()

    prev_time = time.time()

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
                offset_x, offset_y, s1, s2, s3, s4 = calculate_offset(center, angle_rad)

                motor_m1(int(s1))
                motor_m2(int(s2))
                motor_m3(int(s3))
                motor_m4(int(s4))

                cv2.circle(frame_cropped, center, 5, (0, 255, 0), -1)
                cv2.putText(frame_cropped, f"(x,y)=({offset_x:.2f},{offset_y:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

                if angle_rad is not None:
                    angle_deg = math.degrees(angle_rad)
                    cv2.putText(frame_cropped, f"Angle: {angle_deg:.2f} deg", (10, 90),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
            else:
                stop_all()

            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame_cropped, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame_cropped, f"Mouse: ({mouse_x:.2f}, {mouse_y:.2f})", (10, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

            if current_time - last_log_time >= LOG_INTERVAL:
                log_to_csv("mouse_log.csv", mouse_x, mouse_y)
                last_log_time = current_time

            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                break

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        flush_log_entries("mouse_log.csv")
        print("プログラム終了")
