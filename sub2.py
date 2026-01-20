#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main.py
# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）
#
# ✅ 이번 적용 사항(요청한 2-B까지만)
# 1) Z 센서: y가 아니라 x 사용 (센서2의 x값으로 z 이동거리(mm) 측정)
# 2) (2-B) 카메라 버퍼 최소화: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#
# ✅ 유지(원래대로)
# - PID/클램프/제어 로직: 원래 main 프로그램 방식 유지 (카메라 타원피팅 각도 z_smooth 기반 ω 제어)
# - 트리거: 원점 기준 r>=45mm 도달 시 Arduino에 '1' 1회 송신
# - 센서2는 제어에는 사용하지 않고, z 이동거리(mm) -> 구체(직경300mm)로 각도 변환하여 CSV에 추가 저장

import csv
import math
import time
from datetime import datetime

import cv2
import numpy as np
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

import serial

# =========================
# Arduino
# =========================
ARDUINO_PORT = "/dev/arduinoMega"
ARDUINO_BAUD = 9600

# =========================
# ログ設定
# =========================
RUN_DURATION = 300.0
LOG_FILENAME = f"step_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_EVERY_FRAME = True
LOG_INTERVAL = 0.01

# =========================
# 트리거 (원점 기준 45mm)
# =========================
ENABLE_RADIUS_TRIGGER = True
TRIGGER_R_MM = 45.0

# =========================
# カメラと表示設定
# =========================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1

CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x,y)

PIXEL_MM = 63.0 / 480.0

# =========================
# MouseTracker (by-id 고정)
# =========================
# XY 센서(벌레 이동)
DEVICE_PATH_XY = "/dev/input/event9"
SCALING_XY = 0.0172  # mm

# Z 센서(구체 z 이동거리 측정용) - ✅ x만 사용
DEVICE_PATH_Z = "/dev/input/event13"
SCALING_Z = 0.0224   # mm


# =========================
# PID 게인(원래대로)
# =========================
Kp_x = 4.0
Ki_x = 0.0
Kd_x = 0.1
I_LIM_X = 50.0

Kp_y = 11.0
Ki_y = 0.05
Kd_y = 0.1
I_LIM_Y = 50.0

Kp_z = 15.5
Ki_z = 0.2
Kd_z = 0.3
I_LIM_Z = 20.0

Z_BASE = math.radians(90.0)

ANGLE_ALPHA = 0.2
MAX_LOST_SEC = 0.3

# =========================
# 車輪配分/スケール(원래대로)
# =========================
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN = 10.0

CMD_MAX = 127
SPEED_TO_CMD = 0.5

# =========================
# 구체(직경 300mm) -> 반지름
# =========================
SPHERE_DIAMETER_MM = 300.0
SPHERE_RADIUS_MM = SPHERE_DIAMETER_MM / 2.0  # 150mm


# =========================
# 유틸
# =========================
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ref_xyz(t_sec: float):
    xref = 0.0
    yref = 0.0
    zref = wrap_pi(Z_BASE)
    return xref, yref, zref


# =========================
# 카메라 유틸
# =========================
def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cv2.namedWindow("Track", flags=cv2.WINDOW_GUI_NORMAL)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))

    # ✅ (2-B) 카메라 버퍼 최소화: 지연 프레임 방지
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

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
        angle_deg = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]:
            angle_deg += 90.0
        angle_deg = angle_deg % 180.0
        angle_rad = math.radians(angle_deg)
        return ellipse, angle_rad
    return None, None


# =========================
# 휠 배분/구동
# =========================
def wheels_command_from_v(vx, vy, omega):
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds, omega):
    motor_m1(int(cmds[2] + (R_SPIN * omega)))
    motor_m2(int(cmds[1] - (R_SPIN * omega)))
    motor_m3(int(cmds[0] + (R_SPIN * omega)))


# =========================
# MouseTracker 콜백
# =========================
mouse_x, mouse_y = 0.0, 0.0
z_mm_raw = 0.0  # ✅ Z 센서의 x값(스케일 적용 후) 저장

def mouse_callback_xy(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def mouse_callback_z(x, y):
    global z_mm_raw
    z_mm_raw = x  # ✅ y가 아니라 x 사용


# =========================
# CSV
# =========================
def initialize_csv_logger(filename=LOG_FILENAME):
    with open(filename, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time[s]",
                "dt[s]",
                "fps",
                "ex[mm]",
                "ey[mm]",
                "ix",
                "iy",
                "vx",
                "vy",
                "omega",
                "cmd1",
                "cmd2",
                "cmd3",
                "z_cam_meas[rad]",
                "z_ref[rad]",
                "ez[rad]",
                "z_i",
                "dz[rad/s]",
                "lost_sec",
                # 센서 기반 z 이동거리(mm) -> 각도
                "z_sensor_mm",
                "z_sensor_theta[rad]",
                "z_sensor_theta[deg]",
                # 원점/트리거
                "mouse_x",
                "mouse_y",
                "r_from_origin[mm]",
                "arduino_fired",
            ]
        )

def flush_log_entries(filename, entries):
    if not entries:
        return
    with open(filename, mode="a", newline="") as f:
        csv.writer(f).writerows(entries)
    entries.clear()


# =========================
# 오버레이
# =========================
def draw_overlay(frame, center, ex_math, ey_math, ellipse, fps, z_smooth, z_ref,
                 is_running, elapsed, lost_sec, r_mm, fired, z_sensor_mm, z_sensor_deg):
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)

    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"e[mm]=({ex_math:+.2f},{ey_math:+.2f})", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    if z_smooth is not None:
        cv2.putText(frame, f"Angle(meas,CAM): {math.degrees(z_smooth):.2f} deg", (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Angle(ref) : {math.degrees(z_ref):.2f} deg", (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(frame, f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s  lost={lost_sec:.2f}s",
                (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 0), 2)

    cv2.putText(frame, f"r={r_mm:6.2f}mm thr={TRIGGER_R_MM:.1f} fired={int(fired)}",
                (10, 180), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 100, 200), 2)

    cv2.putText(frame, f"Zsensor(x): {z_sensor_mm:+.2f}mm => {z_sensor_deg:+.2f}deg",
                (10, 210), cv2.FONT_HERSHEY_COMPLEX, 0.55, (80, 80, 80), 2)


# =========================
# 메인
# =========================
if __name__ == "__main__":
    # Arduino
    ser = None
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0)
        print(f"✅ Arduino opened: {ARDUINO_PORT}")
    except Exception as e:
        print(f"⚠ Arduino open failed: {e}")
        ser = None

    # MouseTracker start (XY + Z)
    mouse_tracker_xy = MouseTracker(DEVICE_PATH_XY, SCALING_XY)
    mouse_tracker_xy.start(callback=mouse_callback_xy)
    print(f"✅ MouseTracker XY: {DEVICE_PATH_XY}")

    mouse_tracker_z = MouseTracker(DEVICE_PATH_Z, SCALING_Z)
    mouse_tracker_z.start(callback=mouse_callback_z)
    print(f"✅ MouseTracker Z : {DEVICE_PATH_Z} (x-only, distance log)")

    # CSV init
    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("❌ カメラが開けない．")
        mouse_tracker_xy.stop()
        mouse_tracker_z.stop()
        if ser and ser.is_open:
            ser.close()
        raise SystemExit(1)

    # PID states
    ix = iy = 0.0
    ex_prev = ey_prev = None
    z_i = 0.0
    z_prev = None
    z_smooth = None

    last_angle_seen_time = None
    lost_sec = 0.0

    is_running = False
    start_time = None
    last_log_time = 0.0
    prev_time = time.time()

    # 원점/트리거 (센서 XY)
    origin_set = False
    x0 = 0.0
    y0 = 0.0
    arduino_fired = False

    # ✅ 센서 Z 기준점 (x)
    z0_mm = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = clamp(now - prev_time, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]

            # 카메라 기반 처리
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # 카메라 기반 각도 추정 (z_smooth)
            if angle_rad is not None:
                z_raw = wrap_pi(angle_rad)
                if z_smooth is None:
                    z_smooth = z_raw
                else:
                    z_smooth = wrap_pi(z_smooth + ANGLE_ALPHA * wrap_pi(z_raw - z_smooth))
                last_angle_seen_time = now
                lost_sec = 0.0
            else:
                if last_angle_seen_time is None:
                    lost_sec = 1e9
                else:
                    lost_sec = now - last_angle_seen_time

            elapsed = 0.0 if not is_running else (now - start_time)

            # 참조
            if is_running:
                xref_math, yref_math, z_ref = ref_xyz(elapsed)
            else:
                xref_math, yref_math, z_ref = 0.0, 0.0, wrap_pi(Z_BASE)

            # 카메라 좌표계
            xref_cam = xref_math
            yref_cam = -yref_math

            if center is not None:
                x_cam = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_cam = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
                ex_cam = x_cam - xref_cam
                ey_cam = y_cam - yref_cam
            else:
                ex_cam = ey_cam = 0.0

            # 표시용
            ex_math = ex_cam
            ey_math = -ey_cam

            # ✅ 센서 기반 z 이동거리(mm) -> 각도 변환
            z_sensor_mm = (z_mm_raw - z0_mm) if origin_set else 0.0
            z_sensor_theta = z_sensor_mm / SPHERE_RADIUS_MM
            z_sensor_deg = math.degrees(z_sensor_theta)

            # 원점 기준 거리 r (센서 XY)
            if origin_set:
                dx0 = mouse_x - x0
                dy0 = mouse_y - y0
                r_mm = math.sqrt(dx0 * dx0 + dy0 * dy0)
            else:
                r_mm = 0.0

            # 트리거
            if ENABLE_RADIUS_TRIGGER and is_running and origin_set and (not arduino_fired) and (r_mm >= TRIGGER_R_MM):
                if ser and ser.is_open:
                    ser.write(b"1")
                    ser.flush()
                    print(f"[Arduino] Sent '1' at r={r_mm:.2f} mm (thr={TRIGGER_R_MM:.1f})")
                arduino_fired = True

            # ===== 제어 =====
            if is_running:
                # xy PID
                ix = clamp(ix + ex_cam * dt, -I_LIM_X, I_LIM_X)
                iy = clamp(iy + ey_cam * dt, -I_LIM_Y, I_LIM_Y)

                if ex_prev is None:
                    dex = 0.0
                    dey = 0.0
                else:
                    dex = (ex_cam - ex_prev) / dt
                    dey = (ey_cam - ey_prev) / dt
                ex_prev, ey_prev = ex_cam, ey_cam

                vx = Kp_x * ex_cam + Ki_x * ix + Kd_x * dex
                vy = Kp_y * ey_cam + Ki_y * iy + Kd_y * dey

                # z PID (카메라 기반)
                if (z_smooth is None) or (lost_sec > MAX_LOST_SEC):
                    ez = 0.0
                    dz_val = 0.0
                    omega = 0.0
                else:
                    ez = wrap_pi(z_ref - z_smooth)
                    z_i = clamp(z_i + ez * dt, -I_LIM_Z, I_LIM_Z)
                    dz_val = 0.0 if z_prev is None else wrap_pi(z_smooth - z_prev) / dt
                    omega = Kp_z * ez + Ki_z * z_i - Kd_z * dz_val
                    z_prev = z_smooth

                cmds, _ = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds, omega)

                # logging
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    log_entries.append(
                        [
                            f"{elapsed:.6f}",
                            f"{dt:.6f}",
                            f"{fps:.3f}",
                            f"{ex_cam:.6f}",
                            f"{ey_math:.6f}",
                            f"{ix:.6f}",
                            f"{iy:.6f}",
                            f"{vx:.6f}",
                            f"{vy:.6f}",
                            f"{omega:.6f}",
                            int(cmds[0]),
                            int(cmds[1]),
                            int(cmds[2]),
                            f"{(z_smooth if z_smooth is not None else 0.0):.6f}",
                            f"{z_ref:.6f}",
                            f"{ez:.6f}",
                            f"{z_i:.6f}",
                            f"{dz_val:.6f}",
                            f"{lost_sec:.6f}",
                            f"{z_sensor_mm:.6f}",
                            f"{z_sensor_theta:.6f}",
                            f"{z_sensor_deg:.6f}",
                            f"{mouse_x:.6f}",
                            f"{mouse_y:.6f}",
                            f"{r_mm:.6f}",
                            int(arduino_fired),
                        ]
                    )
                    last_log_time = now

                if elapsed >= RUN_DURATION:
                    print(f"{RUN_DURATION:.1f}秒経過．自動停止しCSVをフラッシュします．")
                    flush_log_entries(LOG_FILENAME, log_entries)
                    stop_all()
                    is_running = False

            else:
                stop_all()
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None

            # overlay
            draw_overlay(
                frame_cropped,
                center,
                ex_math,
                ey_math,
                ellipse,
                fps,
                z_smooth,
                z_ref,
                is_running,
                elapsed,
                lost_sec,
                r_mm,
                arduino_fired,
                z_sensor_mm,
                z_sensor_deg,
            )

            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord("q"):
                if log_entries:
                    flush_log_entries(LOG_FILENAME, log_entries)
                break

            elif key == ord("s") and not is_running:
                print(f"駆動とログを開始します（{RUN_DURATION:.1f}秒）．")

                # 원점 설정(XY + Z)
                x0 = mouse_x
                y0 = mouse_y
                z0_mm = z_mm_raw
                origin_set = True
                arduino_fired = False

                # PID reset
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None

                start_time = now
                last_log_time = now
                is_running = True

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()

        try:
            mouse_tracker_xy.stop()
        except Exception:
            pass
        try:
            mouse_tracker_z.stop()
        except Exception:
            pass

        flush_log_entries(LOG_FILENAME, log_entries)
        print("プログラム終了．")

        if ser and ser.is_open:
            ser.close()
