# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）
# ランダム M[?,?,?] 励振バージョン
#   M = 100 → x축 : gain∈[-1,1] * AMP_X_MM
#   M = 010 → y축 : gain∈[-1,1] * AMP_Y_MM
#   M = 001 → z축 : gain∈[-1,1] * AMP_Z_DEG
# 1秒ごとに参照が更新される

import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# ===== 実験/ログ =====
RUN_DURATION    = 10.0
LOG_FILENAME    = f"step_response_xyz_randomM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_EVERY_FRAME = True
LOG_INTERVAL    = 0.01

# ===== カメラ/表示 =====
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS_TARGET    = 60
WAIT          = 1

# ===== クロップ（中央正方領域） =====
CROP_LEFT  = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)

# 画素→mm
PIXEL_MM = 63.0 / 480.0

# ===== マウス入力 =====
DEVICE_PATH = '/dev/input/event9'
SCALING     = 0.0172

# ===== PIDゲイン =====
Kp_x = 7.0;  Ki_x = 0.0; Kd_x = 0.1; I_LIM_X = 40.0
Kp_y = 14.0; Ki_y = 0.2; Kd_y = 0.3; I_LIM_Y = 40.0
Kp_z = 19.0; Ki_z = 0.4; Kd_z = 0.4; I_LIM_Z = 20.0

Z_BASE = math.radians(90.0)

ANGLE_ALPHA  = 0.2
MAX_LOST_SEC = 0.3

THETA   = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], float)
R_SPIN  = 10.0
CMD_MAX = 127
SPEED_TO_CMD = 0.5

# ===== ランダム参照 =====
EXCITE_INTERVAL = 1.0   # 1초마다 새로운 랜덤 자극
AMP_X_MM   = 15.0
AMP_Y_MM   = 15.0
AMP_Z_DEG  = 15.0       # 최고 ±15deg까지

# ===================================================
# Utility
# ===================================================
mouse_x, mouse_y = 0.0, 0.0
def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    return cap if cap.read()[0] else None

def gray_binary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return c, None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return c, (cx, cy)

def fit_ellipse_if_possible(contour):
    if contour is not None and len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        angle_deg = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]:
            angle_deg += 90
        angle_deg %= 180
        return ellipse, math.radians(angle_deg)
    return None, None

def wheels_command_from_v(vx, vy, omega):
    v_wheels = (-vx * np.sin(THETA) + vy*np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds, omega):
    motor_m1(int(cmds[2] + (R_SPIN * omega)))
    motor_m2(int(cmds[1] - (R_SPIN * omega)))
    motor_m3(int(cmds[0] + (R_SPIN * omega)))

# ===================================================
# CSV
# ===================================================
def initialize_csv_logger(filename):
    with open(filename,"w",newline="") as f:
        csv.writer(f).writerow([
            "time","dt","fps",
            "xref","yref","zref","Mx","My","Mz",
            "ex","ey","dex","dey","ix","iy",
            "z_meas","ez","z_i","dz",
            "PX","PY","PZ","IX","IY","IZ","DX","DY","DZ",
            "vx","vy","omega",
            "cmd1","cmd2","cmd3",
            "vwh1","vwh2","vwh3",
            "lost_sec"
        ])

def log_to_csv_buffer(buf, **k):
    buf.append([
        k["t"],k["dt"],k["fps"],
        k["xref"],k["yref"],k["zref"],k["Mx"],k["My"],k["Mz"],
        k["ex"],k["ey"],k["dex"],k["dey"],k["ix"],k["iy"],
        k["z_meas"],k["ez"],k["z_i"],k["dz"],
        k["PX"],k["PY"],k["PZ"],k["IX"],k["IY"],k["IZ"],
        k["DX"],k["DY"],k["DZ"],
        k["vx"],k["vy"],k["omega"],
        int(k["cmds"][0]),int(k["cmds"][1]),int(k["cmds"][2]),
        k["vwh"][0],k["vwh"][1],k["vwh"][2],
        k["lost_sec"]
    ])

def flush_log_entries(filename, entries):
    if not entries:
        return
    with open(filename,"a",newline="") as f:
        csv.writer(f).writerows(entries)
    entries.clear()

# ===================================================
# Main
# ===================================================
if __name__ == "__main__":

    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("Camera Error")
        mouse_tracker.stop()
        exit()

    # PID 상태
    ix = iy = 0.0
    ex_prev = ey_prev = None
    z_i = 0.0
    z_prev = None
    z_smooth = None

    last_angle_seen_time = None
    lost_sec = 0.0

    # 랜덤 참조 상태
    xref = 0.0
    yref = 0.0
    zref = Z_BASE
    Mx = My = Mz = 0
    last_excite_time = 0.0

    is_running = False
    start_time = None
    last_log_time = 0.0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = clamp(now - prev_time, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            # === 이미지 처리 ===
            frame_c = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_c)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # === angle smoothing ===
            if angle_rad is not None:
                z_raw = wrap_pi(angle_rad)
                if z_smooth is None:
                    z_smooth = z_raw
                else:
                    z_smooth = wrap_pi(z_smooth + ANGLE_ALPHA * wrap_pi(z_raw - z_smooth))
                last_angle_seen_time = now
                lost_sec = 0.0
            else:
                lost_sec = now - last_angle_seen_time if last_angle_seen_time else 1e9

            elapsed = 0 if not is_running else (now - start_time)

            # ======================================================
            # 랜덤 참조 생성 (gain ∈ [-1, 1])
            # ======================================================
            if not is_running:
                xref = yref = 0.0
                zref = Z_BASE
                Mx = My = Mz = 0
            else:
                if (now - last_excite_time) >= EXCITE_INTERVAL:
                    axis = np.random.randint(0,3)           # 0:x, 1:y, 2:z
                    gain = float(np.random.uniform(-1.0, 1.0))  # -1 ~ +1

                    if axis == 0:
                        # x 방향 자극
                        xref = gain * AMP_X_MM
                        yref = 0.0
                        zref = Z_BASE
                        Mx, My, Mz = 1, 0, 0
                    elif axis == 1:
                        # y 방향 자극
                        xref = 0.0
                        yref = gain * AMP_Y_MM
                        zref = Z_BASE
                        Mx, My, Mz = 0, 1, 0
                    else:
                        # z 방향 자극 (deg 범위만 그대로, 내부는 rad)
                        xref = 0.0
                        yref = 0.0
                        zref = wrap_pi(Z_BASE + math.radians(gain * AMP_Z_DEG))
                        Mx, My, Mz = 0, 0, 1

                    last_excite_time = now

            # 카메라계 참조 (오른쪽+, 아래+)
            xref_cam = xref
            yref_cam = -yref

            # 실제 중심 → 오차(카메라계)
            if center is not None:
                x_cam = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_cam = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
                ex_cam = x_cam - xref_cam
                ey_cam = y_cam - yref_cam
            else:
                ex_cam = ey_cam = 0.0

            # 수학 좌표계(오른쪽+, 위+), 표시용
            ex = ex_cam
            ey = -ey_cam

            # 초기값
            vx = vy = omega = 0.0
            dex = dey = 0.0
            ez = dz_val = 0.0
            z_meas = z_smooth if z_smooth is not None else 0.0

            if is_running:

                # ===== x/y PID =====
                ix = clamp(ix + ex_cam * dt, -I_LIM_X, I_LIM_X)
                iy = clamp(iy + ey_cam * dt, -I_LIM_Y, I_LIM_Y)

                if ex_prev is None:
                    dex = dey = 0.0
                else:
                    dex = (ex_cam - ex_prev) / dt
                    dey = (ey_cam - ey_prev) / dt
                ex_prev, ey_prev = ex_cam, ey_cam

                vx = Kp_x*ex_cam + Ki_x*ix + Kd_x*dex
                vy = Kp_y*ey_cam + Ki_y*iy + Kd_y*dey

                PX, PY, PZ = Kp_x, Kp_y, Kp_z
                IX, IY, IZ = Ki_x, Ki_y, Ki_z
                DX, DY, DZ = Kd_x, Kd_y, Kd_z

                # ===== z PID =====
                if lost_sec > MAX_LOST_SEC:
                    ez = dz_val = 0.0
                    omega = 0.0
                    z_i = 0.0
                    z_prev = None
                else:
                    ez = wrap_pi(zref - z_meas)
                    z_i = clamp(z_i + ez * dt, -I_LIM_Z, I_LIM_Z)
                    dz_val = 0.0 if z_prev is None else wrap_pi(z_meas - z_prev) / dt
                    omega = Kp_z*ez + Ki_z*z_i - Kd_z*dz_val
                    z_prev = z_meas

                # ===== 휠 배분 =====
                vwh = (-vx * np.sin(THETA) + vy*np.cos(THETA)) * DIR_SGN
                cmds = np.clip(vwh * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
                move_motors_cmds(cmds, omega)

                # ===== CSV =====
                if LOG_EVERY_FRAME or (now-last_log_time) >= LOG_INTERVAL:
                    log_to_csv_buffer(
                        log_entries,
                        t=elapsed, dt=dt, fps=fps,
                        xref=xref, yref=yref, zref=zref,
                        Mx=Mx, My=My, Mz=Mz,
                        ex=ex, ey=ey, dex=dex, dey=dey,
                        ix=ix, iy=iy,
                        z_meas=z_meas, ez=ez, z_i=z_i, dz=dz_val,
                        PX=PX, PY=PY, PZ=PZ,
                        IX=IX, IY=IY, IZ=IZ,
                        DX=DX, DY=DY, DZ=DZ,
                        vx=vx, vy=vy, omega=omega,
                        cmds=cmds, vwh=vwh,
                        lost_sec=lost_sec
                    )
                    last_log_time = now

                if elapsed >= RUN_DURATION:
                    print("10秒終了・保存中…")
                    flush_log_entries(LOG_FILENAME, log_entries)
                    stop_all()
                    is_running = False

            else:
                stop_all()
                ex_prev = ey_prev = None
                ix = iy = 0.0
                z_i = 0.0
                z_prev = None

            # === Overlay (간단 버전) ===
            cv2.circle(frame_c, FRAME_CENTER, 5, (255,0,0), -1)
            cv2.putText(frame_c, f"M={Mx}{My}{Mz}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame_c, f"t={elapsed:.2f}s", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            cv2.imshow("Track", frame_c)

            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                flush_log_entries(LOG_FILENAME, log_entries)
                break
            elif key == ord('s') and not is_running:
                print("Start Random-M Test with gain in [-1,1]!")
                is_running   = True
                start_time   = now
                last_excite_time = now
                last_log_time    = now
                xref = yref = 0.0
                zref = Z_BASE
                Mx = My = Mz = 0

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        flush_log_entries(LOG_FILENAME, log_entries)
        print("終了")
