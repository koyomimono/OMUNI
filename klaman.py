# main_kalman.py
# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）．
# 追加点：
# - x,y の計測位置にカルマンフィルタ（状態: [pos, vel]）を適用可能．
# - USE_KALMAN の True/False で KF 有無を切替．
# - STEP_AXIS（"x" or "y"）で 10 mm ステップ方向を切替．
# - 's' 押下で駆動開始，RUN_DURATION 秒で自動停止してCSVフラッシュ．

import csv
import math
import time
from datetime import datetime

import cv2
import numpy as np
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# =========================
# 計測条件（4回分はここを切替）
# =========================
STEP_AXIS = "x"          # "x" or "y"
STEP_MM = 10.0           # ステップ量 [mm]
USE_KALMAN = True        # True: KFあり，False: KFなし

RUN_DURATION = 10.0      # ステップ応答計測時間 [s]
LOG_EVERY_FRAME = True
LOG_INTERVAL = 0.01

# カメラと表示設定
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1

# クロップ（中央正方領域を抽出）
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x,y)

# 画素→mm 変換
PIXEL_MM = 63.0 / 480.0

# マウス軌跡（今回は使わないが元コード互換のため残す）
DEVICE_PATH = "/dev/input/event9"
SCALING = 0.0172

# =========================
# PIDゲイン
# =========================
# 位置PIDゲイン（x）
Kp_x = 4.0
Ki_x = 0.0
Kd_x = 0.1
I_LIM_X = 50.0

# 位置PIDゲイン（y）
Kp_y = 11.0
Ki_y = 0.05
Kd_y = 0.1
I_LIM_Y = 50.0

# 角度PIDゲイン（ω）
Kp_z = 15.5
Ki_z = 0.2
Kd_z = 0.3
I_LIM_Z = 20.0

# ψ_ref(t) は固定にしておく（必要なら元コード同様にサイン参照へ変更可）
Z_BASE = math.radians(90.0)

# 角度の指数移動平均/検出喪失処理
ANGLE_ALPHA = 0.2
MAX_LOST_SEC = 0.3

# =========================
# 車輪配分/スケール
# =========================
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)

CMD_MAX = 127
SPEED_TO_CMD = 0.5
R_SPIN = 10.0

# =========================
# グローバル（コールバック更新）
# =========================
mouse_x, mouse_y = 0.0, 0.0


def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


# =========================
# ユーティリティ
# =========================
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cv2.namedWindow("Track", flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        angle_deg = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]:
            angle_deg += 90.0
        angle_deg = angle_deg % 180.0
        angle_rad = math.radians(angle_deg)
        return ellipse, angle_rad
    return None, None


def draw_overlay(frame, center, ex_mm, ey_mm, fps, z_smooth, z_ref, is_running, elapsed, lost_sec):
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)
    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"e[mm]=({ex_mm:+.2f},{ey_mm:+.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    if z_smooth is not None:
        cv2.putText(
            frame,
            f"Angle(meas): {math.degrees(z_smooth):.2f} deg",
            (10, 60),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        frame,
        f"Angle(ref) : {math.degrees(z_ref):.2f} deg",
        (10, 90),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(
        frame,
        f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s  lost={lost_sec:.3f}s",
        (10, 150),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def wheels_command_from_v(vx, vy):
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels


def move_motors_cmds(cmds, omega):
    motor_m1(int(cmds[2] + (R_SPIN * omega)))
    motor_m2(int(cmds[1] - (R_SPIN * omega)))
    motor_m3(int(cmds[0] + (R_SPIN * omega)))


def make_log_filename():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "kf" if USE_KALMAN else "raw"
    return f"step_{STEP_AXIS}_{STEP_MM:.0f}mm_{mode}_{ts}.csv"


def initialize_csv_logger(filename):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time[s]",
                "dt[s]",
                "fps",
                "xref[mm]",
                "yref[mm]",
                "x_meas[mm]",
                "y_meas[mm]",
                "x_est[mm]",
                "y_est[mm]",
                "vx_est[mm/s]",
                "vy_est[mm/s]",
                "ex[mm]",
                "ey[mm]",
                "ix",
                "iy",
                "vx_cmd",
                "vy_cmd",
                "omega",
                "cmd1",
                "cmd2",
                "cmd3",
                "z_meas[rad]",
                "z_ref[rad]",
                "lost_sec",
            ]
        )


def flush_log_entries(filename, entries):
    if not entries:
        return
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(entries)
    entries.clear()


# =========================
# 参照（ステップ）
# =========================
def ref_xy_step():
    if STEP_AXIS.lower() == "x":
        return STEP_MM, 0.0
    if STEP_AXIS.lower() == "y":
        return 0.0, STEP_MM
    return 0.0, 0.0


# =========================
# 1Dカルマンフィルタ（状態: [pos, vel]）
# =========================
class Kalman1D:
    def __init__(self, sigma_a=200.0, sigma_z=1.5):
        # sigma_a: 加速度雑音の標準偏差 [mm/s^2]
        # sigma_z: 観測雑音の標準偏差 [mm]
        self.sigma_a = float(sigma_a)
        self.sigma_z = float(sigma_z)
        self.x = np.zeros((2, 1), dtype=float)   # [pos; vel]
        self.P = np.eye(2, dtype=float) * 1e3
        self.initialized = False

    def reset(self, pos, vel=0.0):
        self.x[:] = [[float(pos)], [float(vel)]]
        self.P[:] = np.eye(2, dtype=float) * 1e2
        self.initialized = True

    def predict(self, dt):
        dt = float(dt)
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)

        q = (self.sigma_a ** 2)
        Q = q * np.array(
            [[(dt ** 4) / 4.0, (dt ** 3) / 2.0],
             [(dt ** 3) / 2.0, (dt ** 2)]],
            dtype=float
        )

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1.0, 0.0]], dtype=float)
        R = np.array([[self.sigma_z ** 2]], dtype=float)

        z = np.array([[float(z)]], dtype=float)
        y = z - (H @ self.x)
        S = (H @ self.P @ H.T) + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(2, dtype=float)
        self.P = (I - K @ H) @ self.P

    @property
    def pos(self):
        return float(self.x[0, 0])

    @property
    def vel(self):
        return float(self.x[1, 0])


# =========================
# メイン
# =========================
if __name__ == "__main__":
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        mouse_tracker.stop()
        raise SystemExit(1)

    # PID状態
    ix = iy = 0.0
    ex_prev = ey_prev = None

    # 姿勢PID状態
    z_i = 0.0
    z_prev = None
    z_smooth = None
    last_angle_seen_time = None
    lost_sec = 0.0

    # KF
    kf_x = Kalman1D(sigma_a=250.0, sigma_z=1.5)
    kf_y = Kalman1D(sigma_a=250.0, sigma_z=1.5)

    # 実行・計測フラグ
    is_running = False
    start_time = None
    last_log_time = 0.0
    prev_time = time.time()

    log_filename = None
    log_entries = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = clamp(now - prev_time, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            # 画像処理
            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            _, angle_rad = fit_ellipse_if_possible(contour)

            # 角度前処理
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

            # 参照生成（math座標: 右+, 上+）
            if is_running:
                xref_math, yref_math = ref_xy_step()
                z_ref = wrap_pi(Z_BASE)
            else:
                xref_math, yref_math = 0.0, 0.0
                z_ref = wrap_pi(Z_BASE)

            # cam座標: 右+, 下+
            xref_cam = xref_math
            yref_cam = -yref_math

            # 計測（cam座標）
            meas_ok = (center is not None)
            if meas_ok:
                x_meas = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_meas = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
            else:
                x_meas = 0.0
                y_meas = 0.0

            # ===== KF更新（x,y）=====
            if USE_KALMAN:
                if (not kf_x.initialized) and meas_ok:
                    kf_x.reset(x_meas, 0.0)
                    kf_y.reset(y_meas, 0.0)
                else:
                    if kf_x.initialized:
                        kf_x.predict(dt)
                        kf_y.predict(dt)
                        if meas_ok:
                            kf_x.update(x_meas)
                            kf_y.update(y_meas)

                x_est = kf_x.pos if kf_x.initialized else x_meas
                y_est = kf_y.pos if kf_y.initialized else y_meas
                vx_est = kf_x.vel if kf_x.initialized else 0.0
                vy_est = kf_y.vel if kf_y.initialized else 0.0
            else:
                x_est, y_est = x_meas, y_meas
                vx_est, vy_est = 0.0, 0.0

            # 誤差（cam座標）
            ex_cam = x_est - xref_cam
            ey_cam = y_est - yref_cam

            # 表示用（math座標の誤差: 上+）
            ex_disp = ex_cam
            ey_disp = -ey_cam

            # ===== 制御計算 =====
            if is_running:
                # --- 積分 ---
                ix = clamp(ix + ex_cam * dt, -I_LIM_X, I_LIM_X)
                iy = clamp(iy + ey_cam * dt, -I_LIM_Y, I_LIM_Y)

                # --- 微分 ---
                if USE_KALMAN and (kf_x.initialized):
                    # 参照はステップで速度0とみなす
                    dex = vx_est
                    dey = vy_est
                else:
                    if ex_prev is None:
                        dex = 0.0
                        dey = 0.0
                    else:
                        dex = (ex_cam - ex_prev) / dt
                        dey = (ey_cam - ey_prev) / dt
                    ex_prev, ey_prev = ex_cam, ey_cam

                vx_cmd = Kp_x * ex_cam + Ki_x * ix + Kd_x * dex
                vy_cmd = Kp_y * ey_cam + Ki_y * iy + Kd_y * dey

                # --- z姿勢制御（PID） ---
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

                # 各輪コマンド
                cmds, _ = wheels_command_from_v(vx_cmd, vy_cmd)
                move_motors_cmds(cmds, omega)

                # ロギング
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log and (log_filename is not None):
                    log_entries.append(
                        [
                            f"{elapsed:.6f}",
                            f"{dt:.6f}",
                            f"{fps:.3f}",
                            f"{xref_cam:.6f}",
                            f"{yref_cam:.6f}",
                            f"{x_meas:.6f}",
                            f"{y_meas:.6f}",
                            f"{x_est:.6f}",
                            f"{y_est:.6f}",
                            f"{vx_est:.6f}",
                            f"{vy_est:.6f}",
                            f"{ex_cam:.6f}",
                            f"{ey_cam:.6f}",
                            f"{ix:.6f}",
                            f"{iy:.6f}",
                            f"{vx_cmd:.6f}",
                            f"{vy_cmd:.6f}",
                            f"{omega:.6f}",
                            int(cmds[0]),
                            int(cmds[1]),
                            int(cmds[2]),
                            f"{(z_smooth if z_smooth is not None else 0.0):.6f}",
                            f"{z_ref:.6f}",
                            f"{lost_sec:.6f}",
                        ]
                    )
                    last_log_time = now

                # 自動停止
                if elapsed >= RUN_DURATION:
                    print("RUN_DURATION 経過．自動停止しCSVをフラッシュします．")
                    stop_all()
                    if log_filename is not None:
                        flush_log_entries(log_filename, log_entries)
                    is_running = False

            else:
                # IDLE
                stop_all()
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None

                # KF状態も開始ごとにリセットしたい場合はここで解除
                kf_x.initialized = False
                kf_y.initialized = False

            # 描画
            draw_overlay(
                frame_cropped,
                center,
                ex_disp,
                ey_disp,
                fps,
                z_smooth,
                z_ref,
                is_running,
                elapsed,
                lost_sec,
            )

            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord("q"):
                if (log_filename is not None) and log_entries:
                    flush_log_entries(log_filename, log_entries)
                break

            elif key == ord("s") and (not is_running):
                # ステップ応答開始
                print("駆動とログを開始します．")
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None

                # ログファイルを run ごとに作成
                log_filename = make_log_filename()
                initialize_csv_logger(log_filename)
                log_entries.clear()

                # KF初期化
                kf_x.initialized = False
                kf_y.initialized = False

                start_time = now
                last_log_time = now
                is_running = True

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        if (log_filename is not None) and log_entries:
            flush_log_entries(log_filename, log_entries)
        print("プログラム終了．")
