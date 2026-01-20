# main.py
# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）．
# 変更点：
# - 's' で駆動開始（それまでPIDの積分・微分は計算しない）．
# - 駆動開始から10秒間のステップ応答をCSV保存．
# - 10秒経過で自動停止＆CSVフラッシュ．

import csv
import math
import time
from datetime import datetime

import cv2
import numpy as np
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# >>> ADD: Arduino シリアル送信用
import serial
ARDUINO_PORT = "/dev/arduinoMega"  # 例: /dev/ttyUSB0, COM3など環境に合わせて
ARDUINO_BAUD = 9600

# ログ設定
RUN_DURATION = 600  # 計測時間10秒
# RUN_DURATION = 600.0  # 計測時間`10分（デバッグ用）`

LOG_FILENAME = "step_response.csv"  # ステップ応答保存先
LOG_EVERY_FRAME = True  # Trueなら毎フレーム，FalseならLOG_INTERVALで間引き
LOG_INTERVAL = 0.01  # LOG_EVERY_FRAME=Falseのときに使用

# カメラと表示設定
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1  # cv2.waitKey用

# クロップ（中央正方領域を抽出）
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x,y)

# 画素→mm 変換
PIXEL_MM = 63.0 / 480.0


# マウス軌跡
DEVICE_PATH = "/dev/input/event9"
SCALING = 0.0172  # mm単位

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
PSI_REF = 0.0  # 目標姿勢角［rad］

# ψ_ref(t) = 90deg + AMP_Z*sin(Ω_z t)
AMP_Z = math.radians(90.0)  # 15°
OMEGA_Z = math.pi / 5.0  # [rad/s]（周期 10 s）
Z_BASE = math.radians(90.0)  # 90°

# 角度の指数移動平均/検出喪失処理
ANGLE_ALPHA = 0.2
MAX_LOST_SEC = 0.3

# ===== 車輪配分/スケール =====
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN = 10.0

CMD_MAX = 80
SPEED_TO_CMD = 0.5  # v_wheels[mm/s] → コマンド

# ===== 参照（x,y: 円運動 / z: サイン目標）=====
AMP_MM = 0.0
OMEGA = math.pi / 2.0
X_BIAS = 0.0
Y_BIAS = 0.0


def ref_xyz(t_sec: float):
    xref = X_BIAS
    yref = Y_BIAS
    zref = wrap_pi(Z_BASE)
    return xref, yref, zref


# =========================
# 車輪配分と出力スケール
# =========================
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN = 10.0

CMD_MAX = 127
SPEED_TO_CMD = 0.5  # v_wheels[mm/s] → コマンド

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


def draw_overlay(
    frame,
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
):
    # 原点/誤差
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)
    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"e[mm]=({ex_math:+.2f},{ey_math:+.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
    # 角度表示
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
    # 状態表示
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2
    )
    cv2.putText(
        frame,
        f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
        (10, 150),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    


def wheels_command_from_v(vx, vy, omega):
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels


def move_motors_cmds(cmds, omega):
    motor_m1(int(cmds[2] + (R_SPIN * omega)))
    motor_m2(int(cmds[1] - (R_SPIN * omega)))
    motor_m3(int(cmds[0] + (R_SPIN * omega)))


def initialize_csv_logger(filename=LOG_FILENAME):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time[s]",
                "dt[s]",
                "fps",
                "ex[mm]",
                "ey[mm]",
                "ix",
                "iy",
                "cmd1",
                "cmd2",
                "cmd3",
            ]
        )


def log_to_csv_buffer(buf, **kw):
    # kw から列順に抽出して追加
    buf.append(
        [
            f"{kw['t']:.6f}",
            f"{kw['dt']:.6f}",
            f"{kw['fps']:.3f}",
            f"{kw['ex']:.6f}",
            f"{kw['ey']:.6f}",
            f"{kw['vx']:.6f}",
            f"{kw['vy']:.6f}",
            f"{kw['omega']:.6f}",
            int(kw["cmds"][0]),
            int(kw["cmds"][1]),
            int(kw["cmds"][2]),
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
# メイン
# =========================
if __name__ == "__main__":
    # >>> ADD: Arduino シリアルオープン
    ser = None
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0)

    # マウストラッカー開始
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        mouse_tracker.stop()
        # >>> ADD: シリアルクローズ
        if ser and ser.is_open:
            ser.close()
        exit()

    # PID状態
    ix = iy = 0.0
    ex_prev = ey_prev = None
    z_i = 0.0
    z_prev = None
    z_smooth = None
    # 角度検出喪失管理
    last_angle_seen_time = None
    lost_sec = 0.0

    # ===== 実行・計測フラグ =====
    is_running = False  # 's' 押下で True
    start_time = None  # 駆動開始時刻
    last_log_time = 0.0
    prev_time = time.time()

    # >>> ADD: 29秒で一度だけ'1'送信するフラグ
    arduino_fired = False

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
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # 角度の前処理
            if angle_rad is not None:
                z_raw = wrap_pi(angle_rad)
                if z_smooth is None:
                    z_smooth = z_raw
                else:
                    z_smooth = wrap_pi(
                        z_smooth + ANGLE_ALPHA * wrap_pi(z_raw - z_smooth)
                    )
                last_angle_seen_time = now
                lost_sec = 0.0
            else:
                if last_angle_seen_time is None:
                    lost_sec = 1e9
                else:
                    lost_sec = now - last_angle_seen_time
            # 経過時間
            elapsed = 0.0 if not is_running else (now - start_time)
            # 参照生成（xyz）
            if is_running:
                xref_math, yref_math, z_ref = ref_xyz(elapsed)
            else:
                xref_math, yref_math, z_ref = 0.0, 0.0, Z_BASE

            # カメラ座標系（右+, 下+）
            xref_cam = xref_math
            yref_cam = -yref_math

            if center is not None:
                x_cam = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_cam = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
                ex_cam = x_cam - xref_cam
                ey_cam = y_cam - yref_cam
            else:
                ex_cam = ey_cam = 0.0

            # 表示用（上+）
            ex_math = ex_cam
            ey_math = -ey_cam

            # ===== 制御計算 =====
            if is_running:
                # --- xy位置PID ---
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

                # >>> ADD: 計測開始から29秒で一回だけ '1' を送信
                if (not arduino_fired) and (elapsed >= 29.0):
                    if ser and ser.is_open:
                        ser.write(b'1')
                        ser.flush()
                        print("[Arduino] Sent '1' at t=29s")
                    arduino_fired = True

                # 各輪コマンド
                cmds, vwh = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds,omega)

                # ロギング（毎フレームまたは間引き）
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    log_to_csv_buffer(
                        log_entries,
                        t=elapsed,
                        dt=dt,
                        fps=fps,
                        ex=ex_cam,
                        ey=ey_math,
                        dex=dex,
                        dey=dey,
                        ix=ix,
                        iy=iy,
                        z_meas=(z_smooth if z_smooth is not None else 0.0),
                        z_ref=z_ref,
                        ez=ez,
                        z_i=z_i,
                        dz=dz_val,
                        vx=vx,
                        vy=vy,
                        omega=omega,
                        cmds=cmds,
                        vwh=vwh,
                        lost_sec=lost_sec,
                    )
                    last_log_time = now

                # 10秒経過で停止
                if elapsed >= RUN_DURATION:
                    print("10秒経過．自動停止しCSVをフラッシュします．")
                    flush_log_entries(LOG_FILENAME, log_entries)
                    stop_all()
                    is_running = False  # 停止してIDLEへ戻す

            else:
                # IDLE：出力ゼロ，PID積分・微分は計算しない
                stop_all()
                # 開始前は積分・微分状態を保持せず常に初期化
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None
                # vx,vy,omega,cmdsは表示用に0
                vx = vy = omega = 0.0
                cmds = np.array([0, 0, 0], dtype=int)
                dex = dey = 0.0

            # オーバレイ描画
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
            )

            # 画面表示とキー
            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord("q"):
                # 強制終了
                if log_entries:
                    flush_log_entries(LOG_FILENAME, log_entries)
                break

            elif key == ord("s") and not is_running:
                # ===== ステップ応答開始 =====
                print("駆動とログを開始します（10秒）．")
                # 状態リセット（開始時の微分を0にする）
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None
                start_time = now
                last_log_time = now
                is_running = True

                # >>> ADD: 29秒ワンショット送信のリセット
                arduino_fired = False

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        flush_log_entries(LOG_FILENAME, log_entries)
        print("プログラム終了．")
        # >>> ADD: シリアルクローズ
        if ser and ser.is_open:
            ser.close()
