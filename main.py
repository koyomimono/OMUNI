# main.py
# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）．
# 変更点：
# - 's' で駆動開始（それまでPIDの積分・微分は計算しない）．
# - 駆動開始から10秒間のステップ応答をCSV保存．
# - 10秒経過で自動停止＆CSVフラッシュ．

import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# =========================
# カメラと表示設定
# =========================
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

# =========================
# マウス軌跡（任意の外部センサ）
# =========================
DEVICE_PATH = '/dev/input/event9'
SCALING = 0.0172  # mm単位

# =========================
# 位置PIDゲイン（vx, vy生成）
# =========================
Kp_xy = 3.0
Ki_xy = 0.7
Kd_xy = 0.2
I_LIM_XY = 50.0

# =========================
# 角度PIDゲイン（ω生成）
# =========================
Kp_psi = 0.0
Ki_psi = 0.0
Kd_psi = 0.0
I_LIM_PSI = 20.0
PSI_REF = 0.0  # 目標姿勢角［rad］

# 角度の指数移動平均
ANGLE_ALPHA = 0.2

# =========================
# 車輪配分と出力スケール
# =========================
THETA = np.radians([240.0, 120.0, 0.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN = 1.0

CMD_MAX = 127
SPEED_TO_CMD = 0.5  # v_wheels[mm/s] → コマンド

# =========================
# ログ設定
# =========================
RUN_DURATION = 10.0  # 計測時間10秒
LOG_FILENAME = "step_response.csv"  # ステップ応答保存先
LOG_EVERY_FRAME = True  # Trueなら毎フレーム，FalseならLOG_INTERVALで間引き
LOG_INTERVAL = 0.01     # LOG_EVERY_FRAME=Falseのときに使用

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
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
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
        angle_deg = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]:
            angle_deg += 90.0
        angle_deg = angle_deg % 180.0
        angle_rad = math.radians(angle_deg)
        return ellipse, angle_rad
    return None, None

def draw_overlay(frame, center, ex, ey, ellipse, fps, psi_smooth, is_running, elapsed):
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)
    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"(x,y)[mm]=({ex:.2f},{ey:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    if ellipse is not None and psi_smooth is not None:
        deg = math.degrees(psi_smooth)
        cv2.putText(frame, f"Angle: {deg:.2f} deg", (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
                (10, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

def wheels_command_from_v(vx, vy, omega):
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA) + R_SPIN * omega) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds):
    motor_m1(int(cmds[0]))
    motor_m2(int(cmds[1]))
    motor_m3(int(cmds[2]))

def initialize_csv_logger(filename=LOG_FILENAME):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time[s]", "dt[s]", "fps",
            "ex[mm]", "ey[mm]", "dex[mm/s]", "dey[mm/s]",
            "ix", "iy",
            "psi[rad]", "epsi[rad]", "psi_i", "dpsi[rad/s]",
            "vx[mm/s]", "vy[mm/s]", "omega[rad/s]",
            "cmd1", "cmd2", "cmd3",
            "vwh1[mm/s]", "vwh2[mm/s]", "vwh3[mm/s]"
        ])

def log_to_csv_buffer(buf, **kw):
    # kw から列順に抽出して追加
    buf.append([
        f"{kw['t']:.6f}", f"{kw['dt']:.6f}", f"{kw['fps']:.3f}",
        f"{kw['ex']:.6f}", f"{kw['ey']:.6f}", f"{kw['dex']:.6f}", f"{kw['dey']:.6f}",
        f"{kw['ix']:.6f}", f"{kw['iy']:.6f}",
        f"{kw['psi']:.6f}", f"{kw['epsi']:.6f}", f"{kw['psi_i']:.6f}", f"{kw['dpsi']:.6f}",
        f"{kw['vx']:.6f}", f"{kw['vy']:.6f}", f"{kw['omega']:.6f}",
        int(kw['cmds'][0]), int(kw['cmds'][1]), int(kw['cmds'][2]),
        f"{kw['vwh'][0]:.6f}", f"{kw['vwh'][1]:.6f}", f"{kw['vwh'][2]:.6f}",
    ])

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
    # マウストラッカー開始
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        mouse_tracker.stop()
        exit()

    # ===== PID状態（未開始＝IDLE時は積分・微分を計算しない）=====
    ix = iy = 0.0
    ex_prev = ey_prev = None  # Noneにして開始時の微分を0に
    psi_i = 0.0
    psi_prev = None
    psi_smooth = None

    # ===== 実行・計測フラグ =====
    is_running = False  # 's' 押下で True
    start_time = None   # 駆動開始時刻
    last_log_time = 0.0

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - prev_time
            dt = clamp(dt, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            # 画像処理
            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # 角度の前処理
            if angle_rad is not None:
                psi_raw = wrap_pi(angle_rad)
                if psi_smooth is None:
                    psi_smooth = psi_raw
                else:
                    d = wrap_pi(psi_raw - psi_smooth)
                    psi_smooth = wrap_pi(psi_smooth + ANGLE_ALPHA * d)
            else:
                psi_raw = psi_smooth

            # 中心からの誤差
            if center is not None:
                ex = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                ey = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
            else:
                ex = ey = 0.0  # ターゲット喪失時は0扱い（安全側）
            
            # 実行状態の時間
            elapsed = 0.0 if not is_running else (now - start_time)

            # ===== 制御計算 =====
            if is_running:
                # 積分更新（アンチワインドアップ）
                ix = clamp(ix + ex * dt, -I_LIM_XY, I_LIM_XY)
                iy = clamp(iy + ey * dt, -I_LIM_XY, I_LIM_XY)

                # 微分（開始直後は0）
                if ex_prev is None:
                    dex = 0.0
                    dey = 0.0
                else:
                    dex = (ex - ex_prev) / dt
                    dey = (ey - ey_prev) / dt
                ex_prev, ey_prev = ex, ey

                vx = Kp_xy * ex + Ki_xy * ix + Kd_xy * dex
                vy = Kp_xy * ey + Ki_xy * iy + Kd_xy * dey

                # 角度制御
                if psi_raw is None:
                    epsi = 0.0
                    dpsi = 0.0
                    omega = 0.0
                else:
                    epsi = wrap_pi(PSI_REF - psi_smooth)
                    psi_i = clamp(psi_i + epsi * dt, -I_LIM_PSI, I_LIM_PSI)
                    if psi_prev is None:
                        dpsi = 0.0
                    else:
                        dpsi = wrap_pi(psi_smooth - psi_prev) / dt
                    omega = Kp_psi * epsi + Ki_psi * psi_i - Kd_psi * dpsi
                    psi_prev = psi_smooth

                # 各輪コマンド
                cmds, vwh = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds)

                # ロギング（毎フレームまたは間引き）
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    log_to_csv_buffer(
                        log_entries,
                        t=elapsed, dt=dt, fps=fps,
                        ex=ex, ey=ey, dex=dex, dey=dey,
                        ix=ix, iy=iy,
                        psi=(psi_smooth if psi_smooth is not None else 0.0),
                        epsi=epsi, psi_i=psi_i, dpsi=dpsi,
                        vx=vx, vy=vy, omega=omega,
                        cmds=cmds, vwh=vwh
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
                psi_i = 0.0
                psi_prev = None
                # vx,vy,omega,cmdsは表示用に0
                vx = vy = omega = 0.0
                cmds = np.array([0, 0, 0], dtype=int)

            # オーバレイ描画
            draw_overlay(frame_cropped, center, ex, ey, ellipse, fps, psi_smooth, is_running, elapsed)

            # 画面表示とキー
            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord('q'):
                # 強制終了
                if log_entries:
                    flush_log_entries(LOG_FILENAME, log_entries)
                break

            elif key == ord('s') and not is_running:
                # ===== ステップ応答開始 =====
                print("駆動とログを開始します（10秒）．")
                # 状態リセット（開始時の微分を0にする）
                ix = iy = 0.0
                ex_prev = ey_prev = None
                psi_i = 0.0
                psi_prev = None
                start_time = now
                last_log_time = now
                is_running = True

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        flush_log_entries(LOG_FILENAME, log_entries)
        print("プログラム終了．")
