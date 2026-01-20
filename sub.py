# main.py
# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分），実測dt，アンチワインドアップ，出力飽和つき．

import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker  # マウスの軌跡データ取得
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all  # モーター制御関数

# =========================
# カメラと表示設定
# =========================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1  # cv2.waitKey用

# クロップ（中央正方領域を抽出）→ 座標系は高さ×高さに統一
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x,y)

# 画素→mm 変換（装置に合わせて調整）
PIXEL_MM = 63.0 / 480.0  # 例：視野高480pxが実長63mmに相当

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
I_LIM_XY = 50.0  # 積分上限

# =========================
# 角度PIDゲイン（ω生成）
# =========================
Kp_psi = 0.0
Ki_psi = 0.0
Kd_psi = 0.0
I_LIM_PSI = 20.0
PSI_REF = 0.0  # 目標姿勢角［rad］（例：水平基準）

# 角度の指数移動平均（ノイズ低減，0で無効）
ANGLE_ALPHA = 0.2

# =========================
# 車輪配分と出力スケール
# =========================
# 各輪の取り付け角（ロボット座標系における駆動ローラ方向），単位：rad
THETA = np.radians([240.0, 120.0, 0.0])

# 実機の配線・プラスが前転か後転かの符号をここで吸収（±1で調整）
DIR_SGN = np.array([+1, +1, +1], dtype=float)

# 回転寄与半径 R_spin（R·ω を各輪へ加える）→ 実機でキャリブレーション
R_SPIN = 1.0

# PWM（あるいは速度指令）へのスケールと飽和
CMD_MAX = 127             # 例：-127..+127
SPEED_TO_CMD = 0.5        # v_wheels[mm/s] → コマンドの比例変換係数

# =========================
# ログ設定
# =========================
LOG_INTERVAL = 0.1
MAX_DURATION = 30 * 60
IDLE_THRESHOLD = 30.0
MOVEMENT_EPSILON = 0.5

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
    """角度を [-π, π) に折り返す．"""
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
    return max_contour, (cx, cy)  # (x,y)

def fit_ellipse_if_possible(contour):
    if contour is not None and len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        angle_deg = ellipse[2]  # 0..180
        # 楕円の長短軸で180°ジャンプが出ないように主軸の扱いを固定
        if ellipse[1][0] < ellipse[1][1]:
            angle_deg += 90.0
        angle_deg = angle_deg % 180.0
        angle_rad = math.radians(angle_deg)
        return ellipse, angle_rad
    return None, None

def draw_overlay(frame, center, ex, ey, ellipse, fps, psi_smooth):
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
    cv2.putText(frame, f"trajectory_data (x,y): ({mouse_x:.2f}, {mouse_y:.2f})",
                (100, 440), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

def wheels_command_from_v(vx, vy, omega):
    """
    vx, vy: ロボット座標系の平面速度［mm/s］．
    omega : 角速度［rad/s］．
    戻り値：各輪コマンド（整数）．
    """
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA) + R_SPIN * omega) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds

def move_motors_cmds(cmds):
    motor_m1(int(cmds[0]))
    motor_m2(int(cmds[1]))
    motor_m3(int(cmds[2]))

def initialize_csv_logger(filename="log9.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "mouse_x[mm]", "mouse_y[mm]", "angle_rad"])

def log_to_csv_buffer(buf, elapsed_time, mouse_x, mouse_y, angle_rad):
    buf.append([
        f"{elapsed_time:.3f}",
        f"{mouse_x:.3f}",
        f"{mouse_y:.3f}",
        f"{angle_rad:.6f}" if angle_rad is not None else "-1.000000"
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

    initialize_csv_logger("log9.csv")
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        mouse_tracker.stop()
        exit()

    # PID状態
    ix = iy = 0.0
    ex_prev = ey_prev = 0.0

    psi_i = 0.0
    psi_prev = None
    psi_smooth = None

    # ロギング状態
    is_logging = False
    start_log_time = None
    last_log_time = 0.0

    # 停止検出
    prev_mouse_x, prev_mouse_y = 0.0, 0.0
    last_movement_time = time.time()

    # 時間管理
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - prev_time
            # dtの安全クリップ（カメラドロップやサスペンド復帰で暴れないように）
            dt = clamp(dt, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            # 画像処理
            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # 角度の前処理（スムージングと折り返し）
            if angle_rad is not None:
                psi_raw = wrap_pi(angle_rad)
                if psi_smooth is None:
                    psi_smooth = psi_raw
                else:
                    # 位相差をwrapしてから指数移動平均
                    d = wrap_pi(psi_raw - psi_smooth)
                    psi_smooth = wrap_pi(psi_smooth + ANGLE_ALPHA * d)
            else:
                # ターゲット喪失時は角度を保持（必要なら徐々に0へ緩和しても良い）
                psi_raw = psi_smooth

            # 制御計算
            if center is not None:
                # 位置誤差［mm］
                ex = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                ey = (center[1] - FRAME_CENTER[1]) * PIXEL_MM

                # 位置PID（アンチワインドアップ）
                ix = clamp(ix + ex * dt, -I_LIM_XY, I_LIM_XY)
                iy = clamp(iy + ey * dt, -I_LIM_XY, I_LIM_XY)

                dex = (ex - ex_prev) / dt
                dey = (ey - ey_prev) / dt

                vx = Kp_xy * ex + Ki_xy * ix + Kd_xy * dex
                vy = Kp_xy * ey + Ki_xy * iy + Kd_xy * dey

                ex_prev, ey_prev = ex, ey

                # 角度PID → ω
                if psi_raw is None:
                    # 角度が取れないときは回転させない
                    omega = 0.0
                else:
                    epsi = wrap_pi(PSI_REF - psi_smooth)
                    psi_i = clamp(psi_i + epsi * dt, -I_LIM_PSI, I_LIM_PSI)
                    if psi_prev is None:
                        dpsi = 0.0
                    else:
                        dpsi = wrap_pi(psi_smooth - psi_prev) / dt
                    # 注意：微分項の符号は定義に依存．ここでは -Kd*dpsi で減衰方向へ．
                    omega = Kp_psi * epsi + Ki_psi * psi_i - Kd_psi * dpsi
                    psi_prev = psi_smooth

                # 各輪コマンド
                cmds = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds)

                # 表示
                draw_overlay(frame_cropped, center, ex, ey, ellipse, fps, psi_smooth)

            else:
                # ターゲット喪失時：安全のため出力ゼロ，積分を緩やかに減衰
                stop_all()
                ix *= 0.9
                iy *= 0.9
                psi_i *= 0.9
                draw_overlay(frame_cropped, None, 0.0, 0.0, ellipse, fps, psi_smooth)

            # 停止検出（外部マウス軌跡に基づく）
            movement = math.hypot(mouse_x - prev_mouse_x, mouse_y - prev_mouse_y)
            if movement > MOVEMENT_EPSILON:
                last_movement_time = now
            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            # ログ
            if is_logging:
                elapsed_time = now - start_log_time
                if elapsed_time >= MAX_DURATION:
                    print("30分経過のためログ停止．")
                    flush_log_entries("log9.csv", log_entries)
                    break
                elif now - last_movement_time >= IDLE_THRESHOLD:
                    print("30秒以上停止状態のためログ停止．")
                    flush_log_entries("log9.csv", log_entries)
                    break
                elif now - last_log_time >= LOG_INTERVAL:
                    log_to_csv_buffer(log_entries, elapsed_time, mouse_x, mouse_y,
                                      psi_smooth if psi_smooth is not None else -1.0)
                    last_log_time = now

            # 画面表示とキー
            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                if is_logging:
                    flush_log_entries("log9.csv", log_entries)
                break
            elif key == ord('s') and not is_logging:
                print("ロギング開始．")
                is_logging = True
                mouse_callback(0, 0)
                start_log_time = now
                last_log_time = now
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                last_movement_time = now

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        mouse_tracker.stop()
        # 念のため最終フラッシュ
        flush_log_entries("log9.csv", log_entries)
        print("プログラム終了．")
