# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）
# 目標：x_ref(t)=15*sin((π/2)*t)[mm], y_ref(t)=15*cos((π/2)*t)[mm]，
#       zは ψ_ref(t)=90deg + AMP*sin(Ω t) に追従（角度PID）
# 's'で開始，10 s駆動→自動停止→CSV保存（ファイル名に実行時刻を付与）
#
# 本コードは PIDゲイン（Kp,Ki,Kd）を CSV に毎行ログします。
# ヘッダ順とデータ出力順は完全一致させています。

import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# ===== 実験/ログ =====
RUN_DURATION   = 10.0
LOG_FILENAME   = f"step_response_xyz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_EVERY_FRAME = True
LOG_INTERVAL   = 0.01

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

# ===== マウス入力（既存構成を踏襲） =====
DEVICE_PATH = '/dev/input/event9'
SCALING     = 0.0172

# ===== 位置PID（x, y）=====
Kp_x = 7.0
Ki_x = 0.0
Kd_x = 0.1
I_LIM_X = 40.0

Kp_y = 14.0
Ki_y = 0.2
Kd_y = 0.3
I_LIM_Y = 40.0

# ===== 姿勢PID（z回転）=====
Kp_z = 19.0
Ki_z = 0.4
Kd_z = 0.4
I_LIM_Z = 20.0

# ψ_ref(t) = 90deg + AMP_Z*sin(Ω_z t)
AMP_Z   = math.radians(15.0)     # 15°
OMEGA_Z = math.pi / 5.0          # [rad/s]（周期 10 s）
Z_BASE  = math.radians(90.0)     # 90°

# 角度の指数移動平均/検出喪失処理
ANGLE_ALPHA  = 0.2
MAX_LOST_SEC = 0.3

# ===== 車輪配分/スケール =====
THETA   = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN  = 10.0

CMD_MAX      = 127
SPEED_TO_CMD = 0.5  # v_wheels[mm/s] → コマンド

# ===== 参照（x,y: 円運動 / z: サイン目標）=====
AMP_MM = 15.0
OMEGA  = math.pi / 2.0
X_BIAS = 0.0
Y_BIAS = 0.0

def ref_xyz(t_sec: float):
    """数学座標系（右+, 上+）の目標: x,y（円運動）+ z_ref（サイン）"""
    xref = X_BIAS + AMP_MM * math.sin(OMEGA * t_sec)
    yref = Y_BIAS + AMP_MM * math.cos(OMEGA * t_sec)
    zref = wrap_pi(Z_BASE + AMP_Z * math.sin(OMEGA_Z * t_sec))
    return xref, yref, zref

# =========================
# コールバック/ユーティリティ
# =========================
mouse_x, mouse_y = 0.0, 0.0
def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
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
        angle_deg %= 180.0
        angle_rad = math.radians(angle_deg)
        return ellipse, angle_rad
    return None, None

def draw_overlay(frame, center, ex_math, ey_math, ellipse, fps,
                 z_smooth, z_ref, is_running, elapsed, lost_sec):
    # 原点/誤差
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)
    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"e[mm]=({ex_math:+.2f},{ey_math:+.2f})", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    # 角度表示
    if z_smooth is not None:
        cv2.putText(frame, f"Angle(meas): {math.degrees(z_smooth):.2f} deg", (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Angle(ref) : {math.degrees(z_ref):.2f} deg", (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    # 状態表示
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
                (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    if lost_sec > 0:
        cv2.putText(frame, f"angle LOST: {lost_sec:.2f}s",
                    (10, 180), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

def wheels_command_from_v(vx, vy, omega):
    # 回転項はここでのみ合成（重複注入を防止）
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds, omega):
    # ωはここで R_SPIN を経由して各輪に加算/減算
    motor_m1(int(cmds[2] + (R_SPIN * omega)))
    motor_m2(int(cmds[1] - (R_SPIN * omega)))
    motor_m3(int(cmds[0] + (R_SPIN * omega)))

# =========================
# CSV関連
# =========================
def initialize_csv_logger(filename=LOG_FILENAME):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time[s]", "dt[s]", "fps",
            # 誤差系
            "ex[mm]", "ey[mm]", "dex[mm/s]", "dey[mm/s]", "ix", "iy",
            # 姿勢系
            "z_meas[rad]", "z_ref[rad]", "ez[rad]", "z_i", "dz[rad/s]",
            # 追加: PIDゲイン（定数）
            "PX","PY","PZ","IX","IY","IZ","DX","DY","DZ",
            # 出力
            "vx[mm/s]", "vy[mm/s]", "omega[rad/s]",
            "cmd1", "cmd2", "cmd3",
            "vwh1[mm/s]", "vwh2[mm/s]", "vwh3[mm/s]",
            # 状態
            "lost_sec"
        ])

def log_to_csv_buffer(buf, **kw):
    buf.append([
        f"{kw['t']:.6f}", f"{kw['dt']:.6f}", f"{kw['fps']:.3f}",
        f"{kw['ex']:.6f}", f"{kw['ey']:.6f}", f"{kw['dex']:.6f}", f"{kw['dey']:.6f}", f"{kw['ix']:.6f}", f"{kw['iy']:.6f}",
        f"{kw['z_meas']:.6f}", f"{kw['z_ref']:.6f}", f"{kw['ez']:.6f}", f"{kw['z_i']:.6f}", f"{kw['dz']:.6f}",
        f"{kw['PX']:.6f}", f"{kw['PY']:.6f}", f"{kw['PZ']:.6f}", f"{kw['IX']:.6f}", f"{kw['IY']:.6f}", f"{kw['IZ']:.6f}", f"{kw['DX']:.6f}", f"{kw['DY']:.6f}", f"{kw['DZ']:.6f}",
        f"{kw['vx']:.6f}", f"{kw['vy']:.6f}", f"{kw['omega']:.6f}",
        int(kw['cmds'][0]), int(kw['cmds'][1]), int(kw['cmds'][2]),
        f"{kw['vwh'][0]:.6f}", f"{kw['vwh'][1]:.6f}", f"{kw['vwh'][2]:.6f}",
        f"{kw['lost_sec']:.3f}"
    ])

def flush_log_entries(filename, entries):
    if not entries:
        return
    with open(filename, mode="a", newline="") as f:
        csv.writer(f).writerows(entries)
    entries.clear()

# =========================
# メインループ
# =========================
if __name__ == "__main__":
    # マウストラッカー（既存構成）
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        mouse_tracker.stop()
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

    # 実行フラグ
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

            # 画像処理
            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            # 角度の指数移動平均（EMA）
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

            # 実測中心 → 誤差（カメラ系）
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
                vy = Kp_y  * ey_cam + Ki_y  * iy + Kd_y  * dey

                # ログ用（ゲイン=定数）　※各成分値を出したい場合は Kp_x*ex_cam 等に変更
                PX, PY, PZ = Kp_x, Kp_y, Kp_z
                IX, IY, IZ = Ki_x, Ki_y, Ki_z
                DX, DY, DZ = Kd_x, Kd_y, Kd_z

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

                # --- 3輪配分＆出力 ---
                cmds, vwh = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds, omega)

                # --- ログ出力 ---
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    log_to_csv_buffer(
                        log_entries,
                        t=elapsed, dt=dt, fps=fps,
                        ex=ex_cam, ey=ey_math,
                        dex=dex, dey=dey,
                        ix=ix, iy=iy,
                        z_meas=(z_smooth if z_smooth is not None else 0.0),
                        z_ref=z_ref, ez=ez, z_i=z_i, dz=dz_val,
                        PX=PX, PY=PY, PZ=PZ, IX=IX, IY=IY, IZ=IZ, DX=DX, DY=DY, DZ=DZ,
                        vx=vx, vy=vy, omega=omega,
                        cmds=cmds, vwh=vwh,
                        lost_sec=lost_sec
                    )
                    last_log_time = now

                # --- 自動停止 ---
                if elapsed >= RUN_DURATION:
                    print("10秒経過．自動停止しCSVをフラッシュします．")
                    flush_log_entries(LOG_FILENAME, log_entries)
                    stop_all()
                    is_running = False
            else:
                # IDLE
                stop_all()
                ix = iy = 0.0
                ex_prev = ey_prev = None
                z_i = 0.0
                z_prev = None
                vx = vy = omega = 0.0
                cmds = np.array([0, 0, 0], dtype=int)
                dex = dey = 0.0

            # オーバーレイ描画
            draw_overlay(frame_cropped, center, ex_math, ey_math, ellipse, fps,
                         z_smooth, z_ref, is_running, elapsed, lost_sec)

            # 表示/キー入力
            cv2.imshow("Track", frame_cropped)
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord('q'):
                if log_entries:
                    flush_log_entries(LOG_FILENAME, log_entries)
                break
            elif key == ord('s') and not is_running:
                print("駆動とログを開始します（10秒）．(xyz制御)")
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
        mouse_tracker.stop()
        flush_log_entries(LOG_FILENAME, log_entries)
        print("プログラム終了．")
