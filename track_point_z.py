# 三輪120°オムニホイール制御（位置PID＋角度PID→ω→各輪配分）
# 目標：x_ref(t)=0[mm], y_ref(t)=0[mm]，zは psi_ref(t)=90deg + AMP*sin(Ω t) に追従（角度PID）
# 's' で開始，10 s 駆動→自動停止→CSV保存

import time
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# ===== 実験パラメータ / ログ設定 =====
RUN_DURATION     = 100.0
LOG_FILENAME     = "step_response.csv"
LOG_EVERY_FRAME  = True
LOG_INTERVAL     = 0.01

# ===== カメラ設定 =====
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS_TARGET    = 60
WAIT          = 1  # cv2.waitKey 用

# ===== クロップ（中央正方） =====
CROP_LEFT  = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)

# 画素→mm（誤差換算）
PIXEL_MM = 63.0 / 480.0

# ===== マウス（未使用だが既存構成踏襲） =====
DEVICE_PATH = '/dev/input/event9'
SCALING     = 0.0172

# ===== 位置PID（x, y）=====
Kp_xy, Ki_xy, Kd_xy = 0, 0, 0
Kp_y,  Ki_y,  Kd_y  = 0, 0, 0
I_LIM_XY = 50.0

# ===== 角度PID（z回転）=====
Kp_psi, Ki_psi, Kd_psi = 15.0, 0.0, 0.0   # ※初期: 微分少し追加で過渡安定化
I_LIM_PSI = 20.0

# 目標姿勢：90deg(π/2) ± AMP*sin(Ω t)
AMP_PSI   = math.radians(15.0)   # 目標角の振幅[rad]（=15°）
OMEGA_PSI = math.pi / 2.0        # ラジアン毎秒（周期 4 s）
PSI_BASE  = math.radians(90.0)   # 基準 90°

# 角度の指数移動平均（視覚検出ノイズ低減）
ANGLE_ALPHA   = 0.2
MAX_LOST_SEC  = 0.3   # これを超えて未検出が続けば z 制御を凍結

# ===== 車輪配分とスケール =====
THETA   = np.radians([90.0, 120.0, 240.0])   # 各ホイールの接線方向角
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN  = 1.0                                 # 機体旋回→ホイール速度ゲイン
CMD_MAX = 127
SPEED_TO_CMD = 0.5  # [mm/s]→コマンド

# ===== 参照生成 =====
def ref_xy_psi(t_sec: float):
    """
    数学系（右＋, 上＋）の目標値を返す:
      x_ref[mm], y_ref[mm], psi_ref[rad]
    今回は x=y=0 固定。psi_ref は 90°±AMP*sin(Ω t)。
    """
    xref = 0.0
    yref = 0.0
    psi_ref = (PSI_BASE + AMP_PSI * math.sin(OMEGA_PSI * t_sec))
    # wrap for safety
    psi_ref = wrap_pi(psi_ref)
    return xref, yref, psi_ref

# ========= コールバック（踏襲） =========
mouse_x, mouse_y = 0.0, 0.0
def mouse_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

# ========= ユーティリティ =========
def wrap_pi(a): return (a + np.pi) % (2 * np.pi) - np.pi
def clamp(x, lo, hi): return max(lo, min(hi, x))

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened(): return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
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
    if not contours: return None, None
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0: return max_contour, None
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
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
                 psi_smooth, psi_ref, is_running, elapsed, lost_sec):
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)

    if center is not None:
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"e[mm]=({ex_math:+.2f},{ey_math:+.2f})",
                    (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    if psi_smooth is not None:
        cv2.putText(frame, f"Angle(meas): {math.degrees(psi_smooth):.2f} deg",
                    (10,60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, f"Angle(ref):  {math.degrees(psi_ref):.2f} deg",
                (10,90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10,120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(frame, f"state:{'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
                (10,150), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    if lost_sec > 0:
        cv2.putText(frame, f"angle LOST: {lost_sec:.2f}s",
                    (10,180), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)

def wheels_command_from_v(vx, vy, omega):
    # 標準 3輪オムニ：各輪速度 = -vx*sinθ + vy*cosθ + R_SPIN*omega
    v_wheels = (-vx*np.sin(THETA) + vy*np.cos(THETA) ) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds):
    # 配線順序に合わせて割当（必要なら入替）
    motor_m1(int(cmds[2]+ 15*R_SPIN*omega))
    motor_m2(int(cmds[1]- 15*R_SPIN*omega))
    motor_m3(int(cmds[0]+ 15*R_SPIN*omega))

def initialize_csv_logger(filename=LOG_FILENAME):
    with open(filename,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow([
            "time[s]","dt[s]","fps",
            "ex[mm]","ey[mm]","dex[mm/s]","dey[mm/s]",
            "ix","iy",
            "psi_meas[rad]","psi_ref[rad]","epsi[rad]","psi_i","dpsi[rad/s]",
            "vx[mm/s]","vy[mm/s]","omega[rad/s]",
            "cmd1","cmd2","cmd3",
            "vwh1[mm/s]","vwh2[mm/s]","vwh3[mm/s]",
            "lost_sec"
        ])

def log_to_csv_buffer(buf, **kw):
    buf.append([
        f"{kw['t']:.6f}", f"{kw['dt']:.6f}", f"{kw['fps']:.3f}",
        f"{kw['ex']:.6f}", f"{kw['ey']:.6f}", f"{kw['dex']:.6f}", f"{kw['dey']:.6f}",
        f"{kw['ix']:.6f}", f"{kw['iy']:.6f}",
        f"{kw['psi_meas']:.6f}", f"{kw['psi_ref']:.6f}", f"{kw['epsi']:.6f}", f"{kw['psi_i']:.6f}", f"{kw['dpsi']:.6f}",
        f"{kw['vx']:.6f}", f"{kw['vy']:.6f}", f"{kw['omega']:.6f}",
        int(kw['cmds'][0]), int(kw['cmds'][1]), int(kw['cmds'][2]),
        f"{kw['vwh'][0]:.6f}", f"{kw['vwh'][1]:.6f}", f"{kw['vwh'][2]:.6f}",
        f"{kw['lost_sec']:.3f}"
    ])

def flush_log_entries(filename, entries):
    if not entries: return
    with open(filename,"a",newline="") as f:
        writer=csv.writer(f); writer.writerows(entries)
    entries.clear()

# ========= メイン =========
if __name__=="__main__":
    mouse_tracker = MouseTracker(DEVICE_PATH, SCALING)
    mouse_tracker.start(callback=mouse_callback)

    initialize_csv_logger(LOG_FILENAME)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない")
        mouse_tracker.stop()
        exit()

    # PID状態
    ix = iy = 0.0
    ex_prev = ey_prev = None
    psi_i = 0.0
    psi_prev = None
    psi_smooth = None

    # 検出喪失トラッキング
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

            # 角度推定（EMA）
            if angle_rad is not None:
                psi_raw = wrap_pi(angle_rad)
                if psi_smooth is None:
                    psi_smooth = psi_raw
                else:
                    psi_smooth = wrap_pi(psi_smooth + ANGLE_ALPHA * wrap_pi(psi_raw - psi_smooth))
                last_angle_seen_time = now
                lost_sec = 0.0
            else:
                psi_raw = psi_smooth
                if last_angle_seen_time is None:
                    lost_sec = 1e9  # 初期から未検出
                else:
                    lost_sec = now - last_angle_seen_time

            elapsed = 0.0 if not is_running else (now - start_time)

            # 参照生成
            if is_running:
                xref_math, yref_math, psi_ref = ref_xy_psi(elapsed)
            else:
                xref_math, yref_math, psi_ref = 0.0, 0.0, PSI_BASE

            # カメラ系へ（右＋下＋）
            xref_cam = xref_math
            yref_cam = -yref_math

            # 位置誤差（カメラ系）
            if center is not None:
                x_cam = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_cam = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
                ex_cam = x_cam - xref_cam
                ey_cam = y_cam - yref_cam
            else:
                ex_cam = ey_cam = 0.0

            # 表示用（上＋）
            ex_math = ex_cam
            ey_math = -ey_cam

            if is_running:
                # --- x,y 位置PID（今回は x=y=0 なので微小）---
                ix = clamp(ix + ex_cam * dt, -I_LIM_XY, I_LIM_XY)
                iy = clamp(iy + ey_cam * dt, -I_LIM_XY, I_LIM_XY)

                if ex_prev is None:
                    dex = 0.0; dey = 0.0
                else:
                    dex = (ex_cam - ex_prev) / dt
                    dey = (ey_cam - ey_prev) / dt
                ex_prev, ey_prev = ex_cam, ey_cam

                vx = Kp_xy * ex_cam + Ki_xy * ix + Kd_xy * dex
                vy = Kp_y  * ey_cam + Ki_y  * iy + Kd_y  * dey

                # --- z 姿勢PID（psi_ref(t) に追従）---
                if (psi_smooth is None) or (lost_sec > MAX_LOST_SEC):
                    # 未検出/喪失中：安全のため回転出力ゼロ、積分・微分更新停止
                    epsi = 0.0
                    dpsi = 0.0
                    omega = 0.0
                else:
                    epsi = wrap_pi(psi_ref - psi_smooth)
                    psi_i = clamp(psi_i + epsi * dt, -I_LIM_PSI, I_LIM_PSI)
                    if psi_prev is None:
                        dpsi = 0.0
                    else:
                        dpsi = wrap_pi(psi_smooth - psi_prev) / dt
                    omega_pid = Kp_psi * epsi + Ki_psi * psi_i - Kd_psi * dpsi
                    omega = omega_pid
                    psi_prev = psi_smooth

                # --- 3輪へ配分＆出力 ---
                cmds, vwh = wheels_command_from_v(vx, vy, omega)
                move_motors_cmds(cmds)

                # --- ログ ---
                if LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL):
                    log_to_csv_buffer(
                        log_entries,
                        t=elapsed, dt=dt, fps=fps,
                        ex=ex_cam, ey=ey_math,
                        dex=dex, dey=dey,
                        ix=ix, iy=iy,
                        psi_meas=(psi_smooth if psi_smooth is not None else 0.0),
                        psi_ref=psi_ref,
                        epsi=(epsi if 'epsi' in locals() else 0.0),
                        psi_i=psi_i, dpsi=(dpsi if 'dpsi' in locals() else 0.0),
                        vx=vx, vy=vy, omega=omega,
                        cmds=cmds, vwh=vwh,
                        lost_sec=lost_sec
                    )
                    last_log_time = now

                # --- 自動停止 ---
                if elapsed >= RUN_DURATION:
                    print("10秒経過．自動停止")
                    flush_log_entries(LOG_FILENAME, log_entries)
                    stop_all()
                    is_running = False
            else:
                # IDLE
                stop_all()
                ix = iy = 0.0
                ex_prev = ey_prev = None
                psi_i = 0.0
                psi_prev = None
                vx = vy = omega = 0.0
                cmds = np.array([0,0,0])
                dex = dey = 0.0

            # 表示
            draw_overlay(
                frame_cropped, center, ex_math, ey_math, ellipse, fps,
                psi_smooth, psi_ref, is_running, elapsed, lost_sec
            )
            cv2.imshow("Track", frame_cropped)

            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord('q'):
                if log_entries:
                    flush_log_entries(LOG_FILENAME, log_entries)
                break
            elif key == ord('s') and not is_running:
                print("駆動とログ開始（10秒）: zは 90°±sin(Ωt) 追従")
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
        print("プログラム終了")
