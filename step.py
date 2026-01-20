#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all


# =========================
# 実験条件（ここを切替）
# =========================
STEP_AXIS = "x"            # "x" or "y"
STEP_MM = 15.0             # ステップ量 [mm]
STEP_DELAY = 1.0           # 開始後，何秒後にステップするか [s]
RUN_DURATION = 10.0        # 1回の計測時間 [s]（'s'押下から）
OUT_DIR = Path("step_logs")     # 出力先

# フィルタモード
# "raw"         : 検出位置をそのまま制御に使用
# "kf_pos"      : KF推定位置を制御に使用（D項は差分）
# "kf_posvel_d" : KF推定位置を制御に使用し，D項にKF速度を使用（推奨）
FILTER_MODE = "raw"  # "raw" / "kf_pos" / "kf_posvel_d"

# Kalmanパラメータ（単位は mm 系）
SIGMA_A = 800.0          # 加速度雑音の標準偏差 [mm/s^2]
SIGMA_Z = 0.8            # 観測雑音の標準偏差 [mm]

# ログ
LOG_EVERY_FRAME = True
LOG_INTERVAL = 0.01

# 見失い
MAX_LOST_SEC = 0.3       # これ以上見失ったら安全のため停止


# =========================
# カメラと表示設定
# =========================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1

# クロップ（中央正方領域を抽出）
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x, y) in cropped

# 画素→mm 変換（要キャリブレーション）
PIXEL_MM = 63.0 / 480.0


# =========================
# PIDゲイン（xy）
# =========================
Kp_x = 3.0
Ki_x = 0.00
Kd_x = 0.001
I_LIM_X = 20.0

Kp_y = 10.0
Ki_y = 0.005
Kd_y = 0.00
I_LIM_Y = 20.0


# =========================
# 車輪配分/スケール（omega無し）
# =========================
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)

CMD_MAX = 127
SPEED_TO_CMD = 0.5   # v_wheels[mm/s] → コマンド


# =========================
# ユーティリティ
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
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

    ok, _ = cap.read()
    return cap if ok else None


def gray_binary(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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


def wheels_command_from_v(vx, vy):
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels


def move_motors_cmds(cmds):
    # 既存コードの対応を踏襲（cmdsの並びとモータの並びが一致しないため，ここで対応づける）
    motor_m1(int(cmds[2]))
    motor_m2(int(cmds[1]))
    motor_m3(int(cmds[0]))


def _fmt_num_for_name(x: float) -> str:
    # ファイル名で小数点が扱いにくいので 0.8 -> 0p8
    s = f"{x}".replace(".", "p")
    s = s.replace("-", "m")
    return s


def condition_log_path() -> Path:
    """
    1条件につき1ファイル．時刻は入れない（同条件の繰り返しは同ファイルに追記）．
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if FILTER_MODE == "raw":
        tag = f"ax{STEP_AXIS}_step{STEP_MM:g}mm_delay{STEP_DELAY:g}s_raw"
    else:
        sa = _fmt_num_for_name(SIGMA_A)
        sz = _fmt_num_for_name(SIGMA_Z)
        tag = f"ax{STEP_AXIS}_step{STEP_MM:g}mm_delay{STEP_DELAY:g}s_{FILTER_MODE}_sa{sa}_sz{sz}"

    # PIDも入れる（後で解析するときに便利）
    kx = f"Kx{_fmt_num_for_name(Kp_x)}-{_fmt_num_for_name(Ki_x)}-{_fmt_num_for_name(Kd_x)}"
    ky = f"Ky{_fmt_num_for_name(Kp_y)}-{_fmt_num_for_name(Ki_y)}-{_fmt_num_for_name(Kd_y)}"

    return OUT_DIR / f"step_{tag}_{kx}_{ky}.csv"


# =========================
# 参照生成（ステップ）
# =========================
def ref_xy(t_sec: float):
    x_ref = 0.0
    y_ref = 0.0
    if t_sec >= STEP_DELAY:
        if STEP_AXIS == "x":
            x_ref = STEP_MM
        else:
            y_ref = STEP_MM
    return x_ref, y_ref


# =========================
# Kalman（OpenCV）
# =========================
@dataclass
class KFState:
    kf: cv2.KalmanFilter
    inited: bool = False


def create_kf() -> KFState:
    # 状態: [x, y, vx, vy]^T
    # 観測: [x, y]^T
    kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)

    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )

    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    kf.statePre = np.zeros((4, 1), dtype=np.float32)

    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
    return KFState(kf=kf, inited=False)


def kf_update_matrices(kfs: KFState, dt: float):
    kf = kfs.kf

    # F（遷移行列）
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]], dtype=np.float32
    )

    # Q（プロセス雑音）: 定速モデル＋加速度雑音
    q = float(SIGMA_A) ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    Q = np.array(
        [[dt4/4, 0,     dt3/2, 0],
         [0,     dt4/4, 0,     dt3/2],
         [dt3/2, 0,     dt2,   0],
         [0,     dt3/2, 0,     dt2]], dtype=np.float32
    ) * np.float32(q)
    kf.processNoiseCov = Q

    # R（観測雑音）
    r = float(SIGMA_Z) ** 2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * np.float32(r)


def kf_init_if_needed(kfs: KFState, x_mm: float, y_mm: float):
    if kfs.inited:
        return
    kf = kfs.kf
    kf.statePost = np.array([[x_mm], [y_mm], [0.0], [0.0]], dtype=np.float32)
    kf.statePre = kf.statePost.copy()
    kfs.inited = True


def kf_predict_correct(kfs: KFState, dt: float, meas_xy_mm):
    kf_update_matrices(kfs, dt)
    pred = kfs.kf.predict()

    if meas_xy_mm is not None:
        mx, my = meas_xy_mm
        kf_init_if_needed(kfs, mx, my)
        z = np.array([[mx], [my]], dtype=np.float32)
        post = kfs.kf.correct(z)
        state = post
        corrected = True
    else:
        state = pred
        corrected = False

    x = float(state[0, 0])
    y = float(state[1, 0])
    vx = float(state[2, 0])
    vy = float(state[3, 0])
    return x, y, vx, vy, corrected


# =========================
# 表示（x,y 座標のみ）
# =========================
def draw_overlay(frame, center_px, x_ref_cam, y_ref_cam, x_meas, y_meas, x_used, y_used,
                 ex, ey, fps, is_running, elapsed, lost_sec, mode, log_path):
    # 原点（中心）と検出点のみ描画
    cv2.circle(frame, FRAME_CENTER, 5, (255, 0, 0), -1)
    if center_px is not None:
        cv2.circle(frame, center_px, 5, (0, 255, 0), -1)

    # 表示は座標（mm）のみ
    y0 = 30
    dy = 28
    cv2.putText(frame, f"ref[mm] : x={x_ref_cam:+.2f}  y={y_ref_cam:+.2f}", (10, y0),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"meas[mm]: x={x_meas:+.2f}  y={y_meas:+.2f}", (10, y0 + dy),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"used[mm]: x={x_used:+.2f}  y={y_used:+.2f}", (10, y0 + 2 * dy),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)


# =========================
# CSVロガー（条件ファイルに追記）
# =========================
CSV_HEADER = [
    "run_start_iso",
    "time[s]", "dt[s]", "fps",
    "wall_time_unix[s]",
    "xref_cam[mm]", "yref_cam[mm]",
    "x_meas[mm]", "y_meas[mm]",
    "x_used[mm]", "y_used[mm]",
    "ex[mm]", "ey[mm]",
    "dex", "dey",
    "ix", "iy",
    "ux", "uy",
    "cmd_m1", "cmd_m2", "cmd_m3",
    "detected", "lost_sec",
    "kf_x", "kf_y", "kf_vx", "kf_vy", "kf_corrected",
    "sigma_a", "sigma_z", "filter_mode",
    "step_axis", "step_mm", "step_delay", "pixel_mm"
]


def ensure_csv_header(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)


def csv_append_rows(path: Path, rows: list[list]):
    if not rows:
        return
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    rows.clear()


# =========================
# メイン
# =========================
if __name__ == "__main__":
    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        raise SystemExit(1)

    # 条件ファイル（常に同じ条件なら同じファイル）
    log_path = condition_log_path()
    ensure_csv_header(log_path)

    # 実行・計測フラグ
    is_running = False
    start_time = None
    run_start_iso = ""

    # PID状態
    ix = iy = 0.0
    ex_prev = ey_prev = None

    # 見失い管理
    last_seen_time = None
    lost_sec = 1e9

    # Kalman
    kfs = create_kf()

    # バッファログ
    log_rows: list[list] = []
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
            _, center_px = find_largest_contour(mask)

            detected = 0
            x_meas = 0.0
            y_meas = 0.0
            meas_xy = None

            if center_px is not None:
                x_meas = (center_px[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_meas = (center_px[1] - FRAME_CENTER[1]) * PIXEL_MM
                meas_xy = (x_meas, y_meas)
                detected = 1
                last_seen_time = now
                lost_sec = 0.0
            else:
                if last_seen_time is None:
                    lost_sec = 1e9
                else:
                    lost_sec = now - last_seen_time

            # 経過時間
            elapsed = 0.0 if (not is_running) else (now - start_time)

            # 参照（数学座標）→カメラ座標へ変換（y反転）
            if is_running:
                xref_math, yref_math = ref_xy(elapsed)
            else:
                xref_math, yref_math = 0.0, 0.0

            xref_cam = xref_math
            yref_cam = -yref_math

            # KF推定（必要なら）
            kf_x = kf_y = kf_vx = kf_vy = 0.0
            kf_corrected = 0

            if FILTER_MODE == "raw":
                x_used = x_meas
                y_used = y_meas
            else:
                kf_x, kf_y, kf_vx, kf_vy, corrected = kf_predict_correct(kfs, dt, meas_xy)
                kf_corrected = 1 if corrected else 0
                x_used = kf_x
                y_used = kf_y

            # 誤差（カメラ座標系で統一）
            ex = x_used - xref_cam
            ey = y_used - yref_cam

            # 制御
            ux = uy = 0.0
            cmds = np.array([0, 0, 0], dtype=int)
            dex = dey = 0.0

            if is_running:
                # 見失いが長い場合は安全停止
                if lost_sec > MAX_LOST_SEC:
                    stop_all()
                    ix = iy = 0.0
                    ex_prev = ey_prev = None
                    ux = uy = 0.0
                    cmds[:] = 0
                else:
                    # I更新
                    ix = clamp(ix + ex * dt, -I_LIM_X, I_LIM_X)
                    iy = clamp(iy + ey * dt, -I_LIM_Y, I_LIM_Y)

                    # D項
                    if FILTER_MODE == "kf_posvel_d":
                        dex = kf_vx
                        dey = kf_vy
                    else:
                        if ex_prev is None:
                            dex = 0.0
                            dey = 0.0
                        else:
                            dex = (ex - ex_prev) / dt
                            dey = (ey - ey_prev) / dt
                        ex_prev, ey_prev = ex, ey

                    # PID（出力は vx, vy 相当）
                    ux = Kp_x * ex + Ki_x * ix + Kd_x * dex
                    uy = Kp_y * ey + Ki_y * iy + Kd_y * dey

                    # 車輪コマンド（omega無し）
                    cmds, _ = wheels_command_from_v(ux, uy)
                    move_motors_cmds(cmds)

                # ロギング
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    cmd_m1 = int(cmds[2])
                    cmd_m2 = int(cmds[1])
                    cmd_m3 = int(cmds[0])

                    log_rows.append([
                        run_start_iso,
                        f"{elapsed:.6f}", f"{dt:.6f}", f"{fps:.3f}",
                        f"{now:.6f}",
                        f"{xref_cam:.6f}", f"{yref_cam:.6f}",
                        f"{x_meas:.6f}", f"{y_meas:.6f}",
                        f"{x_used:.6f}", f"{y_used:.6f}",
                        f"{ex:.6f}", f"{ey:.6f}",
                        f"{dex:.6f}", f"{dey:.6f}",
                        f"{ix:.6f}", f"{iy:.6f}",
                        f"{ux:.6f}", f"{uy:.6f}",
                        cmd_m1, cmd_m2, cmd_m3,
                        int(detected), f"{lost_sec:.6f}",
                        f"{kf_x:.6f}", f"{kf_y:.6f}", f"{kf_vx:.6f}", f"{kf_vy:.6f}", int(kf_corrected),
                        f"{SIGMA_A:.6f}", f"{SIGMA_Z:.6f}", FILTER_MODE,
                        STEP_AXIS, f"{STEP_MM:.6f}", f"{STEP_DELAY:.6f}", f"{PIXEL_MM:.9f}"
                    ])
                    last_log_time = now

                # 終了
                if elapsed >= RUN_DURATION:
                    print("計測時間に達したため停止し，ログを追記します．")
                    stop_all()
                    csv_append_rows(log_path, log_rows)
                    is_running = False
                    start_time = None

            else:
                # IDLE
                stop_all()
                ix = iy = 0.0
                ex_prev = ey_prev = None

            # 表示（座標のみ）
            draw_overlay(frame_cropped, center_px, xref_cam, yref_cam,
                         x_meas, y_meas, x_used, y_used, ex, ey, fps,
                         is_running, elapsed, lost_sec, FILTER_MODE, log_path)

            cv2.imshow("Track", frame_cropped)

            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord("q"):
                csv_append_rows(log_path, log_rows)
                break

            elif key == ord("s") and (not is_running):
                # 条件ファイルを再計算（条件をコード上で変えた場合に追従）
                log_path = condition_log_path()
                ensure_csv_header(log_path)

                # 試行識別子（同じ条件ファイル内で run_start_iso により区別）
                run_start_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                # KF初期化（開始時点の観測で初期化したい）
                kfs = create_kf()
                if meas_xy is not None:
                    kf_init_if_needed(kfs, meas_xy[0], meas_xy[1])

                # 状態リセット
                ix = iy = 0.0
                ex_prev = ey_prev = None
                last_log_time = now

                start_time = now
                is_running = True
                print(f"計測開始．条件ファイルに追記します．log = {log_path}")

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        csv_append_rows(log_path, log_rows)
        print("プログラム終了．")
