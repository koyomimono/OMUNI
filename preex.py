# pillbug_measure_mainstyle.py
# main.py 스타일 기반 콩벌레 이동 계측:
# - 's' 시작
# - 시작 후 5분(300s) 자동 종료 + CSV로 이동경로 저장
# - 'd' 샬레 원 재검출(Hough)로 ROI/스케일 갱신
# - 검출: 2진화(THRESH_BINARY_INV) -> largest contour -> centroid
#
# 수정사항(중요):
# - Hough로 검출되는 큰 원은 샬레 외곽(지름 105mm)
# - 실제 측정 범위/마스크/스케일은 내경(지름 100mm) 기준으로 적용

import os
import csv
import time
import math
from datetime import datetime

import cv2
import numpy as np

# =========================
# 로그/실험 설정
# =========================
MEASURE_DURATION = 300.0
LOG_EVERY_FRAME = True
LOG_INTERVAL = 0.02
FLUSH_EVERY_N = 500

BASE_DIR = os.path.join(os.getcwd(), "pillbug_runs")
LOG_BASENAME = "pillbug_path.csv"

# =========================
# 카메라 설정
# =========================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1

CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)  # (x,y) in cropped frame

# =========================
# 샬레 실치수
# =========================
# Hough로 잡히는 큰 원: 외곽 105mm
DISH_OUTER_DIAMETER_MM = 105.0
DISH_OUTER_RADIUS_MM = DISH_OUTER_DIAMETER_MM / 2.0

# 실제 측정 범위: 내경(본판/벽 안쪽) 100mm
DISH_INNER_DIAMETER_MM = 100.0
DISH_INNER_RADIUS_MM = DISH_INNER_DIAMETER_MM / 2.0

# 외곽->내경 비율 (반지름도 동일 비율)
RATIO_INNER_TO_OUTER = DISH_INNER_DIAMETER_MM / DISH_OUTER_DIAMETER_MM  # 100/105

# =========================
# 2진화
# =========================
BIN_THRESH = 50

USE_GAUSS = True
GAUSS_K = 5

USE_MORPH = True
OPEN_K = 3
CLOSE_K = 7
OPEN_ITER = 1
CLOSE_ITER = 1

# =========================
# 샬레 원 자동 재검출 (d 키)
# =========================
USE_AUTO_DISH = True
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 200
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 35
HOUGH_R_MIN = int(HEIGHT * 0.30)
HOUGH_R_MAX = int(HEIGHT * 0.49)

# =========================
# 유틸
# =========================
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

def detect_dish_circle(gray_crop):
    blur = cv2.GaussianBlur(gray_crop, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_R_MIN,
        maxRadius=HOUGH_R_MAX,
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles[0]))
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = int(circles[0][0]), int(circles[0][1]), int(circles[0][2])  # r = outer radius(px)
    return x, y, r

def gray_binary(frame, dish_mask, k_open, k_close):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if USE_GAUSS:
        gray = cv2.GaussianBlur(gray, (GAUSS_K, GAUSS_K), 0)

    _, binary = cv2.threshold(gray, BIN_THRESH, 255, cv2.THRESH_BINARY_INV)

    # 측정 범위(내경 마스크)만
    binary = cv2.bitwise_and(binary, binary, mask=dish_mask)

    if USE_MORPH:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=OPEN_ITER)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=CLOSE_ITER)

    return binary

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    max_contour = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(max_contour))
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return max_contour, None, area
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return max_contour, (cx, cy), area

def make_run_csv_path():
    os.makedirs(BASE_DIR, exist_ok=True)
    existing = []
    for name in os.listdir(BASE_DIR):
        if name.endswith(".csv") and name[:2].isdigit() and name[2:3] == "_":
            existing.append(name)
    run_num = 1
    if existing:
        nums = []
        for n in existing:
            if n[:2].isdigit():
                nums.append(int(n[:2]))
        if nums:
            run_num = max(nums) + 1
    run_tag = f"{run_num:02d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_tag, os.path.join(BASE_DIR, f"{run_tag}_{ts}_{LOG_BASENAME}")

def initialize_csv_logger(filename):
    with open(filename, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time[s]", "dt[s]", "fps",
            "x_px", "y_px",
            "x_mm", "y_mm",
            "x_mm_centered", "y_mm_centered",
            "area_px2", "area_mm2",
            "found",
            # 원 정보: outer/inner 모두 기록
            "dish_cx_px", "dish_cy_px",
            "dish_r_outer_px", "dish_r_inner_px",
            "pixel_to_mm(inner)",
            "bin_thresh"
        ])

def flush_log_entries(filename, entries):
    if not entries:
        return
    with open(filename, mode="a", newline="") as f:
        csv.writer(f).writerows(entries)
    entries.clear()

# =========================
# メイン
# =========================
if __name__ == "__main__":

    run_tag, csv_path = make_run_csv_path()
    initialize_csv_logger(csv_path)
    log_entries = []

    cap = initialize_camera()
    if not cap:
        print("カメラが開けない．")
        raise SystemExit

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))

    # 샬레 초기 fallback (outer 기준으로 잡고 -> inner로 변환)
    dish_cx = FRAME_CENTER[0]
    dish_cy = FRAME_CENTER[1]
    dish_r_outer_px = int(HEIGHT * 0.40)  # outer radius(px)
    dish_r_inner_px = max(1, int(round(dish_r_outer_px * RATIO_INNER_TO_OUTER)))

    dish_mask = np.zeros((HEIGHT, HEIGHT), dtype=np.uint8)
    cv2.circle(dish_mask, (dish_cx, dish_cy), dish_r_inner_px, 255, -1)

    # 스케일은 "내경" 기준으로
    pixel_to_mm = DISH_INNER_RADIUS_MM / float(dish_r_inner_px)
    dish_locked = False

    is_running = False
    start_time = None
    last_log_time = 0.0
    prev_time = time.time()

    print(f"[OK] run={run_tag} CSV: {csv_path}")
    print("キー: 's' 측정開始, 'd' 샬레 재탐색, 'q' 종료")
    print(f"[INFO] MEASURE_DURATION = {MEASURE_DURATION:.1f}s (5분)")
    print(f"[INFO] outer={DISH_OUTER_DIAMETER_MM:.1f}mm, inner={DISH_INNER_DIAMETER_MM:.1f}mm, ratio={RATIO_INNER_TO_OUTER:.6f}")
    print(f"[INFO] BIN_THRESH={BIN_THRESH} (필요시 70~120으로 조정)")
    print("[INFO] d 누르면 샬레 원(Hough) 다시 찾고, inner 범위로 마스크/스케일 갱신")

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
            disp = frame_cropped.copy()

            # ---- 키 ----
            key = cv2.waitKey(WAIT) & 0xFF

            if key == ord("q"):
                flush_log_entries(csv_path, log_entries)
                break

            elif key == ord("s") and not is_running:
                is_running = True
                start_time = now
                last_log_time = now
                print("[START] 측정 시작 (5분 후 자동 저장/종료)")

            elif key == ord("d"):
                dish_locked = False
                print("[Dish] 재탐색 요청")

            # ---- 샬레 원 재탐색 (outer 검출 -> inner로 변환하여 적용) ----
            if USE_AUTO_DISH and (not dish_locked):
                gray_for_dish = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
                circle = detect_dish_circle(gray_for_dish)
                if circle is not None:
                    dish_cx, dish_cy, dish_r_outer_px = circle

                    # outer -> inner 변환
                    dish_r_inner_px = max(1, int(round(dish_r_outer_px * RATIO_INNER_TO_OUTER)))

                    # inner 범위 마스크 생성
                    dish_mask = np.zeros((HEIGHT, HEIGHT), dtype=np.uint8)
                    cv2.circle(dish_mask, (dish_cx, dish_cy), dish_r_inner_px, 255, -1)

                    # 스케일은 inner 반지름 기준
                    pixel_to_mm = DISH_INNER_RADIUS_MM / float(dish_r_inner_px)

                    dish_locked = True
                    print(
                        f"[Dish] locked(outer->inner): "
                        f"cx={dish_cx}, cy={dish_cy}, "
                        f"r_outer={dish_r_outer_px}px, r_inner={dish_r_inner_px}px, "
                        f"pixel_to_mm(inner)={pixel_to_mm:.8f}"
                    )
                else:
                    dish_locked = False

            # ---- 콩벌레 검출 ----
            mask = gray_binary(frame_cropped, dish_mask, k_open, k_close)
            contour, center, area_px2 = find_largest_contour(mask)

            found = 0
            x_px = -1
            y_px = -1

            x_mm = ""
            y_mm = ""
            x_mm_c = ""
            y_mm_c = ""
            area_mm2 = ""

            if center is not None:
                found = 1
                x_px, y_px = center

                x_mm = x_px * pixel_to_mm
                y_mm = y_px * pixel_to_mm

                # 중심 기준(내경 중심)
                x_mm_c = (x_px - dish_cx) * pixel_to_mm
                y_mm_c = (y_px - dish_cy) * pixel_to_mm

                area_mm2 = area_px2 * (pixel_to_mm ** 2)

                cv2.drawContours(disp, [contour], -1, (0, 255, 0), 2)
                cv2.circle(disp, (x_px, y_px), 5, (0, 0, 255), -1)

            # ---- 오버레이: outer/inner 둘 다 표시 ----
            # inner(실측 범위)
            cv2.circle(disp, (dish_cx, dish_cy), dish_r_inner_px, (255, 0, 0), 2)
            # outer(검출 원) 참고용 점선 느낌 대신 얇게 표시
            cv2.circle(disp, (dish_cx, dish_cy), dish_r_outer_px, (0, 255, 255), 1)

            cv2.circle(disp, (dish_cx, dish_cy), 4, (255, 0, 255), -1)

            elapsed = 0.0 if not is_running else (now - start_time)

            cv2.putText(
                disp,
                f"run={run_tag}  state={'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                disp,
                f"FPS:{fps:.2f} found={found} area={area_px2} thr={BIN_THRESH}",
                (10, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                disp,
                f"outer=105mm (r={dish_r_outer_px}px)  inner=100mm (r={dish_r_inner_px}px)",
                (10, 75),
                cv2.FONT_HERSHEY_COMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                disp,
                "keys: s=start  d=re-detect dish  q=quit",
                (10, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

            cv2.imshow("Track", disp)
            cv2.imshow("Mask", mask)

            # ---- 로깅 ----
            if is_running:
                do_log = LOG_EVERY_FRAME or ((now - last_log_time) >= LOG_INTERVAL)
                if do_log:
                    log_entries.append([
                        f"{elapsed:.6f}", f"{dt:.6f}", f"{fps:.3f}",
                        x_px, y_px,
                        (f"{x_mm:.6f}" if found else ""),
                        (f"{y_mm:.6f}" if found else ""),
                        (f"{x_mm_c:.6f}" if found else ""),
                        (f"{y_mm_c:.6f}" if found else ""),
                        area_px2,
                        (f"{area_mm2:.6f}" if found else ""),
                        found,
                        dish_cx, dish_cy,
                        dish_r_outer_px, dish_r_inner_px,
                        f"{pixel_to_mm:.8f}",
                        BIN_THRESH
                    ])
                    last_log_time = now

                if len(log_entries) >= FLUSH_EVERY_N:
                    flush_log_entries(csv_path, log_entries)

                if elapsed >= MEASURE_DURATION:
                    flush_log_entries(csv_path, log_entries)
                    is_running = False
                    print("[DONE] 5분 경과. CSV 저장 완료 & 측정 종료.")
                    print(f"[DONE] saved: {csv_path}")

    finally:
        flush_log_entries(csv_path, log_entries)
        cap.release()
        cv2.destroyAllWindows()
        print("프로그램 종료.")
