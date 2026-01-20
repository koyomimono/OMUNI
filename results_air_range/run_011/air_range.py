#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# air_range_runfolder_grid20_warmup_measure_bottomexclude_bitmapside_csv.py
# - Per-run folder save: results_air_range/run_###/
# - Warmup then measure (accumulation starts after stabilization)
# - 20x20 exact grid partition (linspace edges)
# - Bottom exclusion for ALL outputs: ratio 0.40 (same as 4/10)
# - Bitmap extra side exclusion: ratio 0.10 (same as 1/10)  [bitmap only]
# - Grid counts <= 10 => 0
# - Save ONE set per run: CSV + heatmap + bitmap + overlay + grid plot + script copy + meta.txt
# - Center marker: always at image center (tiny, non-blocking)
# - Matplotlib colormap warning fixed (cmap.copy())

import os
import time
import csv
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Save (per-run folder)
# =========================
RESULTS_ROOT = "results_air_range"
RUN_PREFIX = "run_"

# =========================
# Arduino serial
# =========================
USE_ARDUINO = True
ARDUINO_PORT = "/dev/arduinoMega"   # 실제 포트가 /dev/ttyACM0 이면 그걸로 변경
ARDUINO_BAUD = 9600

# =========================
# Camera
# =========================
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FPS_TARGET = 60
WAIT = 1

# =========================
# Timing
# =========================
WARMUP_SEC = 2.0
MEASURE_DURATION = 20.0

# optical flow downscale
SMALL_W, SMALL_H = 160, 120

# =========================
# Filters (baseline/robust)
# =========================
PRE_BLUR_SIGMA = 1.0

BASELINE_EMA_ALPHA = 0.10
NOISE_K = 3.0
MOTION_EMA_ALPHA = 0.30

PCT_HIGH = 85
PCT_LOW  = 65

USE_DIFF_GATE = True
DIFF_BLUR_SIGMA = 1.0
DIFF_THRESH = 6

MORPH_KERNEL_SIZE = 2
MIN_BLOB_AREA = 40

# =========================
# Grid
# =========================
GRID_N = 20
GRID_DRAW = True
GRID_LINE_THICK = 1
GRID_ON_RATIO = 0.04

# counts <= 10 => 0
MIN_GRID_COUNT = 10

# =========================
# Exclusion ratios
# =========================
BOTTOM_EXCLUDE_RATIO = 0.40       # 하단 40% 제외 (모든 출력 공통)
BITMAP_SIDE_EXCLUDE_RATIO = 0.10  # 비트맵에서만 좌/우 10% 제외

# =========================
# Display scaling (mm) for grid plot extent
# =========================
DISPLAY_MM = 910.0
HALF_DISPLAY_MM = DISPLAY_MM / 2.0
MM_PER_PX_X = DISPLAY_MM / float(FRAME_W)
MM_PER_PX_Y = DISPLAY_MM / float(FRAME_H)

# =========================
# Min motion threshold
# =========================
MIN_MOVE_MM = 3.0

# =========================
# Hot pixel rejection
# =========================
HOT_PIXEL_FRAC = 0.50

# =========================
# Auto flicker mask calibration
# =========================
CALIB_SEC = 1.0
FLICKER_FREQ_TH = 0.60
FLICKER_DILATE_K = 5

# =========================
# ROI for hysteresis thresholding
# =========================
ROI_MARGIN_FRAC = 0.15

# =========================
# CLAHE
# =========================
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# =========================
# Visualization
# =========================
ALPHA_HEAT = 0.6
COLORMAP = cv2.COLORMAP_JET

# Save image scaling
SAVE_UPSCALE = 2.5

# Center marker style (small)
CENTER_TEXT = "(0,0)"
CENTER_TEXT_SCALE = 0.42
CENTER_TEXT_THICK = 1
CENTER_DOT_R = 2
CENTER_CROSS = 6  # half length (px)


# =========================
# Utility: run folder
# =========================
def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _next_run_id(root_dir, prefix):
    _safe_mkdir(root_dir)
    max_id = 0
    for name in os.listdir(root_dir):
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        if tail.isdigit():
            v = int(tail)
            if v > max_id:
                max_id = v
    return max_id + 1


def _make_run_dir():
    run_id = _next_run_id(RESULTS_ROOT, RUN_PREFIX)
    run_name = f"{RUN_PREFIX}{run_id:03d}"
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    _safe_mkdir(run_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(run_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"run={run_name}\n")
        f.write(f"timestamp={ts}\n")
        f.write(f"warmup_sec={WARMUP_SEC}\n")
        f.write(f"measure_sec={MEASURE_DURATION}\n")
        f.write(f"grid={GRID_N}x{GRID_N}\n")
        f.write(f"bottom_exclude_ratio={BOTTOM_EXCLUDE_RATIO}\n")
        f.write(f"bitmap_side_exclude_ratio={BITMAP_SIDE_EXCLUDE_RATIO}\n")
        f.write(f"min_move_mm={MIN_MOVE_MM}\n")
        f.write(f"min_grid_count_le={MIN_GRID_COUNT}\n")
    return run_id, run_dir, ts


def _copy_this_script(run_dir):
    try:
        this_file = os.path.abspath(__file__)
        dst = os.path.join(run_dir, os.path.basename(this_file))
        if os.path.isfile(this_file):
            with open(this_file, "rb") as rf:
                data = rf.read()
            with open(dst, "wb") as wf:
                wf.write(data)
            print(f"[SAVE] script copy -> {dst}")
    except Exception as e:
        print(f"[WARN] script copy failed: {e}")


# =========================
# Camera / Arduino
# =========================
def initialize_camera_for_flow():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cv2.namedWindow("LIVE", flags=cv2.WINDOW_GUI_NORMAL)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] actual {aw}x{ah}, fps={afps}")
    return cap


def open_arduino():
    if not USE_ARDUINO:
        return None
    try:
        import serial
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(1.2)  # open 시 리셋되는 보드 대비
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        print(f"[Arduino] Open OK: {ARDUINO_PORT} @ {ARDUINO_BAUD}")
        return ser
    except Exception as e:
        print(f"[Arduino] Open failed: {e}")
        return None


def select_and_send_signal(ser):
    sel = "1"
    try:
        raw = input("Arduino signal select (1~5) [default=1]: ").strip()
        if raw in ["1", "2", "3", "4", "5"]:
            sel = raw
        else:
            if raw != "":
                print("[Arduino] invalid input -> default 1")
    except Exception:
        sel = "1"

    if ser is None or (not getattr(ser, "is_open", False)):
        print(f"[Arduino] (no serial) skip send '{sel}'")
        return sel

    payload = (sel + "\n").encode("ascii")
    for k in range(5):
        try:
            ser.write(payload)
            ser.flush()
            print(f"[Arduino] Sent '{sel}' (try {k+1}/5)")
            break
        except Exception as e:
            print(f"[Arduino] Send failed (try {k+1}/5): {e}")
            time.sleep(0.2)
    return sel


# =========================
# Image helpers
# =========================
def _gauss_if(gray, sigma):
    if sigma is None or sigma <= 0:
        return gray
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)


def _mad(vals, med):
    return np.median(np.abs(vals - med))


def _postprocess_mask(mask):
    if MORPH_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    if MIN_BLOB_AREA > 0:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) >= float(MIN_BLOB_AREA):
                cv2.drawContours(cleaned, [c], -1, 1, -1)
        mask = cleaned

    return mask


def _roi_slices(h, w, margin_frac):
    mx = int(round(w * margin_frac))
    my = int(round(h * margin_frac))
    mx = max(0, min(mx, w // 2 - 1))
    my = max(0, min(my, h // 2 - 1))
    return slice(my, h - my), slice(mx, w - mx)


def _hysteresis_mask_roi(mag_ema, prev_mask, roi_y, roi_x):
    roi = mag_ema[roi_y, roi_x]
    valid = roi > 0
    if np.count_nonzero(valid) == 0:
        high = low = 0.0
    else:
        vals = roi[valid]
        high = np.percentile(vals, PCT_HIGH)
        low = np.percentile(vals, PCT_LOW)

    new_on = mag_ema >= high
    keep_on = (prev_mask > 0) & (mag_ema >= low)
    return (new_on | keep_on).astype(np.uint8)


def _grid_edges(n, length):
    # linspace -> round -> monotonic fix: "칸 크기 최대한 동일" + 마지막은 정확히 length
    edges = np.linspace(0, int(length), int(n) + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = int(length)
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _ratio_to_cells(ratio, n, min_count=0, max_count=None):
    if max_count is None:
        max_count = n
    c = int(round(float(ratio) * float(n)))
    c = max(min_count, min(c, max_count))
    return c


def _make_common_masks(x_edges, y_edges):
    # bottom exclusion aligned to grid rows
    excl_bottom_rows = _ratio_to_cells(BOTTOM_EXCLUDE_RATIO, GRID_N, 0, GRID_N)
    keep_rows = GRID_N - excl_bottom_rows
    keep_rows = max(0, min(keep_rows, GRID_N))
    bottom_cut_y = int(y_edges[keep_rows])

    bottom_mask = np.ones((FRAME_H, FRAME_W), dtype=np.uint8)
    if excl_bottom_rows > 0 and bottom_cut_y < FRAME_H:
        bottom_mask[bottom_cut_y:, :] = 0

    # bitmap side exclusion aligned to grid cols
    excl_side_cols = _ratio_to_cells(BITMAP_SIDE_EXCLUDE_RATIO, GRID_N, 0, max(0, GRID_N // 2 - 1))
    left_px = int(x_edges[excl_side_cols])
    right_px = int(FRAME_W - x_edges[GRID_N - excl_side_cols])

    bitmap_side_mask = np.ones((FRAME_H, FRAME_W), dtype=np.uint8)
    if excl_side_cols > 0:
        if left_px > 0:
            bitmap_side_mask[:, :left_px] = 0
        if right_px > 0:
            bitmap_side_mask[:, (FRAME_W - right_px):] = 0

    return bottom_mask, bottom_cut_y, excl_bottom_rows, bitmap_side_mask, excl_side_cols


def _save_big_png(path, img_bgr):
    img = img_bgr
    if SAVE_UPSCALE is not None and SAVE_UPSCALE > 1.0:
        new_w = int(round(img.shape[1] * SAVE_UPSCALE))
        new_h = int(round(img.shape[0] * SAVE_UPSCALE))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def _draw_center_marker_bgr(img_bgr, color_bgr, text_color_bgr):
    out = img_bgr.copy()
    cx, cy = FRAME_W // 2, FRAME_H // 2

    cv2.line(out, (cx - CENTER_CROSS, cy), (cx + CENTER_CROSS, cy), color_bgr, 1)
    cv2.line(out, (cx, cy - CENTER_CROSS), (cx, cy + CENTER_CROSS), color_bgr, 1)
    cv2.circle(out, (cx, cy), CENTER_DOT_R, color_bgr, -1)

    pos = (cx + 6, cy - 6)
    cv2.putText(
        out, CENTER_TEXT, pos,
        cv2.FONT_HERSHEY_SIMPLEX, CENTER_TEXT_SCALE,
        text_color_bgr, CENTER_TEXT_THICK, cv2.LINE_AA
    )
    return out


def _draw_center_marker_bitmap_gray(img_gray):
    out = img_gray.copy()
    cx, cy = FRAME_W // 2, FRAME_H // 2

    cv2.circle(out, (cx, cy), CENTER_DOT_R, 0, -1)

    # black text (with thin white outline for readability)
    pos = (cx + 6, cy - 6)
    cv2.putText(out, CENTER_TEXT, pos, cv2.FONT_HERSHEY_SIMPLEX, CENTER_TEXT_SCALE, 255, 2, cv2.LINE_AA)
    cv2.putText(out, CENTER_TEXT, pos, cv2.FONT_HERSHEY_SIMPLEX, CENTER_TEXT_SCALE, 0, 1, cv2.LINE_AA)
    return out


def save_grid_count_annotated_big(grid_counts, out_path, title, extent):
    g = grid_counts.astype(np.float32)
    vmax = float(np.max(g))
    if vmax <= 0:
        vmax = 1.0

    plt.figure(figsize=(16, 16))
    im = plt.imshow(
        g,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
        cmap="viridis",
    )

    # colormap warning fix: copy()
    cmap = mpl.cm.get_cmap("viridis").copy()
    im.set_cmap(cmap)

    plt.xlabel("x (mm)", fontsize=16)
    plt.ylabel("y (mm)", fontsize=16)
    plt.title(title, fontsize=18)

    plt.axvline(0.0, linewidth=2)
    plt.axhline(0.0, linewidth=2)
    plt.text(0.0, 0.0, " (0,0)", fontsize=9, color="red")

    cbar = plt.colorbar(im)
    cbar.set_label("count (frames)", fontsize=14)

    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / float(GRID_N)
    dy = (ymax - ymin) / float(GRID_N)

    fs = 7
    for j in range(GRID_N):
        for i in range(GRID_N):
            val = int(grid_counts[j, i])
            if val <= 0:
                continue
            cx = xmin + (i + 0.5) * dx
            cy = ymin + (j + 0.5) * dy
            color = "black" if val >= 0.6 * vmax else "white"
            plt.text(cx, cy, str(val), ha="center", va="center", fontsize=fs, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=450)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    ser = open_arduino()
    selected = select_and_send_signal(ser)

    # ===== create run folder =====
    run_id, run_dir, ts = _make_run_dir()
    print(f"[RUN_DIR] {run_dir}")
    _copy_this_script(run_dir)

    cap = initialize_camera_for_flow()
    if cap is None:
        print("카메라 오픈 실패 (V4L2 + YUYV)")
        if ser is not None and getattr(ser, "is_open", False):
            ser.close()
        return

    ok, prev_bgr = cap.read()
    if not ok:
        print("첫 프레임 읽기 실패")
        cap.release()
        if ser is not None and getattr(ser, "is_open", False):
            ser.close()
        return

    # adapt to actual camera size
    H, W = prev_bgr.shape[:2]
    global FRAME_W, FRAME_H, MM_PER_PX_X, MM_PER_PX_Y
    if (W != FRAME_W) or (H != FRAME_H):
        print(f"[WARN] requested {FRAME_W}x{FRAME_H}, but got {W}x{H}. Use actual.")
        FRAME_W, FRAME_H = W, H
        MM_PER_PX_X = DISPLAY_MM / float(FRAME_W)
        MM_PER_PX_Y = DISPLAY_MM / float(FRAME_H)

    x_edges = _grid_edges(GRID_N, FRAME_W)
    y_edges = _grid_edges(GRID_N, FRAME_H)

    bottom_mask, bottom_cut_y, excl_bottom_rows, bitmap_side_mask, excl_side_cols = _make_common_masks(x_edges, y_edges)
    print(f"[BOTTOM] exclude rows={excl_bottom_rows} ratio={BOTTOM_EXCLUDE_RATIO:.2f} cut_y={bottom_cut_y}px")
    print(f"[BITMAP] side exclude cols={excl_side_cols} ratio={BITMAP_SIDE_EXCLUDE_RATIO:.2f}")

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE) if USE_CLAHE else None
    roi_y, roi_x = _roi_slices(FRAME_H, FRAME_W, ROI_MARGIN_FRAC)

    prev_gray_full = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_full = _gauss_if(prev_gray_full, PRE_BLUR_SIGMA)
    if clahe is not None:
        prev_gray_full = clahe.apply(prev_gray_full)

    prev_small = cv2.resize(prev_gray_full, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)

    baseline_mag = 0.0
    mag_ema_full = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    prev_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    motion_accum = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    bitmap_any = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    grid_counts = np.zeros((GRID_N, GRID_N), dtype=np.int32)

    pixel_on_count = np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)
    hot_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    flicker_count = np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)
    flicker_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    calib_frames_target = int(max(1, round(CALIB_SEC * FPS_TARGET)))
    frames_seen = 0

    mm_per_px = min(MM_PER_PX_X, MM_PER_PX_Y)
    min_move_px = float(MIN_MOVE_MM) / float(mm_per_px)

    est_measure_frames = int(max(1.0, MEASURE_DURATION * FPS_TARGET))
    hot_count_th = int(max(1, round(HOT_PIXEL_FRAC * est_measure_frames)))

    total_needed = WARMUP_SEC + MEASURE_DURATION
    print(f"[RUN] sig={selected} | warmup={WARMUP_SEC}s + measure={MEASURE_DURATION}s (total {total_needed}s)")
    print(f"[CFG] min_move_mm={MIN_MOVE_MM} -> min_move_px≈{min_move_px:.2f}px")
    print(f"[CFG] MIN_GRID_COUNT={MIN_GRID_COUNT} (<= {MIN_GRID_COUNT} => 0)")

    # CSV path (in run folder)
    csv_path = os.path.join(run_dir, f"optflow_run{run_id:03d}_sig{selected}.csv")
    fcsv = open(csv_path, "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow([
        "t_meas_s",
        "med_mag_full", "mad_mag_full", "floor",
        "masked_area_px", "masked_ratio",
        "mean_mag_masked_fullpx",
        "mean_vx_small", "mean_vy_small", "mean_mag_small_masked",
    ])

    start_time = time.time()
    measure_started = False
    measure_start_t = None

    while True:
        ok, bgr = cap.read()
        if not ok:
            print("프레임 읽기 실패, 종료합니다.")
            break

        now = time.time()
        elapsed_total = now - start_time

        if (not measure_started) and (elapsed_total >= WARMUP_SEC):
            measure_started = True
            measure_start_t = now
            pixel_on_count[:] = 0
            hot_mask[:] = 0
            print("[PHASE] MEASURE START (accumulation enabled)")

        elapsed_meas = (now - measure_start_t) if measure_started else 0.0

        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_full = _gauss_if(gray_full, PRE_BLUR_SIGMA)
        if clahe is not None:
            gray_full = clahe.apply(gray_full)

        # diff gate
        if USE_DIFF_GATE:
            diff = cv2.absdiff(gray_full, prev_gray_full)
            diff = _gauss_if(diff, DIFF_BLUR_SIGMA)
            _, diff_mask = cv2.threshold(diff, DIFF_THRESH, 1, cv2.THRESH_BINARY)
        else:
            diff_mask = None

        # flicker calibration (early)
        if (diff_mask is not None) and (frames_seen < calib_frames_target):
            flicker_count += diff_mask.astype(np.uint16)
            if frames_seen == calib_frames_target - 1:
                freq = flicker_count.astype(np.float32) / float(max(1, calib_frames_target))
                flicker_mask = (freq >= float(FLICKER_FREQ_TH)).astype(np.uint8)
                if FLICKER_DILATE_K > 0 and np.any(flicker_mask):
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FLICKER_DILATE_K, FLICKER_DILATE_K))
                    flicker_mask = cv2.dilate(flicker_mask, k, iterations=1)
                print(f"[CALIB] flicker_mask built. masked_pixels={int(np.count_nonzero(flicker_mask))}")

        small_gray = cv2.resize(gray_full, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            prev_small, small_gray, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        vx_small = flow[..., 0]
        vy_small = flow[..., 1]
        mag_small, _ = cv2.cartToPolar(vx_small, vy_small)

        # scale SMALL magnitude to FULL pixel magnitude
        sx = float(FRAME_W) / float(SMALL_W)
        sy = float(FRAME_H) / float(SMALL_H)
        scale_mag = 0.5 * (sx + sy)

        mag_full = cv2.resize(mag_small, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
        mag_full = mag_full * scale_mag

        # robust floor
        med = float(np.median(mag_full))
        mad = float(_mad(mag_full, med)) if mag_full.size > 0 else 0.0
        baseline_mag = (1.0 - BASELINE_EMA_ALPHA) * baseline_mag + BASELINE_EMA_ALPHA * med
        floor = baseline_mag + NOISE_K * mad

        mag_detrend = mag_full - floor
        mag_detrend[mag_detrend < 0] = 0.0
        mag_detrend[mag_detrend < min_move_px] = 0.0

        mag_ema_full = (1.0 - MOTION_EMA_ALPHA) * mag_ema_full + MOTION_EMA_ALPHA * mag_detrend

        mask = _hysteresis_mask_roi(mag_ema_full, prev_mask, roi_y, roi_x)
        mask = _postprocess_mask(mask)

        if diff_mask is not None:
            mask = (mask & diff_mask.astype(np.uint8))
        if np.any(flicker_mask):
            mask[flicker_mask == 1] = 0

        # bottom exclusion (all outputs)
        mask = (mask & bottom_mask)

        # hot pixel rejection only during measure
        if measure_started:
            pixel_on_count += mask.astype(np.uint16)
            hot_mask[pixel_on_count >= hot_count_th] = 1
            if np.any(hot_mask):
                mask[hot_mask == 1] = 0

        mag_detrend[mask == 0] = 0.0

        # accumulate only during measure
        if measure_started and (0.0 <= elapsed_meas <= MEASURE_DURATION):
            motion_accum += (mag_detrend * mask.astype(np.float32))

            bitmap_mask = (mask & bitmap_side_mask)
            bitmap_any |= bitmap_mask.astype(np.uint8)

            # exact grid count (cell-wise)
            for j in range(GRID_N):
                y0, y1 = int(y_edges[j]), int(y_edges[j + 1])
                if y1 <= y0:
                    continue
                row = mask[y0:y1, :]
                for i in range(GRID_N):
                    x0, x1 = int(x_edges[i]), int(x_edges[i + 1])
                    if x1 <= x0:
                        continue
                    cell = row[:, x0:x1]
                    ratio = float(np.count_nonzero(cell)) / float(cell.size)
                    if ratio >= float(GRID_ON_RATIO):
                        grid_counts[j, i] += 1

            # CSV features
            masked_area = int(np.count_nonzero(mask))
            masked_ratio = float(masked_area) / float(FRAME_W * FRAME_H)

            if masked_area > 0:
                mean_mag_masked = float(np.sum(mag_full * mask.astype(np.float32)) / float(masked_area))
            else:
                mean_mag_masked = 0.0

            mask_small = cv2.resize(mask.astype(np.uint8), (SMALL_W, SMALL_H), interpolation=cv2.INTER_NEAREST)
            cnt_small = int(np.count_nonzero(mask_small))
            if cnt_small > 0:
                mean_vx = float(np.sum(vx_small * mask_small.astype(np.float32)) / float(cnt_small))
                mean_vy = float(np.sum(vy_small * mask_small.astype(np.float32)) / float(cnt_small))
                mean_mag_small_masked = float(np.sum(mag_small * mask_small.astype(np.float32)) / float(cnt_small))
            else:
                mean_vx = 0.0
                mean_vy = 0.0
                mean_mag_small_masked = 0.0

            wcsv.writerow([
                f"{elapsed_meas:.4f}",
                f"{med:.6g}", f"{mad:.6g}", f"{floor:.6g}",
                masked_area, f"{masked_ratio:.6g}",
                f"{mean_mag_masked:.6g}",
                f"{mean_vx:.6g}", f"{mean_vy:.6g}", f"{mean_mag_small_masked:.6g}",
            ])

        # LIVE view
        mag_vis = cv2.normalize(mag_ema_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(mag_vis, COLORMAP)
        overlay_all = cv2.addWeighted(bgr, 1.0 - ALPHA_HEAT, heat, ALPHA_HEAT, 0)

        mask_3c = cv2.merge([mask, mask, mask])
        result = np.where(mask_3c == 1, overlay_all, bgr)

        if GRID_DRAW:
            for k in range(1, GRID_N):
                x = int(x_edges[k])
                cv2.line(result, (x, 0), (x, FRAME_H - 1), (255, 255, 255), GRID_LINE_THICK)
            for k in range(1, GRID_N):
                y = int(y_edges[k])
                cv2.line(result, (0, y), (FRAME_W - 1, y), (255, 255, 255), GRID_LINE_THICK)

        if excl_bottom_rows > 0:
            cv2.line(result, (0, bottom_cut_y), (FRAME_W - 1, bottom_cut_y), (0, 0, 255), 2)

        result = _draw_center_marker_bgr(result, (0, 0, 255), (0, 0, 255))

        phase = "WARMUP" if not measure_started else "MEASURE"
        cv2.putText(
            result,
            f"{phase} total={elapsed_total:.2f}/{total_needed:.0f}s  meas={elapsed_meas:.2f}/{MEASURE_DURATION:.0f}s",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.imshow("LIVE", result)

        prev_small = small_gray
        prev_gray_full = gray_full
        prev_mask = mask
        frames_seen += 1

        key = cv2.waitKey(WAIT) & 0xFF
        if key == ord("q"):
            print("사용자 종료(q).")
            break
        if measure_started and (elapsed_meas >= MEASURE_DURATION):
            print("[DONE] measurement duration reached.")
            break

    cap.release()
    try:
        fcsv.close()
    except Exception:
        pass

    # counts <= MIN_GRID_COUNT => 0
    grid_counts[grid_counts <= int(MIN_GRID_COUNT)] = 0

    print(f"[SAVE] {csv_path}")

    # ===== Save heatmap motion_accum
    max_val = float(motion_accum.max())
    if max_val <= 0.0:
        acc_norm = np.zeros_like(motion_accum, dtype=np.uint8)
    else:
        acc_norm = cv2.normalize(motion_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(acc_norm, COLORMAP)
    heatmap_color = _draw_center_marker_bgr(heatmap_color, (0, 0, 255), (0, 0, 255))
    heatmap_path = os.path.join(run_dir, f"heatmap_run{run_id:03d}_{int(MEASURE_DURATION)}s_sig{selected}.png")
    _save_big_png(heatmap_path, heatmap_color)
    print(f"[SAVE] {heatmap_path}")

    # ===== Save bitmap (mask any)
    bm = (bitmap_any * 255).astype(np.uint8)
    bm = _draw_center_marker_bitmap_gray(bm)
    bitmap_path = os.path.join(run_dir, f"bitmap_run{run_id:03d}_{int(MEASURE_DURATION)}s_sig{selected}.png")
    _save_big_png(bitmap_path, cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR))
    print(f"[SAVE] {bitmap_path}")

    # ===== Save overlay
    bg = cv2.cvtColor(prev_gray_full, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(bg, 0.55, heatmap_color, 0.45, 0)
    overlay = _draw_center_marker_bgr(overlay, (0, 0, 255), (0, 0, 255))
    overlay_path = os.path.join(run_dir, f"overlay_run{run_id:03d}_{int(MEASURE_DURATION)}s_sig{selected}.png")
    _save_big_png(overlay_path, overlay)
    print(f"[SAVE] {overlay_path}")

    # ===== Save grid plot
    extent = [-HALF_DISPLAY_MM, HALF_DISPLAY_MM, -HALF_DISPLAY_MM, HALF_DISPLAY_MM]
    grid_plot_path = os.path.join(run_dir, f"grid_count_run{run_id:03d}_{GRID_N}x{GRID_N}_sig{selected}_LE{MIN_GRID_COUNT}.png")
    title = f"Grid Motion Count(frames) {GRID_N}x{GRID_N} | <= {MIN_GRID_COUNT} => 0"
    save_grid_count_annotated_big(grid_counts, grid_plot_path, title, extent)
    print(f"[SAVE] {grid_plot_path}")

    # ===== Show results
    cv2.namedWindow("HEATMAP", flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("HEATMAP", heatmap_color)

    cv2.namedWindow("BITMAP", flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("BITMAP", bm)

    ov_img = cv2.imread(overlay_path)
    if ov_img is not None:
        cv2.namedWindow("OVERLAY", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("OVERLAY", ov_img)

    grid_img = cv2.imread(grid_plot_path)
    if grid_img is not None:
        cv2.namedWindow("GRID_COUNT", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("GRID_COUNT", grid_img)

    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ser is not None and getattr(ser, "is_open", False):
        ser.close()


if __name__ == "__main__":
    main()
