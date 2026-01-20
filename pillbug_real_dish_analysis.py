#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pillbug_real_dish_analysis.py  (extended, robust + better heatmap + more speed plots)

현실 샬레(지름 100mm)에서 콩벌레 자유 탐색 행동 분석 전용.
- 자극 없음
- run 단위 분석 + 해석
- VR 비교용 feature table(ALL_real_summary.csv) 자동 생성

=== 입력 CSV 필수 컬럼 (너 포맷 기준) ===
time[s], found, x_mm_centered, y_mm_centered

=== 있으면 적극 사용(강력 추천) ===
dt[s], fps, area_mm2, area_px2, bin_thresh, pixel_to_mm(inner)

=== 핵심 방어 로직(속도/거리 폭발 방지) ===
1) time[s]가 이상하면 dt[s] 누적으로 time 재구성
2) 프레임당 이동거리 점프(max_step_mm) 이상이면 좌표 NaN 처리 후 보간
3) speed 상한(speed_clip_mm_s) 클리핑

출력:
out_root/<csv_stem>/
  - processed_timeseries.csv
  - summary.csv
  - interpretation.txt
  - figures/*.png
out_root/ALL_real_summary.csv

사용:
python3 pillbug_real_dish_analysis.py --input /path/to/file_or_dir --out real_dish_analysis
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================
# 설정(필요하면 여기만 조절)
# =====================
DISH_DIAMETER_MM_DEFAULT = 100.0

# 정지 판정 임계 속도(mm/s)
STOP_SPEED_THRESH_DEFAULT = 1.0

# thigmotaxis 기준(반지름 비율)
EDGE_THRESHOLDS = [0.7, 0.8, 0.9]

# 속도 폭발 방지 파라미터
MAX_DT_SEC = 0.2
MIN_DT_SEC = 1e-4
MAX_STEP_MM_DEFAULT = 20.0
SPEED_CLIP_MM_S_DEFAULT = 200.0

# 히트맵 시각화
HEATMAP_CMAP = "inferno"     # 잘 보이는 컬러맵
HEATMAP_BG = "#111111"       # 어두운 배경
HEATMAP_DPI = 300


# =====================
# 유틸
# =====================
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def pick_optional_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def plot_circle_boundary(r_mm: float):
    th = np.linspace(0, 2 * np.pi, 361)
    plt.plot(r_mm * np.cos(th), r_mm * np.sin(th), linewidth=1)


def safe_interp_nans(arr: np.ndarray) -> np.ndarray:
    """NaN을 선형 보간(양끝은 최근값 유지)."""
    a = np.asarray(arr, float).copy()
    idx = np.arange(len(a))
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return a
    if ok.sum() == 1:
        a[~ok] = a[ok][0]
        return a
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    return a


def occupancy_hist2d(x, y, lim, bins):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[[-lim, lim], [-lim, lim]])
    s = H.sum()
    if s > 0:
        H = H / s
    return H


def rolling_mean(y, win):
    y = np.asarray(y, float)
    if win <= 1 or len(y) < win:
        return y
    # nan 대응 간단 처리: nan은 0으로 두고 가중치로 나눔
    y0 = np.where(np.isfinite(y), y, 0.0)
    w0 = np.where(np.isfinite(y), 1.0, 0.0)
    kernel = np.ones(win, dtype=float)
    num = np.convolve(y0, kernel, mode="same")
    den = np.convolve(w0, kernel, mode="same")
    out = np.where(den > 0, num / den, np.nan)
    return out


# =====================
# 데이터 로드 + 전처리(robust)
# =====================
def build_time_axis(df: pd.DataFrame) -> np.ndarray:
    """
    time[s]가 깨졌을 가능성이 있으니:
    - dt[s]가 있으면 dt 누적으로 time 재구성(가장 안정)
    - 아니면 time[s] 사용
    """
    if "dt[s]" in df.columns:
        dt = _num(df["dt[s]"]).to_numpy()
        dt = np.where(np.isfinite(dt) & (dt > MIN_DT_SEC) & (dt < MAX_DT_SEC), dt, np.nan)
        med = np.nanmedian(dt)
        if not np.isfinite(med):
            med = 1.0 / 60.0
        dt = np.where(np.isfinite(dt), dt, med)
        t = np.cumsum(dt)
        t = t - t[0]
        return t

    t = _num(df["time[s]"]).to_numpy()
    return t


def load_and_process(csv_path: Path, dish_diameter_mm: float, max_step_mm: float, speed_clip_mm_s: float):
    df = pd.read_csv(csv_path)

    required = ["time[s]", "found", "x_mm_centered", "y_mm_centered"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{csv_path.name}: missing column: {c}")

    df = df.copy()
    df["time_s"] = build_time_axis(df)

    df["found"] = _num(df["found"]).fillna(0).astype(int)
    df["x_mm_c"] = _num(df["x_mm_centered"])
    df["y_mm_c"] = _num(df["y_mm_centered"])

    # optional
    fps_col = pick_optional_col(df.columns, ["fps", "FPS"])
    df["fps_val"] = _num(df[fps_col]) if fps_col else np.nan

    area_mm2_col = pick_optional_col(df.columns, ["area_mm2", "area_mm2_val"])
    df["area_mm2_val"] = _num(df[area_mm2_col]) if area_mm2_col else np.nan

    area_px2_col = pick_optional_col(df.columns, ["area_px2", "area_px2_val"])
    df["area_px2_val"] = _num(df[area_px2_col]) if area_px2_col else np.nan

    pixmm_col = pick_optional_col(df.columns, ["pixel_to_mm(inner)", "pixel_to_mm", "pixel_to_mm_inner"])
    df["pixel_to_mm_val"] = _num(df[pixmm_col]) if pixmm_col else np.nan

    thr_col = pick_optional_col(df.columns, ["bin_thresh", "BIN_THRESH"])
    df["bin_thresh_val"] = _num(df[thr_col]) if thr_col else np.nan

    det = df[df["found"] == 1].copy()
    det = det.dropna(subset=["time_s", "x_mm_c", "y_mm_c"])
    det = det.sort_values("time_s").reset_index(drop=True)

    # Δt<=0 제거
    if len(det) >= 2:
        t0 = det["time_s"].to_numpy()
        keep = np.ones(len(det), dtype=bool)
        keep[1:] = np.diff(t0) > 1e-9
        det = det[keep].reset_index(drop=True)

    meta = {
        "file": csv_path.name,
        "found_rate_raw": float(df["found"].mean()) if len(df) else np.nan,
        "mean_fps_raw": float(df["fps_val"].dropna().mean()) if df["fps_val"].dropna().shape[0] else np.nan,
        "pixel_to_mm": float(det["pixel_to_mm_val"].dropna().iloc[0]) if det["pixel_to_mm_val"].dropna().shape[0] else np.nan,
        "bin_thresh": float(det["bin_thresh_val"].dropna().iloc[0]) if det["bin_thresh_val"].dropna().shape[0] else np.nan,
        "time_axis_source": "dt_cumsum" if "dt[s]" in df.columns else "time[s]",
    }

    if len(det) < 2:
        for c in ["vx_mm_s", "vy_mm_s", "speed_mm_s", "r_mm", "cumdist_mm"]:
            det[c] = np.nan
        return df, det, meta

    # arrays
    t = det["time_s"].to_numpy()
    x = det["x_mm_c"].to_numpy()
    y = det["y_mm_c"].to_numpy()

    # [방어 1] 점프 제거
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx * dx + dy * dy)
    bad = step > max_step_mm
    if bad.any():
        x2 = x.copy()
        y2 = y.copy()
        x2[1:][bad] = np.nan
        y2[1:][bad] = np.nan
        x = safe_interp_nans(x2)
        y = safe_interp_nans(y2)

    # kinematics
    if len(det) >= 3:
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
    else:
        dt0 = (t[1] - t[0]) if (t[1] - t[0]) > 0 else 1.0 / 60.0
        vx = np.array([0.0, (x[1] - x[0]) / dt0])
        vy = np.array([0.0, (y[1] - y[0]) / dt0])

    speed = np.sqrt(vx * vx + vy * vy)

    # [방어 2] 속도 클리핑
    vx = np.where(np.isfinite(vx), vx, np.nan)
    vy = np.where(np.isfinite(vy), vy, np.nan)
    speed = np.where(np.isfinite(speed), speed, np.nan)
    speed = np.clip(speed, 0.0, speed_clip_mm_s)

    # distance
    step2 = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    step2 = np.where(np.isfinite(step2), step2, 0.0)
    cumdist = np.concatenate([[0.0], np.cumsum(step2)])

    det["x_mm_c"] = x
    det["y_mm_c"] = y
    det["vx_mm_s"] = vx
    det["vy_mm_s"] = vy
    det["speed_mm_s"] = speed
    det["cumdist_mm"] = cumdist
    det["r_mm"] = np.sqrt(x * x + y * y)

    return df, det, meta


# =====================
# bout 분석
# =====================
def bout_analysis(det: pd.DataFrame, stop_speed_thresh: float):
    s = det["speed_mm_s"].to_numpy()
    t = det["time_s"].to_numpy()
    ok = np.isfinite(s) & np.isfinite(t)
    s = s[ok]
    t = t[ok]
    if len(s) < 3:
        return {
            "stop_ratio": np.nan,
            "mean_stop_bout_s": 0.0,
            "mean_move_bout_s": 0.0,
            "n_stop_bouts": 0,
            "n_move_bouts": 0,
        }

    is_stop = s < stop_speed_thresh
    bouts = []
    cur = is_stop[0]
    t0 = t[0]
    for i in range(1, len(is_stop)):
        if is_stop[i] != cur:
            bouts.append((cur, t[i] - t0))
            cur = is_stop[i]
            t0 = t[i]
    bouts.append((cur, t[-1] - t0))

    stop_bouts = [d for st, d in bouts if st]
    move_bouts = [d for st, d in bouts if not st]
    total = t[-1] - t[0]

    return {
        "stop_ratio": float(np.sum(stop_bouts) / total) if total > 0 else np.nan,
        "mean_stop_bout_s": float(np.mean(stop_bouts)) if stop_bouts else 0.0,
        "mean_move_bout_s": float(np.mean(move_bouts)) if move_bouts else 0.0,
        "n_stop_bouts": int(len(stop_bouts)),
        "n_move_bouts": int(len(move_bouts)),
    }


def thigmotaxis_analysis(det: pd.DataFrame, dish_radius_mm: float):
    r = det["r_mm"].to_numpy()
    r = r[np.isfinite(r)]
    out = {}
    for th in EDGE_THRESHOLDS:
        out[f"edge_ratio_r_gt_{th}R"] = float(np.mean(r > th * dish_radius_mm)) if len(r) else np.nan
    return out


# =====================
# feature 요약
# =====================
def compute_summary(csv_path: Path, df_raw: pd.DataFrame, det: pd.DataFrame, meta: dict,
                    dish_diameter_mm: float, stop_speed_thresh: float, heat_bins: int):
    dish_radius_mm = dish_diameter_mm * 0.5

    if len(det) >= 2:
        duration = float(det["time_s"].iloc[-1] - det["time_s"].iloc[0])
        total_dist = float(det["cumdist_mm"].iloc[-1]) if np.isfinite(det["cumdist_mm"].iloc[-1]) else np.nan
    else:
        duration = 0.0
        total_dist = np.nan

    s = det["speed_mm_s"].to_numpy()
    s = s[np.isfinite(s)]

    # occupancy entropy
    if len(det) >= 20:
        H = occupancy_hist2d(det["x_mm_c"].to_numpy(), det["y_mm_c"].to_numpy(),
                             lim=dish_radius_mm, bins=heat_bins)
        p = H[H > 0]
        occ_ent = float(-np.sum(p * np.log(p))) if len(p) else np.nan
    else:
        occ_ent = np.nan

    bout = bout_analysis(det, stop_speed_thresh)
    thig = thigmotaxis_analysis(det, dish_radius_mm)

    runinfo = csv_path.stem.split("_")
    run_id = runinfo[0] if len(runinfo) >= 1 else ""
    date = runinfo[1] if len(runinfo) >= 2 else ""
    time_str = runinfo[2] if len(runinfo) >= 3 else ""

    summary = {
        "run": csv_path.stem,
        "file": csv_path.name,
        "run_id": run_id,
        "date": date,
        "time": time_str,

        "dish_diameter_mm": float(dish_diameter_mm),
        "stop_speed_thresh_mm_s": float(stop_speed_thresh),

        "n_raw": int(len(df_raw)),
        "n_det": int(len(det)),
        "found_rate_raw": float(meta.get("found_rate_raw", np.nan)),
        "mean_fps_raw": float(meta.get("mean_fps_raw", np.nan)),
        "time_axis_source": meta.get("time_axis_source", ""),

        "duration_s": duration,
        "mean_speed": float(np.mean(s)) if len(s) else np.nan,
        "median_speed": float(np.median(s)) if len(s) else np.nan,
        "p90_speed": float(np.percentile(s, 90)) if len(s) else np.nan,
        "max_speed": float(np.max(s)) if len(s) else np.nan,
        "total_distance_mm": total_dist,

        "mean_r_mm": float(np.nanmean(det["r_mm"])) if len(det) else np.nan,
        "p90_r_mm": float(np.nanpercentile(det["r_mm"], 90)) if len(det) else np.nan,
        "occupancy_entropy": occ_ent,

        "mean_area_mm2": float(np.nanmean(det["area_mm2_val"])) if "area_mm2_val" in det.columns else np.nan,
        "std_area_mm2": float(np.nanstd(det["area_mm2_val"])) if "area_mm2_val" in det.columns else np.nan,

        **bout,
        **thig,

        "pixel_to_mm_logged": float(meta.get("pixel_to_mm", np.nan)),
        "bin_thresh_logged": float(meta.get("bin_thresh", np.nan)),
    }
    return summary


def make_interpretation_text(summary: dict):
    def f(x, fmt):
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "NaN"
            return format(x, fmt)
        except Exception:
            return str(x)

    lines = []
    lines.append("[Free exploration behavior summary]")
    lines.append("")
    lines.append(f"File               : {summary.get('file','')}")
    lines.append(f"Run                : {summary.get('run','')}")
    lines.append(f"Time axis source   : {summary.get('time_axis_source','')}")
    lines.append("")
    lines.append(f"Duration           : {f(summary.get('duration_s'), '.2f')} s")
    lines.append(f"Mean speed         : {f(summary.get('mean_speed'), '.3f')} mm/s")
    lines.append(f"Median speed       : {f(summary.get('median_speed'), '.3f')} mm/s")
    lines.append(f"P90 speed          : {f(summary.get('p90_speed'), '.3f')} mm/s")
    lines.append(f"Max speed          : {f(summary.get('max_speed'), '.3f')} mm/s")
    lines.append(f"Total distance     : {f(summary.get('total_distance_mm'), '.2f')} mm")
    lines.append("")
    lines.append("[Stop/Move bout]")
    lines.append(f"Stop ratio         : {f(summary.get('stop_ratio'), '.3f')}")
    lines.append(f"Mean stop bout      : {f(summary.get('mean_stop_bout_s'), '.2f')} s")
    lines.append(f"Mean move bout      : {f(summary.get('mean_move_bout_s'), '.2f')} s")
    lines.append(f"# stop bouts        : {summary.get('n_stop_bouts',0)}")
    lines.append(f"# move bouts        : {summary.get('n_move_bouts',0)}")
    lines.append("")
    lines.append("[Spatial behavior / Thigmotaxis]")
    lines.append(f"Dish diameter      : {f(summary.get('dish_diameter_mm'), '.1f')} mm")
    lines.append(f"Mean r             : {f(summary.get('mean_r_mm'), '.2f')} mm")
    lines.append(f"P90 r              : {f(summary.get('p90_r_mm'), '.2f')} mm")
    for th in EDGE_THRESHOLDS:
        k = f"edge_ratio_r_gt_{th}R"
        lines.append(f"r > {th}R ratio     : {f(summary.get(k), '.3f')}")
    lines.append(f"Occupancy entropy  : {f(summary.get('occupancy_entropy'), '.3f')}")
    lines.append("")
    lines.append("[Data quality]")
    lines.append(f"Found rate (raw)   : {f(summary.get('found_rate_raw'), '.3f')}")
    lines.append(f"Mean FPS (raw)     : {f(summary.get('mean_fps_raw'), '.2f')}")
    lines.append(f"pixel_to_mm(log)   : {f(summary.get('pixel_to_mm_logged'), '.6f')} mm/px")
    lines.append(f"bin_thresh(log)    : {f(summary.get('bin_thresh_logged'), '.0f')}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Run-wise scalar descriptors (speed, bout statistics, spatial preference) were extracted under stimulus-free conditions.")
    lines.append("- Robust preprocessing was applied: time-axis reconstruction (when dt available), jump rejection, and speed clipping to mitigate tracking artifacts.")
    lines.append("- The resulting summary table can be directly used for later comparison with VR-dish experiments.")
    lines.append("")
    return "\n".join(lines)


# =====================
# 그림 저장 (업데이트: heatmap 잘 보이게 + speed-time 추가)
# =====================
def save_figs(df_raw: pd.DataFrame, det: pd.DataFrame, out_dir: Path,
              dish_diameter_mm: float, heat_bins: int, hist_bins: int,
              stop_speed_thresh: float):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dish_radius = dish_diameter_mm * 0.5

    # raw time axis
    t_raw = df_raw["time_s"].to_numpy()
    found = df_raw["found"].to_numpy()

    # dt vs time
    if "dt[s]" in df_raw.columns:
        dt_raw = _num(df_raw["dt[s]"]).to_numpy()
        ok = np.isfinite(t_raw) & np.isfinite(dt_raw)
        if ok.sum() >= 3:
            plt.figure()
            plt.plot(t_raw[ok], dt_raw[ok])
            plt.xlabel("Time (s)")
            plt.ylabel("dt (s)")
            plt.title("dt vs time")
            plt.savefig(fig_dir / "dt_vs_time.png", dpi=300)
            plt.close()

    # fps vs time
    fps = df_raw["fps_val"].to_numpy()
    ok = np.isfinite(t_raw) & np.isfinite(fps)
    if ok.sum() >= 3:
        plt.figure()
        plt.plot(t_raw[ok], fps[ok])
        plt.xlabel("Time (s)")
        plt.ylabel("FPS")
        plt.title("fps vs time")
        plt.savefig(fig_dir / "fps_vs_time.png", dpi=300)
        plt.close()

    # found vs time
    ok = np.isfinite(t_raw)
    if ok.sum() >= 3:
        plt.figure()
        plt.plot(t_raw[ok], found[ok])
        plt.xlabel("Time (s)")
        plt.ylabel("found (0/1)")
        plt.title("detection flag vs time")
        plt.savefig(fig_dir / "found_vs_time.png", dpi=300)
        plt.close()

    if len(det) < 2:
        return

    t = det["time_s"].to_numpy()
    x = det["x_mm_c"].to_numpy()
    y = det["y_mm_c"].to_numpy()

    okxy = np.isfinite(x) & np.isfinite(y)
    if okxy.sum() >= 2:
        xx = x[okxy]
        yy = y[okxy]

        # trajectory
        plt.figure()
        plt.plot(xx, yy, linewidth=1)
        plt.scatter([xx[0]], [yy[0]], s=30, marker="o")
        plt.scatter([xx[-1]], [yy[-1]], s=30, marker="x")
        plot_circle_boundary(dish_radius)
        plt.axis("equal")
        plt.xlabel("x (mm, centered)")
        plt.ylabel("y (mm, centered)")
        plt.title("Trajectory (after jump-reject)")
        plt.savefig(fig_dir / "trajectory.png", dpi=300)
        plt.close()

        # occupancy heatmap (better visibility)
        H = occupancy_hist2d(xx, yy, lim=dish_radius, bins=heat_bins)

        plt.figure()
        ax = plt.gca()
        ax.set_facecolor(HEATMAP_BG)
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[-dish_radius, dish_radius, -dish_radius, dish_radius],
            aspect="equal",
            cmap=HEATMAP_CMAP,
            interpolation="nearest",
        )
        plot_circle_boundary(dish_radius)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title("Occupancy heatmap (prob.)")
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.set_label("Probability")
        plt.savefig(fig_dir / "occupancy_heatmap.png", dpi=HEATMAP_DPI)
        plt.close()

    # position vs time
    okx = np.isfinite(t) & np.isfinite(x)
    oky = np.isfinite(t) & np.isfinite(y)
    if okx.sum() >= 3 or oky.sum() >= 3:
        plt.figure()
        if okx.sum() >= 3:
            plt.plot(t[okx], x[okx], label="x")
        if oky.sum() >= 3:
            plt.plot(t[oky], y[oky], label="y")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (mm)")
        plt.title("Position vs time")
        plt.legend()
        plt.savefig(fig_dir / "position_vs_time.png", dpi=300)
        plt.close()

    # speed plots (추가 강화)
    sp = det["speed_mm_s"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(sp)
    if ok.sum() >= 3:
        # 기본 speed vs time
        plt.figure()
        plt.plot(t[ok], sp[ok])
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (mm/s)")
        plt.title("Speed vs time (clipped)")
        plt.savefig(fig_dir / "speed_vs_time.png", dpi=300)
        plt.close()

        # 임계선 포함
        plt.figure()
        plt.plot(t[ok], sp[ok], linewidth=1)
        plt.axhline(stop_speed_thresh, linestyle="--", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (mm/s)")
        plt.title("Speed vs time (with stop threshold)")
        plt.savefig(fig_dir / "speed_vs_time_with_threshold.png", dpi=300)
        plt.close()

        # rolling mean (보기 좋게)
        win = max(5, int(ok.sum() * 0.01))  # 대략 1% 길이
        sp_rm = rolling_mean(sp, win)
        ok2 = np.isfinite(t) & np.isfinite(sp_rm)
        if ok2.sum() >= 3:
            plt.figure()
            plt.plot(t[ok], sp[ok], linewidth=0.8, alpha=0.5, label="raw")
            plt.plot(t[ok2], sp_rm[ok2], linewidth=1.5, label=f"rolling mean (win={win})")
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (mm/s)")
            plt.title("Speed vs time (smoothed view)")
            plt.legend()
            plt.savefig(fig_dir / "speed_vs_time_rolling.png", dpi=300)
            plt.close()

    # speed histogram
    sp2 = sp[np.isfinite(sp)]
    if len(sp2) >= 3:
        plt.figure()
        plt.hist(sp2, bins=hist_bins)
        plt.xlabel("Speed (mm/s)")
        plt.ylabel("Count")
        plt.title("Speed distribution (clipped)")
        plt.savefig(fig_dir / "speed_hist.png", dpi=300)
        plt.close()

    # cumulative distance
    cd = det["cumdist_mm"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(cd)
    if ok.sum() >= 3:
        plt.figure()
        plt.plot(t[ok], cd[ok])
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative distance (mm)")
        plt.title("Cumulative distance vs time")
        plt.savefig(fig_dir / "cumdist_vs_time.png", dpi=300)
        plt.close()

    # radial
    rr = det["r_mm"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(rr)
    if ok.sum() >= 3:
        plt.figure()
        plt.plot(t[ok], rr[ok])
        plt.xlabel("Time (s)")
        plt.ylabel("r (mm)")
        plt.title("Radial distance vs time")
        plt.savefig(fig_dir / "radial_vs_time.png", dpi=300)
        plt.close()

    rr2 = rr[np.isfinite(rr)]
    if len(rr2) >= 3:
        plt.figure()
        plt.hist(rr2, bins=hist_bins)
        plt.xlabel("r (mm)")
        plt.ylabel("Count")
        plt.title("Radial distance distribution")
        plt.savefig(fig_dir / "radial_hist.png", dpi=300)
        plt.close()

    # area (optional)
    if "area_mm2_val" in det.columns:
        a = det["area_mm2_val"].to_numpy()
        oka = np.isfinite(t) & np.isfinite(a)
        if oka.sum() >= 3:
            plt.figure()
            plt.plot(t[oka], a[oka])
            plt.xlabel("Time (s)")
            plt.ylabel("Area (mm^2)")
            plt.title("Projected area vs time")
            plt.savefig(fig_dir / "area_vs_time.png", dpi=300)
            plt.close()


# =====================
# run 분석
# =====================
def analyze_one(csv_path: Path, out_root: Path,
                dish_diameter_mm: float, stop_speed_thresh: float,
                heat_bins: int, hist_bins: int,
                max_step_mm: float, speed_clip_mm_s: float):
    out_dir = out_root / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, det, meta = load_and_process(csv_path, dish_diameter_mm, max_step_mm, speed_clip_mm_s)

    if len(det) < 2:
        print(f"[SKIP] too few detections: {csv_path.name}")
        return None

    det.to_csv(out_dir / "processed_timeseries.csv", index=False)

    summary = compute_summary(csv_path, df_raw, det, meta, dish_diameter_mm, stop_speed_thresh, heat_bins)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    text = make_interpretation_text(summary)
    (out_dir / "interpretation.txt").write_text(text, encoding="utf-8")

    save_figs(df_raw, det, out_dir, dish_diameter_mm, heat_bins, hist_bins, stop_speed_thresh)

    print(
        f"[OK] {csv_path.name} -> {out_dir} | "
        f"dur={summary['duration_s']:.2f}s "
        f"meanV={summary['mean_speed']:.3f}mm/s "
        f"dist={summary['total_distance_mm']:.2f}mm "
        f"stop={summary['stop_ratio']:.3f} "
        f"edge0.8R={summary.get('edge_ratio_r_gt_0.8R', np.nan):.3f} "
        f"time={summary.get('time_axis_source','')}"
    )
    return summary


# =====================
# 메인
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV 파일 1개 또는 CSV 폴더")
    ap.add_argument("--out", default="real_dish_analysis", help="출력 루트 폴더")

    ap.add_argument("--dish_diameter_mm", type=float, default=DISH_DIAMETER_MM_DEFAULT)
    ap.add_argument("--stop_speed_thresh", type=float, default=STOP_SPEED_THRESH_DEFAULT)
    ap.add_argument("--heat_bins", type=int, default=60)
    ap.add_argument("--hist_bins", type=int, default=60)

    ap.add_argument("--max_step_mm", type=float, default=MAX_STEP_MM_DEFAULT)
    ap.add_argument("--speed_clip_mm_s", type=float, default=SPEED_CLIP_MM_S_DEFAULT)

    ap.add_argument("--recursive", type=int, default=0, help="폴더 입력일 때 하위폴더까지(1/0)")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    if in_path.is_file():
        s = analyze_one(
            in_path, out_root,
            args.dish_diameter_mm, args.stop_speed_thresh,
            args.heat_bins, args.hist_bins,
            args.max_step_mm, args.speed_clip_mm_s
        )
        if s:
            rows.append(s)

    elif in_path.is_dir():
        csvs = sorted(in_path.rglob("*.csv") if args.recursive else in_path.glob("*.csv"))
        if not csvs:
            raise SystemExit(f"No CSV files in {in_path}")
        for p in csvs:
            s = analyze_one(
                p, out_root,
                args.dish_diameter_mm, args.stop_speed_thresh,
                args.heat_bins, args.hist_bins,
                args.max_step_mm, args.speed_clip_mm_s
            )
            if s:
                rows.append(s)
    else:
        raise SystemExit(f"Not found: {in_path}")

    if rows:
        df_all = pd.DataFrame(rows)
        df_all.to_csv(out_root / "ALL_real_summary.csv", index=False)
        print(f"[OK] updated: {out_root / 'ALL_real_summary.csv'}")

    print("[DONE]")


if __name__ == "__main__":
    main()
