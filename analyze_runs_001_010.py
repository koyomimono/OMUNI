#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# analyze_runs_001_010.py
# - Collect run_001 ~ run_010 from results_air_range/
# - Read optflow_run###_sigX.csv + meta.txt
# - Make:
#   (1) summary CSV
#   (2) big plots (per-run timeseries + across-run comparison)
#   (3) interpretation report txt
#
# Expected structure:
#   results_air_range/run_001/optflow_run001_sig5.csv
#   results_air_range/run_001/meta.txt
#   ...
#
# Notes:
# - Uses only CSV (robust, no OCR on images)
# - If a run is missing, it will be skipped and noted in report.

import os
import re
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
RESULTS_ROOT = "results_air_range"
RUN_START = 1
RUN_END = 10

# If you want to compare by Arduino "sig", it will be parsed from filename "..._sigX.csv"
# If missing, sig = -1

# Plots
SAVE_DPI = 300
FIG_W = 18
FIG_H = 10

# Interpretation thresholds (adjust if you want)
ACTIVE_MASK_RATIO_TH = 0.002   # average masked_ratio above this => "clear activity"
ACTIVE_FRAC_TH = 0.15          # fraction of active frames above this => "frequent"
NOISE_WARN_MAD_TH = 0.8        # median MAD above this => "noisy-ish" (heuristic)


# =========================
# Helpers
# =========================
def read_meta_txt(meta_path):
    meta = {}
    if not os.path.isfile(meta_path):
        return meta
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()
    except Exception:
        pass
    return meta


def find_optflow_csv(run_dir, run_id):
    # prefer exact match optflow_run###_sigX.csv
    pat = re.compile(rf"^optflow_run{run_id:03d}_sig(\d+)\.csv$")
    if not os.path.isdir(run_dir):
        return None, None
    for name in os.listdir(run_dir):
        m = pat.match(name)
        if m:
            sig = int(m.group(1))
            return os.path.join(run_dir, name), sig
    # fallback: any optflow*.csv
    for name in os.listdir(run_dir):
        if name.lower().endswith(".csv") and "optflow" in name.lower():
            # try parse sig
            mm = re.search(r"_sig(\d+)\.csv$", name)
            sig = int(mm.group(1)) if mm else -1
            return os.path.join(run_dir, name), sig
    return None, None


def read_optflow_csv(csv_path):
    # Returns dict of numpy arrays
    # Columns written by your acquisition code:
    # t_meas_s, med_mag_full, mad_mag_full, floor, masked_area_px, masked_ratio,
    # mean_mag_masked_fullpx, mean_vx_small, mean_vy_small, mean_mag_small_masked
    data = {
        "t": [],
        "med": [],
        "mad": [],
        "floor": [],
        "masked_area": [],
        "masked_ratio": [],
        "mean_mag_masked": [],
        "mean_vx": [],
        "mean_vy": [],
        "mean_mag_small": [],
    }

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                data["t"].append(float(row.get("t_meas_s", "0")))
                data["med"].append(float(row.get("med_mag_full", "0")))
                data["mad"].append(float(row.get("mad_mag_full", "0")))
                data["floor"].append(float(row.get("floor", "0")))
                data["masked_area"].append(float(row.get("masked_area_px", "0")))
                data["masked_ratio"].append(float(row.get("masked_ratio", "0")))
                data["mean_mag_masked"].append(float(row.get("mean_mag_masked_fullpx", "0")))
                data["mean_vx"].append(float(row.get("mean_vx_small", "0")))
                data["mean_vy"].append(float(row.get("mean_vy_small", "0")))
                data["mean_mag_small"].append(float(row.get("mean_mag_small_masked", "0")))
            except Exception:
                continue

    for k in data:
        data[k] = np.asarray(data[k], dtype=np.float64)
    return data


def robust_stats(x):
    # returns mean, median, p90, max
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(x))
    med = float(np.median(x))
    p90 = float(np.percentile(x, 90))
    mx = float(np.max(x))
    return mean, med, p90, mx


def safe_div(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


# =========================
# Main
# =========================
def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_ROOT, f"analysis_runs_{RUN_START:03d}_{RUN_END:03d}_{now}")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    missing_runs = []
    per_run_data = {}  # run_id -> dict

    # ---------
    # Load runs
    # ---------
    for run_id in range(RUN_START, RUN_END + 1):
        run_name = f"run_{run_id:03d}"
        run_dir = os.path.join(RESULTS_ROOT, run_name)

        meta = read_meta_txt(os.path.join(run_dir, "meta.txt"))
        csv_path, sig = find_optflow_csv(run_dir, run_id)

        if csv_path is None or (not os.path.isfile(csv_path)):
            missing_runs.append(run_name)
            continue

        data = read_optflow_csv(csv_path)
        per_run_data[run_id] = {
            "run_name": run_name,
            "run_dir": run_dir,
            "csv_path": csv_path,
            "sig": sig,
            "meta": meta,
            "data": data,
        }

        t = data["t"]
        dur = float(np.max(t)) if t.size > 0 else 0.0
        n = int(t.size)

        masked_ratio = data["masked_ratio"]
        mean_mag_masked = data["mean_mag_masked"]
        mad = data["mad"]
        floor = data["floor"]

        # Active frames = masked_area > 0 (or masked_ratio > 0)
        active = (masked_ratio > 0.0)
        active_frames = int(np.count_nonzero(active))
        active_frac = safe_div(active_frames, n)

        # Stats
        mr_mean, mr_med, mr_p90, mr_max = robust_stats(masked_ratio)
        mm_mean, mm_med, mm_p90, mm_max = robust_stats(mean_mag_masked)
        mad_mean, mad_med, mad_p90, mad_max = robust_stats(mad)
        fl_mean, fl_med, fl_p90, fl_max = robust_stats(floor)

        summary_rows.append({
            "run": run_name,
            "sig": sig,
            "csv": os.path.basename(csv_path),
            "duration_s": f"{dur:.3f}",
            "n_frames": str(n),
            "active_frames": str(active_frames),
            "active_frac": f"{active_frac:.4f}",
            "masked_ratio_mean": f"{mr_mean:.6g}",
            "masked_ratio_p90": f"{mr_p90:.6g}",
            "masked_ratio_max": f"{mr_max:.6g}",
            "mean_mag_masked_mean": f"{mm_mean:.6g}",
            "mean_mag_masked_p90": f"{mm_p90:.6g}",
            "mean_mag_masked_max": f"{mm_max:.6g}",
            "mad_median": f"{mad_med:.6g}",
            "floor_median": f"{fl_med:.6g}",
        })

    # -----------------
    # Save summary CSV
    # -----------------
    summary_csv_path = os.path.join(out_dir, f"summary_runs_{RUN_START:03d}_{RUN_END:03d}.csv")
    fields = [
        "run", "sig", "csv",
        "duration_s", "n_frames",
        "active_frames", "active_frac",
        "masked_ratio_mean", "masked_ratio_p90", "masked_ratio_max",
        "mean_mag_masked_mean", "mean_mag_masked_p90", "mean_mag_masked_max",
        "mad_median", "floor_median",
    ]
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # -----------------
    # Plot: across runs
    # -----------------
    if len(summary_rows) > 0:
        runs = [r["run"] for r in summary_rows]
        sigs = [int(r["sig"]) for r in summary_rows]
        active_frac = [float(r["active_frac"]) for r in summary_rows]
        mm_mean = [float(r["mean_mag_masked_mean"]) for r in summary_rows]
        mr_mean = [float(r["masked_ratio_mean"]) for r in summary_rows]
        mad_med = [float(r["mad_median"]) for r in summary_rows]

        x = np.arange(len(runs))

        plt.figure(figsize=(FIG_W, FIG_H))
        plt.plot(x, active_frac, marker="o", label="active_frac")
        plt.plot(x, mr_mean, marker="o", label="masked_ratio_mean")
        plt.plot(x, mm_mean, marker="o", label="mean_mag_masked_mean")
        plt.plot(x, mad_med, marker="o", label="mad_median")
        plt.xticks(x, [f"{runs[i]}\n(sig{str(sigs[i])})" for i in range(len(runs))], rotation=0)
        plt.xlabel("Run")
        plt.ylabel("Value (mixed scales)")
        plt.title("Across-run overview (activity / mask / motion / noise)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        across_plot_path = os.path.join(out_dir, "plot_across_runs_overview.png")
        plt.savefig(across_plot_path, dpi=SAVE_DPI)
        plt.close()

    # -----------------
    # Plot: per-run time series (big)
    # -----------------
    for run_id, pack in per_run_data.items():
        data = pack["data"]
        t = data["t"]
        if t.size == 0:
            continue

        masked_ratio = data["masked_ratio"]
        mean_mag_masked = data["mean_mag_masked"]
        mad = data["mad"]
        floor = data["floor"]

        plt.figure(figsize=(FIG_W, FIG_H))
        plt.plot(t, masked_ratio, label="masked_ratio")
        plt.plot(t, mean_mag_masked, label="mean_mag_masked_fullpx")
        plt.plot(t, mad, label="mad_mag_full")
        plt.plot(t, floor, label="floor")
        plt.xlabel("t_meas (s)")
        plt.title(f"{pack['run_name']} (sig={pack['sig']}) time series")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        p = os.path.join(out_dir, f"plot_timeseries_{pack['run_name']}_sig{pack['sig']}.png")
        plt.savefig(p, dpi=SAVE_DPI)
        plt.close()

    # -----------------
    # Interpretation report
    # -----------------
    report_path = os.path.join(out_dir, "interpretation_report.txt")
    lines = []
    lines.append("=== Air-range experiment summary & interpretation ===")
    lines.append(f"Created: {now}")
    lines.append(f"Runs: {RUN_START:03d} ~ {RUN_END:03d}")
    lines.append("")

    if len(missing_runs) > 0:
        lines.append("[Missing runs]")
        for r in missing_runs:
            lines.append(f"- {r} (CSV not found)")
        lines.append("")

    if len(summary_rows) == 0:
        lines.append("No valid runs found. Check folder structure and CSV filenames.")
    else:
        lines.append("[Key observations]")
        # Sort by run order already
        for r in summary_rows:
            run = r["run"]
            sig = int(r["sig"])
            af = float(r["active_frac"])
            mr = float(r["masked_ratio_mean"])
            mm = float(r["mean_mag_masked_mean"])
            mad_med = float(r["mad_median"])

            # simple labels
            if mr >= ACTIVE_MASK_RATIO_TH and af >= ACTIVE_FRAC_TH:
                activity = "CLEAR activity"
            elif af > 0.01:
                activity = "weak / intermittent activity"
            else:
                activity = "almost no activity"

            if mad_med >= NOISE_WARN_MAD_TH:
                noise = "noise seems HIGH (check lighting/flicker/ROI)"
            else:
                noise = "noise OK"

            lines.append(
                f"- {run} (sig={sig}): active_frac={af:.3f}, "
                f"masked_ratio_mean={mr:.4g}, mean_mag_masked_mean={mm:.4g} -> {activity}; {noise}"
            )

        lines.append("")
        lines.append("[How to interpret these numbers]")
        lines.append("- active_frac: fraction of frames where motion mask detected anything.")
        lines.append("- masked_ratio_mean: how much area was moving on average (higher = larger moving region).")
        lines.append("- mean_mag_masked_mean: average optical-flow magnitude inside mask (higher = faster movement).")
        lines.append("- mad_median / floor_median: proxy of noise floor; larger often means flicker or sensor noise.")
        lines.append("")
        lines.append("[Practical advice]")
        lines.append("- If only top-right is active but center is not: flicker_mask / lighting gradient is dominating.")
        lines.append("- If noise is high: stabilize lighting (DC light), reduce reflections, lock exposure/gain if possible.")
        lines.append("- If activity is too low despite movement: MIN_MOVE_MM might be too strict or ROI thresholds too high.")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -----------------
    # Print outputs
    # -----------------
    print("=== DONE ===")
    print(f"[OUT_DIR] {out_dir}")
    print(f"[SUMMARY] {summary_csv_path}")
    print(f"[REPORT]  {report_path}")
    print("Plots saved in the same folder.")


if __name__ == "__main__":
    main()
