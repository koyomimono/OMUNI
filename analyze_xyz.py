#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZ omniwheel log analyzer (newest CSV -> run_xyz/)
- Auto-picks newest 'step_response_xyz_*.csv' (fallback: 'step_response_xyz.csv')
- Recreates x_ref, y_ref, z_ref (must match robot-side params)
- Reconstructs x_meas, y_meas from ex/ey convention in logger
- Estimates time delay (Δt) via cross-correlation (ref vs meas)
- Reports Δt [s] and std-dev of (aligned meas - ref) on each plot
- Annotates PID gains from CSV (PX/IX/DX, PY/IY/DY, PZ/IZ/DZ) if present
- Saves plots into run_xyz/x_time_<timestamp>.png etc.
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Reference parameters (MUST match your robot code) =====
AMP_MM   = 15.0                 # [mm] for x,y
OMEGA    = math.pi / 2.0        # [rad/s] (period = 4 s)
X_BIAS   = 0.0
Y_BIAS   = 0.0

Z_BASE   = math.radians(90.0)   # [rad]
AMP_Z    = math.radians(15.0)   # [rad]
OMEGA_Z  = math.pi / 5.0        # [rad/s] (period = 10 s)

# ===== Helpers =====
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def find_latest_csv(cli_arg: str | None) -> str:
    """Pick CSV path: CLI arg > newest step_response_xyz_*.csv > step_response_xyz.csv"""
    if cli_arg and os.path.isfile(cli_arg):
        return cli_arg
    cand = sorted(glob.glob("step_response_xyz_*.csv"))
    if cand:
        return cand[-1]
    if os.path.isfile("step_response_xyz.csv"):
        return "step_response_xyz.csv"
    raise FileNotFoundError(
        "No CSV found. Put 'step_response_xyz_YYYYMMDD_HHMMSS.csv' or 'step_response_xyz.csv' "
        "here, or pass a path as an argument."
    )

def estimate_delay_via_xcorr(t: np.ndarray, ref: np.ndarray, meas: np.ndarray) -> float:
    """
    Estimate time delay Δt (seconds) where meas ≈ ref shifted by Δt.
    Positive Δt means meas LAGS ref (meas(t) ≈ ref(t - Δt)).
    Uses cross-correlation with de-meaned signals.
    """
    # Make sure shapes are ok and finite
    mask = np.isfinite(ref) & np.isfinite(meas)
    ref = ref[mask]
    meas = meas[mask]
    t = t[mask]
    if len(ref) < 5:
        return 0.0
    # Assume roughly uniform sampling; use median dt
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0

    # De-mean to avoid bias
    xr = ref - np.mean(ref)
    ym = meas - np.mean(meas)

    # Cross-correlation
    c = np.correlate(xr, ym, mode="full")
    lags = np.arange(-len(xr)+1, len(xr))
    k = int(np.argmax(c))
    lag_samples = lags[k]
    # Interpretation: if meas is delayed by +Δt, cross-corr peaks at +lag.
    delay_sec = lag_samples * dt
    return delay_sec

def align_series(t: np.ndarray, y: np.ndarray, delay_sec: float) -> np.ndarray:
    """
    Shift signal y by -delay (advance) onto t-grid:
      y_aligned(t) = y(t + delay_sec)
    So if delay_sec>0 (meas lags), we advance meas to align with ref.
    """
    return np.interp(t, t + delay_sec, y, left=np.nan, right=np.nan)

def safe_std(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.std(a, ddof=0))

def read_gain(df: pd.DataFrame, p: str, i: str, d: str):
    """Return (P,I,D) as strings for annotation (or 'N/A' if missing)."""
    def get(name):
        return f"{df[name].iloc[0]:.3f}" if name in df.columns else "N/A"
    return get(p), get(i), get(d)

# ===== Load CSV =====
csv_path = find_latest_csv(sys.argv[1] if len(sys.argv) > 1 else None)
print(f"[info] Using CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Required columns for reconstruction
required = ["time[s]", "ex[mm]", "ey[mm]", "z_meas[rad]", "z_ref[rad]"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ===== Time base =====
t = df["time[s]"].to_numpy(dtype=float)

# ===== Recreate references in math frame (right+, up+) =====
x_ref = X_BIAS + AMP_MM * np.sin(OMEGA * t)
y_ref = Y_BIAS + AMP_MM * np.cos(OMEGA * t)
z_ref = wrap_pi(Z_BASE + AMP_Z * np.sin(OMEGA_Z * t))

# ===== Reconstruct measured x,y from logged errors =====
# Logger convention:
#   ex[mm]  = ex_cam (camera: right+, down+)
#   ey[mm]  = ey_math (math: right+, up+)
ex_cam  = df["ex[mm]"].to_numpy(dtype=float)
ey_math = df["ey[mm]"].to_numpy(dtype=float)

# Reference -> camera
x_ref_cam = x_ref
y_ref_cam = -y_ref

# Camera-frame measured
ey_cam = -ey_math
x_meas_cam = x_ref_cam + ex_cam
y_meas_cam = y_ref_cam + ey_cam

# Back to math frame for plotting
x_meas = x_meas_cam
y_meas = -y_meas_cam

# Z (angles)
z_meas = wrap_pi(df["z_meas[rad]"].to_numpy(dtype=float))
z_ref_csv = wrap_pi(df["z_ref[rad]"].to_numpy(dtype=float))  # use logged ref for display

# ===== Estimate time delay (Δt) and std-dev for each axis =====
# X
delay_x = estimate_delay_via_xcorr(t, x_ref, x_meas)
x_meas_aligned = align_series(t, x_meas, delay_x)
err_x = x_meas_aligned - x_ref
std_x = safe_std(err_x)

# Y
delay_y = estimate_delay_via_xcorr(t, y_ref, y_meas)
y_meas_aligned = align_series(t, y_meas, delay_y)
err_y = y_meas_aligned - y_ref
std_y = safe_std(err_y)

# Z (use angular wrap)
delay_z = estimate_delay_via_xcorr(t, z_ref, z_meas)
z_meas_aligned = align_series(t, z_meas, delay_z)
# angular error after alignment
ang_err = wrap_pi(z_meas_aligned - z_ref)
std_z_deg = float(np.std(np.rad2deg(ang_err)[np.isfinite(ang_err)], ddof=0))

# Also compute phase (deg) from delays for display
phase_x_deg = float(np.degrees(delay_x * OMEGA))
phase_y_deg = float(np.degrees(delay_y * OMEGA))
phase_z_deg = float(np.degrees(delay_z * OMEGA_Z))

# ===== Read PID gains from CSV if present =====
PX, IX, DX = read_gain(df, "PX", "IX", "DX")
PY, IY, DY = read_gain(df, "PY", "IY", "DY")
PZ, IZ, DZ = read_gain(df, "PZ", "IZ", "DZ")

# ===== Output dir & filenames =====
outdir = "run_xyz"
os.makedirs(outdir, exist_ok=True)
base = os.path.splitext(os.path.basename(csv_path))[0]
suffix = base.replace("step_response_xyz_", "")

# ===== Plot X =====
plt.figure(figsize=(9, 4.8))
plt.plot(t, x_ref, "--", label="x_ref [mm]")
plt.plot(t, x_meas, label="x_meas [mm]")
plt.xlabel("time [s]"); plt.ylabel("x [mm]")
plt.title("X time series (ref vs meas)")
plt.grid(True, alpha=0.3); plt.legend()

txt = (
    f"PID (X): P={PX}, I={IX}, D={DX}\n"
    f"Phase diff: {phase_x_deg:+.2f}°\n"
    f"Time delay Δt: {delay_x:+.3f} s\n"
    f"σ(error): {std_x:.3f} mm"
)
plt.gca().text(0.98, 0.02, txt, transform=plt.gca().transAxes,
               va="bottom", ha="right")
plt.tight_layout()
x_path = os.path.join(outdir, f"x_time_{suffix}.png")
plt.savefig(x_path, dpi=150)

# ===== Plot Y =====
plt.figure(figsize=(9, 4.8))
plt.plot(t, y_ref, "--", label="y_ref [mm]")
plt.plot(t, y_meas, label="y_meas [mm]")
plt.xlabel("time [s]"); plt.ylabel("y [mm]")
plt.title("Y time series (ref vs meas)")
plt.grid(True, alpha=0.3); plt.legend()

txt = (
    f"PID (Y): P={PY}, I={IY}, D={DY}\n"
    f"Phase diff: {phase_y_deg:+.2f}°\n"
    f"Time delay Δt: {delay_y:+.3f} s\n"
    f"σ(error): {std_y:.3f} mm"
)
plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
               va="top", ha="left")
plt.tight_layout()
y_path = os.path.join(outdir, f"y_time_{suffix}.png")
plt.savefig(y_path, dpi=150)

# ===== Plot Z =====
plt.figure(figsize=(9, 4.8))
plt.plot(t, np.rad2deg(z_ref_csv), "--", label="z_ref [deg]")
plt.plot(t, np.rad2deg(z_meas), label="z_meas [deg]")
plt.xlabel("time [s]"); plt.ylabel("angle [deg]")
plt.title("Z (attitude) time series (ref vs meas)")
plt.grid(True, alpha=0.3); plt.legend()

txt = (
    f"PID (Z): P={PZ}, I={IZ}, D={DZ}\n"
    f"Phase diff: {phase_z_deg:+.2f}°\n"
    f"Time delay Δt: {delay_z:+.3f} s\n"
    f"σ(error): {std_z_deg:.3f} deg"
)
plt.gca().text(0.02, 0.02, txt, transform=plt.gca().transAxes,
               va="bottom", ha="left")
plt.tight_layout()
z_path = os.path.join(outdir, f"z_time_{suffix}.png")
plt.savefig(z_path, dpi=150)

print(f"Saved in '{outdir}/':")
print(f"  - {os.path.basename(x_path)}")
print(f"  - {os.path.basename(y_path)}")
print(f"  - {os.path.basename(z_path)}")
