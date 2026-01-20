# -*- coding: utf-8 -*-
# 三輪120°オムニホイール制御 (x軸専用チューナ)
# 参照: x_ref(t)=A*sin(ωt), y_ref(t)=0
# 手順: 段階探索(2パス, 順序=P→D→I) → 座標降下Refine(順序=P→D→I) → 最終評価

import os, sys, cv2, math, csv, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---- 環境依存ライブラリ ----
from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# ========= カメラ/画像/幾何 =========
CAMERA_INDEX = 0
WIDTH, HEIGHT = 640, 480
FPS_TARGET = 60
WAIT = 1
CROP_LEFT = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT
FRAME_CENTER = (HEIGHT // 2, HEIGHT // 2)
PIXEL_MM = 63.0 / 480.0
DEVICE_PATH = '/dev/input/event9'
SCALING = 0.0172

ANGLE_ALPHA = 0.2

# 車輪配分(あなたの実機に合わせる)
THETA = np.radians([90.0, 120.0, 240.0])
DIR_SGN = np.array([+1, +1, +1], dtype=float)
R_SPIN = 1.0
CMD_MAX = 127
SPEED_TO_CMD = 0.5

def wrap_pi(a): return (a + np.pi) % (2 * np.pi) - np.pi
def clamp(x, lo, hi): return max(lo, min(hi, x))

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened(): return None
    cv2.namedWindow('Track', flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    ok, _ = cap.read()
    return cap if ok else None

def gray_binary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0: return c, None
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    return c, (cx, cy)

def fit_ellipse_if_possible(contour):
    if contour is not None and len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        angle_deg = ellipse[2]
        if ellipse[1][0] < ellipse[1][1]: angle_deg += 90.0
        angle_deg %= 180.0
        return ellipse, math.radians(angle_deg)
    return None, None

def draw_overlay(frame, center, ex_math, ey_math, ellipse, fps, psi_smooth, is_running, elapsed):
    cv2.circle(frame, FRAME_CENTER, 5, (255,0,0), -1)
    if center is not None:
        cv2.circle(frame, center, 5, (0,255,0), -1)
        cv2.putText(frame, f"e[mm]=({ex_math:+.2f},{ey_math:+.2f})", (10,30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
    if ellipse is not None and psi_smooth is not None:
        cv2.putText(frame, f"Angle: {math.degrees(psi_smooth):.2f} deg", (10,60),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(frame, f"state: {'RUNNING' if is_running else 'IDLE'}  t={elapsed:.2f}s",
                (10,120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

def wheels_command_from_v(vx, vy, omega):
    v_wheels = (omega - vx*np.sin(THETA) + vy*np.cos(THETA) + R_SPIN*omega) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds):
    # 実機マッピング: m1<-cmd[2], m2<-cmd[1], m3<-cmd[0]
    motor_m1(int(cmds[2])); motor_m2(int(cmds[1])); motor_m3(int(cmds[0]))

# 参照 (x-only)
def ref_xy_math_x(t, A, W):
    return A*math.sin(W*t), 0.0

# CSV
def initialize_csv_logger(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "time[s]","dt[s]","fps",
            "ex[mm]","ey[mm]","dex[mm/s]","dey[mm/s]",
            "ix","iy",
            "psi[rad]","epsi[rad]","psi_i","dpsi[rad/s]",
            "vx[mm/s]","vy[mm/s]","omega[rad/s]",
            "cmd1","cmd2","cmd3",
            "vwh1[mm/s]","vwh2[mm/s]","vwh3[mm/s]"
        ])

def log_to_csv_buffer(buf, **kw):
    buf.append([
        f"{kw['t']:.6f}", f"{kw['dt']:.6f}", f"{kw['fps']:.3f}",
        f"{kw['ex']:.6f}", f"{kw['ey']:.6f}", f"{kw['dex']:.6f}", f"{kw['dey']:.6f}",
        f"{kw['ix']:.6f}", f"{kw['iy']:.6f}",
        f"{kw['psi']:.6f}", f"{kw['epsi']:.6f}", f"{kw['psi_i']:.6f}", f"{kw['dpsi']:.6f}",
        f"{kw['vx']:.6f}", f"{kw['vy']:.6f}", f"{kw['omega']:.6f}",
        int(kw['cmds'][0]), int(kw['cmds'][1]), int(kw['cmds'][2]),
        f"{kw['vwh'][0]:.6f}", f"{kw['vwh'][1]:.6f}", f"{kw['vwh'][2]:.6f}",
    ])

def flush_log_entries(path, entries):
    if not entries: return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows(entries)
    entries.clear()

# 実験(1本) — x軸PIDのみ
def run_experiment(csv_path, A, W, duration, Kp, Ki, Kd,
                   I_LIM=50.0, show_window=False):
    mouse_x=mouse_y=0.0
    def mouse_callback(x, y):
        nonlocal mouse_x, mouse_y
        mouse_x, mouse_y = x, y

    mt = MouseTracker(DEVICE_PATH, SCALING); mt.start(callback=mouse_callback)
    initialize_csv_logger(csv_path); log_entries=[]
    cap = initialize_camera()
    if not cap:
        print("カメラが開けない"); mt.stop(); return False

    ix=0.0; ex_prev=None
    iy=0.0; ey_prev=None  # yは制御しないがログは取る
    psi_smooth=None; psi_prev=None; psi_i=0.0

    t0=time.time(); last_log=time.time(); prev_time=time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            now=time.time()
            dt = clamp(now - prev_time, 1e-3, 0.1)
            prev_time = now
            fps = 1.0/dt
            elapsed = now - t0

            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
            mask = gray_binary(frame_cropped)
            contour, center = find_largest_contour(mask)
            ellipse, angle_rad = fit_ellipse_if_possible(contour)

            if angle_rad is not None:
                psi_raw = wrap_pi(angle_rad)
                psi_smooth = psi_raw if psi_smooth is None else wrap_pi(psi_smooth + ANGLE_ALPHA*wrap_pi(psi_raw-psi_smooth))
            else:
                psi_raw = psi_smooth

            # 参照 (x=A sin, y=0) → カメラ系
            xref_m, yref_m = ref_xy_math_x(elapsed, A, W)
            xref_cam, yref_cam = xref_m, -yref_m  # yは0

            if center is not None:
                x_cam = (center[0] - FRAME_CENTER[0]) * PIXEL_MM
                y_cam = (center[1] - FRAME_CENTER[1]) * PIXEL_MM
                ex_cam = x_cam - xref_cam
                ey_cam = y_cam - yref_cam
            else:
                ex_cam = ey_cam = 0.0

            ex_math = ex_cam; ey_math = -ey_cam

            # xのみPID
            ix = clamp(ix + ex_cam*dt, -I_LIM, I_LIM)
            if ex_prev is None: dex=0.0
            else: dex = (ex_cam - ex_prev)/dt
            ex_prev = ex_cam
            vx = Kp*ex_cam + Ki*ix + Kd*dex

            # yはゼロ指令 (保持)
            vy = 0.0
            dey = 0.0

            omega = 0.0  # 姿勢制御オフ

            cmds, vwh = wheels_command_from_v(vx, vy, omega)
            move_motors_cmds(cmds)

            if (now - last_log) >= 0.0:
                log_to_csv_buffer(log_entries,
                    t=elapsed, dt=dt, fps=fps,
                    ex=ex_cam, ey=ey_math,
                    dex=dex, dey=dey,
                    ix=ix, iy=iy,
                    psi=(psi_smooth if psi_smooth is not None else 0.0),
                    epsi=0.0, psi_i=0.0, dpsi=0.0,
                    vx=vx, vy=vy, omega=omega,
                    cmds=cmds, vwh=vwh
                )
                last_log = now

            if show_window:
                draw_overlay(frame_cropped, center, ex_math, ey_math, ellipse, fps, psi_smooth, True, elapsed)
                if (cv2.waitKey(WAIT) & 0xFF) == ord('q'):
                    break

            if elapsed >= duration: break

        stop_all(); flush_log_entries(csv_path, log_entries)
        return True
    finally:
        stop_all()
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        mt.stop()
        flush_log_entries(csv_path, log_entries)

# 評価 (xのみ)
def analyze_x(csv_path, A, W, out_prefix, Kp, Ki, Kd):
    df = pd.read_csv(csv_path)
    t = df['time[s]'].to_numpy(float)
    ex = df['ex[mm]'].to_numpy(float)
    x_ref = A*np.sin(W*t)
    x_act = x_ref + ex

    # メトリクス
    bias = float(np.mean(ex))
    mae  = float(np.mean(np.abs(ex)))
    rmse = float(np.sqrt(np.mean(ex**2)))

    # 遅れ推定
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1/60.0
    xr = x_ref - np.mean(x_ref); xa = x_act - np.mean(x_act)
    if np.std(xr) > 1e-9 and np.std(xa) > 1e-9:
        corr = np.correlate(xr, xa, mode='full')
        lags = np.arange(-len(xr)+1, len(xr))
        lag  = int(lags[int(np.argmax(corr))])
        tau  = lag * dt_med
        phi  = (tau * W) * 180.0 / math.pi
    else:
        tau = None; phi = None

    # プロット
    plt.figure(figsize=(9,4.5), dpi=120)
    plt.plot(t, x_act, label="x_actual (mm)")
    plt.plot(t, x_ref, linestyle="--", label="x_ref (mm)")
    plt.xlabel("time (s)"); plt.ylabel("x (mm)")
    plt.title("X position: actual vs reference")
    plt.grid(True, linestyle=":")
    txt = f"P={Kp}, I={Ki}, D={Kd}\nRMSE={rmse:.3f}, MAE={mae:.3f}, bias={bias:.3f}\n"
    if phi is not None: txt += f"delay≈{tau*1e3:.0f} ms (φ≈{phi:.1f}°)"
    plt.gca().text(0.01,0.98,txt,transform=plt.gca().transAxes,ha="left",va="top",
                   bbox=dict(boxstyle="round", alpha=0.15))
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_prefix}_x.png"); plt.close()

    return {"rmse":rmse, "mae":mae, "bias":bias, "phi":phi}

def objective_x(m, w_rmse=1.0, w_bias=0.2, w_phi=0.1):
    phi = 0.0 if m["phi"] is None else abs(m["phi"])
    return float(w_rmse*m["rmse"] + w_bias*abs(m["bias"]) + w_phi*phi)

# 1D ラインサーチ
def line_search(args, stage, var_name, init_val, init_step, fixed, outdir, run_idx):
    val = float(init_val); step=float(init_step)
    best_val=None; bestJ=None
    while step >= args.tol:
        cands=[]
        for d in [0.0, +step, -step]:
            v = val + d
            if var_name=="Kp": v = float(clamp(v, 0.0, args.Kp_max))
            if var_name=="Ki": v = float(clamp(v, 0.0, args.Ki_max))
            if var_name=="Kd": v = float(clamp(v, 0.0, args.Kd_max))
            Kp,Ki,Kd = fixed["Kp"], fixed["Ki"], fixed["Kd"]
            if var_name=="Kp": Kp=v
            if var_name=="Ki": Ki=v
            if var_name=="Kd": Kd=v
            run_idx += 1
            tag = f"run_{run_idx:02d}_{stage}_{var_name}_{v:.4f}"
            csvp = os.path.join(outdir, f"{tag}.csv")
            ok = run_experiment(csvp, args.amp, args.omega, args.duration, Kp, Ki, Kd, show_window=args.show)
            if not ok: continue
            met = analyze_x(csvp, args.amp, args.omega, os.path.join(outdir, tag), Kp, Ki, Kd)
            J = objective_x(met, args.w_rmse, args.w_bias, args.w_phi)
            with open(os.path.join(outdir, "history.csv"), "a", newline="") as f:
                csv.writer(f).writerow([stage, tag, Kp, Ki, Kd, J, met["rmse"], met["mae"], met["bias"], ("" if met["phi"] is None else met["phi"])])
            cands.append((J,v))
        if not cands: break
        cands.sort(key=lambda z:z[0]); Jb, vb = cands[0]
        if (bestJ is None) or (Jb+1e-9<bestJ):
            val=vb; best_val=vb; bestJ=Jb
            print(f"[{stage}] {var_name} 改善: {val:.4f}, J={bestJ:.4f}")
        else:
            step*=0.5; print(f"[{stage}] 改善なし → Δ半減: {step:.4g}")
    return (best_val if best_val is not None else val), (bestJ if bestJ is not None else float('inf')), run_idx

# 仕上げ(Kp→Kd→Ki) の座標降下
def refine_pass(args, Kp,Ki,Kd, outdir, run_idx, label, base_dKp, base_dKi, base_dKd):
    def one_axis(name, val, step, fixed):
        nonlocal run_idx
        if step < args.refine_tol: return val, step, run_idx
        while step >= args.refine_tol:
            best=None
            for d in [0.0, +step, -step]:
                v = val + d
                if name=="Kp": v=float(clamp(v,0, args.Kp_max))
                if name=="Ki": v=float(clamp(v,0, args.Ki_max))
                if name=="Kd": v=float(clamp(v,0, args.Kd_max))
                Kp_c,Ki_c,Kd_c = fixed["Kp"], fixed["Ki"], fixed["Kd"]
                if name=="Kp": Kp_c=v
                if name=="Ki": Ki_c=v
                if name=="Kd": Kd_c=v
                run_idx+=1
                tag=f"run_{run_idx:02d}_{label}_{name}_{v:.4f}"
                csvp=os.path.join(outdir, f"{tag}.csv")
                ok=run_experiment(csvp, args.amp, args.omega, args.duration, Kp_c, Ki_c, Kd_c, show_window=args.show)
                if not ok: continue
                met=analyze_x(csvp, args.amp, args.omega, os.path.join(outdir, tag), Kp_c, Ki_c, Kd_c)
                J=objective_x(met, args.w_rmse, args.w_bias, args.w_phi)
                with open(os.path.join(outdir,"history.csv"),"a",newline="") as f:
                    csv.writer(f).writerow([f"{label}_{name}", tag, Kp_c, Ki_c, Kd_c, J, met["rmse"], met["mae"], met["bias"], ("" if met["phi"] is None else met["phi"])])
                if (best is None) or (J<best[0]-1e-12): best=(J,v)
            if best is None: break
            if best[1]!=val:
                val=best[1]; print(f"[{label}] {name} 改善: {val:.4f} (J={best[0]:.4f})"); break
            else:
                step*=args.refine_shrink; print(f"[{label}] {name} 改善なし → Δ縮小: {step:.4g}")
        return val, step, run_idx

    fixed={"Kp":Kp,"Ki":Ki,"Kd":Kd}
    sKp=max(base_dKp*args.refine_scale, args.refine_tol)
    sKi=max(base_dKi*args.refine_scale, args.refine_tol)
    sKd=max(base_dKd*args.refine_scale, args.refine_tol)

    # 順序: Kp → Kd → Ki
    fixed["Kp"], sKp, run_idx = one_axis("Kp", fixed["Kp"], sKp, fixed); Kp=fixed["Kp"]
    fixed["Kd"], sKd, run_idx = one_axis("Kd", fixed["Kd"], sKd, fixed); Kd=fixed["Kd"]
    fixed["Ki"], sKi, run_idx = one_axis("Ki", fixed["Ki"], sKi, fixed); Ki=fixed["Ki"]
    return Kp,Ki,Kd,run_idx

def tune_loop(args):
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("runs_x", session); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"history.csv"),"w",newline="") as f:
        csv.writer(f).writerow(["stage","run","Kp","Ki","Kd","J","rmse","mae","bias","phi"])

    Kp=args.kp; Ki=0.0; Kd=0.0
    dKp=max(args.dKp,args.tol); dKi=max(args.dKi,args.tol); dKd=max(args.dKd,args.tol)
    run_idx=0

    # === Stage Pass #1: Kp → Kd → Ki ===
    print("\n=== X: Stage Pass #1 (P→D→I) ===")
    Kp, J1,  run_idx = line_search(args,"S1","Kp",Kp, dKp, {"Kp":Kp,"Ki":0.0,"Kd":0.0}, outdir, run_idx)
    Kd, J1d, run_idx = line_search(args,"S1","Kd",0.0,dKd, {"Kp":Kp,"Ki":0.0,"Kd":0.0}, outdir, run_idx)
    Ki, J1i, run_idx = line_search(args,"S1","Ki",0.0,dKi, {"Kp":Kp,"Ki":0.0,"Kd":Kd}, outdir, run_idx)

    # === Stage Pass #2: Kp → Kd → Ki ===
    print("\n=== X: Stage Pass #2 (P→D→I) ===")
    Kp, J2,  run_idx = line_search(args,"S2","Kp",Kp, dKp, {"Kp":Kp,"Ki":Ki,"Kd":Kd}, outdir, run_idx)
    Kd, J2d, run_idx = line_search(args,"S2","Kd",Kd, dKd, {"Kp":Kp,"Ki":Ki,"Kd":Kd}, outdir, run_idx)
    Ki, J2i, run_idx = line_search(args,"S2","Ki",Ki, dKi, {"Kp":Kp,"Ki":Ki,"Kd":Kd}, outdir, run_idx)

    # === Refine: Kp → Kd → Ki ===
    base_dKp,base_dKi,base_dKd=dKp,dKi,dKd
    for p in range(1, args.refine_passes+1):
        label=f"RefineP{p}"
        print(f"\n=== X: Coordinate Descent Refinement {p}/{args.refine_passes} (P→D→I) ===")
        Kp,Ki,Kd,run_idx = refine_pass(args, Kp,Ki,Kd, outdir, run_idx, label, base_dKp,base_dKi,base_dKd)

    # 最終評価
    final_tag="final_eval_x"
    final_csv=os.path.join(outdir, f"{final_tag}.csv")
    ok = run_experiment(final_csv, args.amp, args.omega, args.duration, Kp, Ki, Kd, show_window=args.show)
    if ok:
        met=analyze_x(final_csv, args.amp, args.omega, os.path.join(outdir, final_tag), Kp, Ki, Kd)
        Jf=objective_x(met, args.w_rmse, args.w_bias, args.w_phi)
        print(f"\n[X 最終] Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}, J={Jf:.4f}")
        with open(os.path.join(outdir,"best.txt"),"w") as f:
            f.write(f"Kp_x={Kp:.6f}\nKi_x={Ki:.6f}\nKd_x={Kd:.6f}\nJ={Jf:.6f}\n")
    else:
        print("\n[X 最終] 実験失敗")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="X-axis PID autotuner (3-wheel omni, order P→D→I)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--amp", type=float, default=15.0)
    ap.add_argument("--omega", type=float, default=math.pi/2.0)
    ap.add_argument("--kp", type=float, default=5.0)
    ap.add_argument("--dKp", type=float, default=2.0)
    ap.add_argument("--dKi", type=float, default=0.5)
    ap.add_argument("--dKd", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--Kp_max", type=float, default=20.0)
    ap.add_argument("--Ki_max", type=float, default=5.0)
    ap.add_argument("--Kd_max", type=float, default=5.0)
    ap.add_argument("--w_rmse", type=float, default=1.0)
    ap.add_argument("--w_bias", type=float, default=0.2)
    ap.add_argument("--w_phi",  type=float, default=0.1)
    ap.add_argument("--refine_passes", type=int, default=2)
    ap.add_argument("--refine_scale", type=float, default=0.25)
    ap.add_argument("--refine_shrink", type=float, default=0.5)
    ap.add_argument("--refine_tol", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    args=ap.parse_args()
    if args.refine_tol is None: args.refine_tol=max(args.tol/5.0,1e-3)
    tune_loop(args)
