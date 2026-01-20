# --- 必要なライブラリ ---
# pip install pandas matplotlib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 入力CSVと制御パラメータ（実験時と合わせる）=====
CSV_FILE = "step_response_xyz.csv"

# y_ref は実験仕様どおり 0 を既定にする
Y_BIAS = 0.0
USE_COS_REF = True         # y も周期参照を使った実験なら True にし、下の OMEGA/AMP_Y を設定
OMEGA  = math.pi / 2.0       # [rad/s]（x と同じ周波数を想定）
AMP_Y  = 15.0                # [mm]    （必要なら実験値に合わせて）

# 図に載せるPID（ログ時の値に合わせて書き換え）
Kp_xy = 5.0
Ki_xy = 1.0
Kd_xy = 0.2
Kp_psi = 0.0
Ki_psi = 0.0
Kd_psi = 0.0

# ===== CSV読み込み =====
df = pd.read_csv(CSV_FILE)

# 時間[s]とy誤差[mm]（ey = y_actual - y_ref, 数学座標系）
t  = df['time[s]'].to_numpy(dtype=float)
ey = df['ey[mm]'].to_numpy(dtype=float)

# ===== 参照と実測の復元（数学座標系 右+上+）=====
if USE_COS_REF:
    y_ref = Y_BIAS + AMP_Y * np.cos(OMEGA * t)
else:
    y_ref = Y_BIAS + 0.0 * t  # 既定は定数0

y_act = y_ref + ey

# ===== メトリクス: 平均誤差と（可能なら）時間遅れ =====
bias_y = float(np.mean(ey))                         # 符号付き平均誤差
mae_y  = float(np.mean(np.abs(ey)))                 # 平均絶対誤差
rmse_y = float(np.sqrt(np.mean(ey**2)))             # 二乗平均平方根誤差

yr = y_ref - np.mean(y_ref)
ya = y_act - np.mean(y_act)
if np.std(yr) > 1e-9 and np.std(ya) > 1e-9:
    # 参照が周期なら相互相関で遅れ推定（正の遅れ=実測が遅い）
    dt_med = float(np.median(np.diff(t)))
    corr = np.correlate(yr, ya, mode='full')
    lags = np.arange(-len(yr)+1, len(yr))
    kmax = int(np.argmax(corr))
    lag  = int(lags[kmax])
    tau  = lag * dt_med                              # [s]
    phi  = (tau * OMEGA) * 180.0 / math.pi           # [deg]
    tau_str = f"{tau*1000:.0f} ms (φ≈{phi:.1f}°)"
else:
    tau_str = "n/a (const ref)"

# ===== 図：yのみ =====
plt.figure(figsize=(9, 4.5), dpi=120)
plt.plot(t, y_act, label="y_actual (mm)")                 # 実線
plt.plot(t, y_ref, linestyle="--", label="y_ref (mm)")    # 点線
plt.xlabel("time (s)")
plt.ylabel("y (mm)")
plt.title("Y position: actual (solid) vs reference (dashed)")
plt.grid(True, which="both", linestyle=":")
plt.legend()

# 図中にPID値とメトリクス表示
txt = (
    f"P_xy={Kp_xy}, I_xy={Ki_xy}, D_xy={Kd_xy}\n"
    f"P_psi={Kp_psi}, I_psi={Ki_psi}, D_psi={Kd_psi}\n"
    f"bias={bias_y:.3f} mm, MAE={mae_y:.3f} mm, RMSE={rmse_y:.3f} mm\n"
    f"delay≈{tau_str}"
)
ax = plt.gca()
ax.text(0.01, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", alpha=0.15))

plt.tight_layout()
plt.savefig("plot_y_with_xyz.png")
plt.show()
