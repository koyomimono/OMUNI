# --- 必要なライブラリ ---
# pip install pandas matplotlib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 入力CSVと制御パラメータ（実験時と合わせる）=====
CSV_FILE = "step_response_xyz.csv"

AMP_MM = 15.0               # x_ref 振幅[mm]
OMEGA  = math.pi / 2.0      # x_ref 角速度[rad/s]（周期=4 s）
X_BIAS = 0.0

# 図に載せるPID（ログ時の値に合わせて書き換え）
Kp_xy = 5.0
Ki_xy = 1.0
Kd_xy = 0.2
Kp_psi = 0.0
Ki_psi = 0.0
Kd_psi = 0.0

# ===== CSV読み込み =====
df = pd.read_csv(CSV_FILE)

# 時間[s]とx誤差[mm]（ex = x_actual - x_ref, 数学座標系）
t  = df['time[s]'].to_numpy(dtype=float)
ex = df['ex[mm]'].to_numpy(dtype=float)

# ===== 参照と実測の復元（数学座標系 右+上+）=====
x_ref = X_BIAS + AMP_MM * np.sin(OMEGA * t)
x_act = x_ref + ex

# ===== メトリクス: 平均誤差と時間遅れ =====
# 標本周期（可変FPS対策として中央値）
dt_med = float(np.median(np.diff(t)))

bias_x = float(np.mean(ex))                         # 符号付き平均誤差
mae_x  = float(np.mean(np.abs(ex)))                 # 平均絶対誤差
rmse_x = float(np.sqrt(np.mean(ex**2)))             # 二乗平均平方根誤差

# 時間遅れ: 相互相関のピークから推定（正の遅れ=実測が遅れる）
xr = x_ref - np.mean(x_ref)
xa = x_act - np.mean(x_act)
if np.std(xr) > 1e-9 and np.std(xa) > 1e-9:
    corr = np.correlate(xr, xa, mode='full')
    lags = np.arange(-len(xr)+1, len(xr))
    kmax = int(np.argmax(corr))
    lag  = int(lags[kmax])
    tau  = lag * dt_med                              # [s]
    phi  = (tau * OMEGA) * 180.0 / math.pi           # [deg]
    tau_str = f"{tau*1000:.0f} ms (φ≈{phi:.1f}°)"
else:
    tau_str = "n/a"

# ===== 図：xのみ =====
plt.figure(figsize=(9, 4.5), dpi=120)
plt.plot(t, x_act, label="x_actual (mm)")                 # 実線
plt.plot(t, x_ref, linestyle="--", label="x_ref (mm)")    # 点線
plt.xlabel("time (s)")
plt.ylabel("x (mm)")
plt.title("X position: actual (solid) vs reference (dashed)")
plt.grid(True, which="both", linestyle=":")
plt.legend()

# 図中にPID値とメトリクス表示
txt = (
    f"P_xy={Kp_xy}, I_xy={Ki_xy}, D_xy={Kd_xy}\n"
    f"P_psi={Kp_psi}, I_psi={Ki_psi}, D_psi={Kd_psi}\n"
    f"bias={bias_x:.3f} mm, MAE={mae_x:.3f} mm, RMSE={rmse_x:.3f} mm\n"
    f"delay≈{tau_str}"
)
ax = plt.gca()
ax.text(0.01, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", alpha=0.15))

plt.tight_layout()
plt.savefig("plot_x_with_xyz.png")
plt.show()
