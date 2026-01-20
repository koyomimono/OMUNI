import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_FILE = "frequency_response_raw_20251223_160821.csv"  # <- change
AXIS = "z"  # "x","y","z"

# ---------- Robust filters ----------
def hampel_filter(x, k=5, t0=3.0):
    """
    Hampel filter (median + MAD). Replaces outliers with local median.
    k : half window size
    t0: threshold (3~4 typical)
    """
    x = np.asarray(x, dtype=float)
    y = x.copy()
    n = len(x)
    for i in range(n):
        i0 = max(0, i - k)
        i1 = min(n, i + k + 1)
        w = x[i0:i1]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-12
        sigma = 1.4826 * mad
        if np.abs(x[i] - med) > t0 * sigma:
            y[i] = med
    return y

def smooth_savgol(y, window=11, poly=2):
    """
    Simple Savitzky-Golay smoothing without scipy.
    (small window + poly is fine for your 100 pts)
    """
    y = np.asarray(y, dtype=float)
    if window % 2 == 0:
        window += 1
    if window < poly + 2:
        window = poly + 2
        if window % 2 == 0:
            window += 1
    half = window // 2

    # precompute design matrix for centered positions
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, N=poly + 1, increasing=True)
    # smoothing corresponds to evaluating at 0 => first row of pseudoinverse * y_window
    pinv = np.linalg.pinv(A)
    coeff = pinv[0]  # weights for y_window -> y_smoothed at center

    ypad = np.pad(y, (half, half), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        w = ypad[i:i + window]
        out[i] = np.dot(coeff, w)
    return out

def wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0

# ---------- Load CSV and build FRF per freq ----------
data = defaultdict(lambda: {"u": [], "y": []})

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        freq = float(row["freq_hz"])
        u = float(row["input"])
        if AXIS == "x":
            y = float(row["mouse_x"])
        elif AXIS == "y":
            y = float(row["mouse_y"])
        else:
            y = float(row["mouse_z"])

        data[freq]["u"].append(u)
        data[freq]["y"].append(y)

freqs = []
mag_db = []
ph_deg = []

for freq, d in sorted(data.items()):
    u = np.asarray(d["u"], dtype=float)
    y = np.asarray(d["y"], dtype=float)

    # FFT
    U = np.fft.fft(u)
    Y = np.fft.fft(y)

    # pick dominant input bin (excluding DC)
    k = np.argmax(np.abs(U[1:])) + 1
    G = Y[k] / (U[k] + 1e-12)

    freqs.append(freq)
    mag_db.append(20.0 * np.log10(np.abs(G) + 1e-12))
    ph_deg.append(np.degrees(np.angle(G)))

freqs = np.asarray(freqs)
mag_db = np.asarray(mag_db)
ph_deg = np.asarray(ph_deg)

# ---------- Phase unwrap then filtering ----------
ph_unwrap = np.degrees(np.unwrap(np.radians(ph_deg)))

# 1) remove spikes (Hampel)
mag_h = hampel_filter(mag_db, k=4, t0=3.0)
ph_h  = hampel_filter(ph_unwrap, k=4, t0=3.0)

# 2) smooth (Savitzky-Golay-like)
mag_f = smooth_savgol(mag_h, window=11, poly=2)
ph_f  = smooth_savgol(ph_h,  window=11, poly=2)

# wrap phase back to [-180, 180]
ph_f_wrapped = wrap_deg(ph_f)

# ---------- Plot: scatter + line ----------
plt.figure(figsize=(9, 6))

plt.subplot(2, 1, 1)
plt.semilogx(freqs, mag_db, "o", alpha=0.5, label="raw (points)")
plt.semilogx(freqs, mag_f,  "-", linewidth=2, label="filtered (line)")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogx(freqs, wrap_deg(ph_deg), "o", alpha=0.5, label="raw (points)")
plt.semilogx(freqs, ph_f_wrapped,     "-", linewidth=2, label="filtered (line)")
plt.ylabel("Phase [deg]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f"bode_{AXIS}_filtered.png", dpi=200)
plt.show()

