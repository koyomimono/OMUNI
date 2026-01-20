#!/usr/bin/env python3
# step.py
# ex（赤），ey（青），z＝psi/epsi/omega（緑）をplot表示し，評価数値と読みやすい要約を出力する．
# グラフはピーク点・最終値・±5%整定帯を注釈表示．基準線はy=0（黒），時刻はt[0]を原点として0秒スタート．
#
# 使い方：
#   表示のみ：python step.py --csv step_response.csv
#   表示＋保存（画像・CSV・Markdownレポート）：python step.py --csv step_response.csv --save out
#   プロット無しで数値だけ：python step.py --csv step_response.csv --no-plot
#
# 出力：
#   - 端末：各信号ごとの一行要約＋主要指標
#   - out/metrics.csv：指標の表
#   - out/report.md：日本語の読みやすい要約レポート
#   - out/step_x.png／step_y.png／step_z.png：注釈入りプロット

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 入出力と基本描画 ======

def load_csv(path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for r in ["time[s]", "ex[mm]", "ey[mm]"]:
        if r not in df.columns:
            raise ValueError(f"CSVに必要な列 {r} が見つかりません：{path}")
    # z列は優先順で自動選択
    for cand in ["psi[rad]", "epsi[rad]", "omega[rad/s]"]:
        if cand in df.columns:
            return df, cand
    raise ValueError("zに相当する列 psi[rad]／epsi[rad]／omega[rad/s] が見つかりません．")

def make_plot_with_annotations(
    t, y, title, ylabel, save_path=None, color=None, zero_color="k",
    y_final=None, t_peak=None, y_peak=None, band=None, Ts5=None, t10=None, t90=None
):
    plt.figure()
    plt.plot(t, y, label=title, color=color, linewidth=1.6)
    # 基準線（y=0）
    plt.axhline(0.0, color=zero_color, linewidth=1.0)
    # 最終値
    if y_final is not None:
        plt.axhline(y_final, color=color, linestyle="--", linewidth=1.0, alpha=0.7, label="final")
    # 5%整定帯
    if (band is not None) and (band > 0.0):
        plt.fill_between(t, +band, -band, color="gray", alpha=0.12, label="±5% band")
    # 立上り10→90%
    if (t10 is not None) and (t90 is not None) and (t90 >= t10):
        plt.axvline(t10, color="gray", linestyle=":", linewidth=1.0)
        plt.axvline(t90, color="gray", linestyle=":", linewidth=1.0)
    # Ts5
    if Ts5 is not None:
        plt.axvline(Ts5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Ts5")
    # ピーク
    if (t_peak is not None) and (y_peak is not None):
        plt.scatter([t_peak], [y_peak], color=color, s=28, zorder=3)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.xlim(left=0.0, right=max(t) if len(t) else None)
    # y軸は見やすいように少し余白
    if len(y):
        ymin, ymax = float(np.min(y)), float(np.max(y))
        pad = 0.05 * max(1e-9, ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

# ====== メトリクス計算 ======

def _window_mask(t: np.ndarray, head_sec: float, tail_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(t) == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    t0, tN = t[0], t[-1]
    head_mask = t <= (t0 + head_sec)
    tail_mask = t >= (tN - tail_sec)
    if not head_mask.any():
        head_mask[: max(1, int(0.05 * len(t)))] = True
    if not tail_mask.any():
        tail_mask[-max(1, int(0.10 * len(t))):] = True
    return head_mask, tail_mask

def _first_crossing_time(t: np.ndarray, y: np.ndarray, level: float, rising: bool = True) -> Optional[float]:
    if len(t) < 2:
        return None
    if rising:
        cond = y[:-1] < level
        cond2 = y[1:] >= level
    else:
        cond = y[:-1] > level
        cond2 = y[1:] <= level
    idx = np.where(cond & cond2)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(t[i + 1])
    ratio = (level - y0) / (y1 - y0)
    return float(t[i] + ratio * (t[i + 1] - t[i]))

def compute_metrics(t: np.ndarray, y: np.ndarray, target: float = 0.0) -> Dict[str, Optional[float]]:
    eps = 1e-12
    if len(t) == 0 or len(y) == 0:
        keys = ["n","Tend","y_init","y_final","ess","y_peak","t_peak",
                "OS_pct","Ts2","Ts5","Tr10_90","t10","t90","IAE","ISE","ITAE","RMS","band5"]
        return {k: None for k in keys}
    Tspan = max(0.0, float(t[-1] - t[0]))
    head_sec = max(0.2, 0.05 * Tspan)
    tail_sec = max(0.2, 0.10 * Tspan)
    head_mask, tail_mask = _window_mask(t, head_sec, tail_sec)

    y_init = float(np.median(y[head_mask]))
    y_final = float(np.median(y[tail_mask]))
    ess = y_final - target

    A_step = abs(y_final - y_init)
    A_peak = float(np.max(np.abs(y - target)))
    A = A_step if A_step > 1e-6 else A_peak
    if A < eps:
        A = eps

    if (y_final - y_init) >= 0:
        i_peak = int(np.argmax(y))
    else:
        i_peak = int(np.argmin(y))
    y_peak = float(y[i_peak])
    t_peak = float(t[i_peak])

    if (y_final - y_init) >= 0:
        OS = max(0.0, (y_peak - y_final) / A * 100.0)
    else:
        OS = max(0.0, (y_final - y_peak) / A * 100.0)

    def settling_time(rel_band: float) -> Optional[float]:
        band = rel_band * A
        err = np.abs(y - target)
        inside = err <= band
        last_violate = np.where(~inside)[0]
        if len(last_violate) == 0:
            return float(t[0])
        k = last_violate[-1]
        if k == len(t) - 1:
            return None
        return float(t[k + 1])

    Ts2 = settling_time(0.02)
    Ts5 = settling_time(0.05)
    band5 = 0.05 * A  # プロット用の±5%帯

    sign = 1.0 if (y_final - y_init) >= 0 else -1.0
    y_adj = sign * (y - y_init)
    A_eff = abs(y_final - y_init)
    Tr = None
    t10 = None
    t90 = None
    if A_eff > 1e-9:
        t10 = _first_crossing_time(t, y_adj, 0.1 * A_eff, rising=True)
        t90 = _first_crossing_time(t, y_adj, 0.9 * A_eff, rising=True)
        if (t10 is not None) and (t90 is not None) and (t90 >= t10):
            Tr = float(t90 - t10)

    err = y - target
    IAE = float(np.trapz(np.abs(err), t))
    ISE = float(np.trapz(err * err, t))
    ITAE = float(np.trapz(np.abs(err) * (t - t[0]), t))
    RMS = float(np.sqrt(ISE / max(Tspan, eps)))

    return {
        "n": int(len(t)),
        "Tend": float(Tspan),
        "y_init": y_init,
        "y_final": y_final,
        "ess": ess,
        "y_peak": y_peak,
        "t_peak": t_peak,
        "OS_pct": OS,
        "Ts2": Ts2,
        "Ts5": Ts5,
        "Tr10_90": Tr,
        "t10": t10,
        "t90": t90,
        "IAE": IAE,
        "ISE": ISE,
        "ITAE": ITAE,
        "RMS": RMS,
        "band5": band5,
    }

# ====== 表示とレポート ======

def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        row = {"signal": name}
        row.update(m)
        rows.append(row)
    cols = ["signal","n","Tend","y_init","y_final","ess","y_peak","t_peak",
            "OS_pct","Ts2","Ts5","Tr10_90","IAE","ISE","ITAE","RMS","t10","t90","band5"]
    return pd.DataFrame(rows)[cols]

def _u(name: str) -> str:
    if name.startswith("ex") or name.startswith("ey"):
        return "mm"
    if "rad/s" in name:
        return "rad/s"
    return "rad"

def one_line_summary(sig: str, m: Dict[str, Optional[float]]) -> str:
    # 主要4点だけを短く
    ess = m["ess"]; Ts5 = m["Ts5"]; Tr = m["Tr10_90"]; OS = m["OS_pct"]
    u = _u(sig)
    ess_s = "NA" if ess is None else f"{ess:.3g} {u}"
    Ts5_s = "NA" if Ts5 is None else f"{Ts5:.3g} s"
    Tr_s = "NA" if Tr is None else f"{Tr:.3g} s"
    OS_s = "NA" if OS is None else f"{OS:.3g} %"
    status = "整定済" if Ts5 is not None else "未整定"
    return f"[{sig}] {status}｜ess={ess_s}｜Ts5={Ts5_s}｜Tr10-90={Tr_s}｜OS={OS_s}"

def md_block_for_signal(sig: str, m: Dict[str, Optional[float]]) -> str:
    u = _u(sig)
    def fmt(v, suf=""):
        if v is None: return "NA"
        return f"{v:.5g}{suf}"
    lines = []
    lines.append(f"### {sig}")
    lines.append(f"- **定常偏差** ess：{fmt(m['ess'], ' ' + u)}")
    lines.append(f"- **整定時間** Ts5：{fmt(m['Ts5'], ' s')}（2%帯 Ts2：{fmt(m['Ts2'], ' s')}）")
    lines.append(f"- **立上り時間** Tr10–90：{fmt(m['Tr10_90'], ' s')}（t10={fmt(m['t10'], ' s')}，t90={fmt(m['t90'], ' s')}）")
    lines.append(f"- **オーバーシュート** OS：{fmt(m['OS_pct'], ' %')}，**ピーク** y_peak={fmt(m['y_peak'], ' ' + u)}@{fmt(m['t_peak'], ' s')}")
    lines.append(f"- **RMS**：{fmt(m['RMS'], ' ' + u)}，**IAE**：{fmt(m['IAE'])}，**ISE**：{fmt(m['ISE'])}，**ITAE**：{fmt(m['ITAE'])}")
    return "\n".join(lines)

def save_markdown_report(path_dir: str, metrics: Dict[str, Dict[str, float]]):
    md = ["# ステップ応答レポート（要約）", ""]
    for sig, m in metrics.items():
        md.append(md_block_for_signal(sig, m))
        md.append("")
    out_md = os.path.join(path_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"レポートを保存しました：{out_md}")

def print_compact_dashboard(metrics: Dict[str, Dict[str, float]]):
    print("\n== 要約（最重要指標だけを簡潔に） ==")
    for sig in metrics.keys():
        print(one_line_summary(sig, metrics[sig]))

# ====== メイン ======

def main():
    ap = argparse.ArgumentParser(description="x，y，zのステップ応答（基準線0，t=0原点）をplot表示し評価数値を出力")
    ap.add_argument("--csv", default="step_response.csv", help="入力CSVパス")
    ap.add_argument("--save", default=None, help="保存ディレクトリ（指定しなければ保存しない）")
    ap.add_argument("--no-plot", dest="no_plot", action="store_true", help="プロットを表示しない")
    # 既定色：x=赤，y=青，z=緑，基準線=黒（実行時に上書き可）
    ap.add_argument("--color-x", default="r")
    ap.add_argument("--color-y", default="b")
    ap.add_argument("--color-z", default="g")
    ap.add_argument("--zero-color", default="k")
    args = ap.parse_args()

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    df, z_col = load_csv(args.csv)

    # 時刻の原点合わせ（0秒スタート）
    t_raw = df["time[s]"].to_numpy(dtype=float)
    t0 = t_raw[0] if len(t_raw) else 0.0
    t = np.maximum(t_raw - t0, 0.0)

    x = df["ex[mm]"].to_numpy(dtype=float)
    y = df["ey[mm]"].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    # 評価数値
    m_x = compute_metrics(t, x, target=0.0)
    m_y = compute_metrics(t, y, target=0.0)
    m_z = compute_metrics(t, z, target=0.0)
    metrics = {"ex[mm]": m_x, "ey[mm]": m_y, f"{z_col}": m_z}

    # コンパクト表示
    print_compact_dashboard(metrics)

    # CSV保存
    mdf = metrics_to_dataframe(metrics)
    if args.save:
        out_csv = os.path.join(args.save, "metrics.csv")
        mdf.to_csv(out_csv, index=False)
        print(f"評価数値を保存しました：{out_csv}")
        save_markdown_report(args.save, metrics)

    # プロット
    if not args.no_plot:
        sp = (lambda name: os.path.join(args.save, name) if args.save else None)
        # x
        make_plot_with_annotations(
            t, x, "Step Response x", "ex [mm]",
            save_path=sp("step_x.png"), color=args.color_x, zero_color=args.zero_color,
            y_final=m_x["y_final"], t_peak=m_x["t_peak"], y_peak=m_x["y_peak"],
            band=m_x["band5"], Ts5=m_x["Ts5"], t10=m_x["t10"], t90=m_x["t90"]
        )
        # y
        make_plot_with_annotations(
            t, y, "Step Response y", "ey [mm]",
            save_path=sp("step_y.png"), color=args.color_y, zero_color=args.zero_color,
            y_final=m_y["y_final"], t_peak=m_y["t_peak"], y_peak=m_y["y_peak"],
            band=m_y["band5"], Ts5=m_y["Ts5"], t10=m_y["t10"], t90=m_y["t90"]
        )
        # z
        make_plot_with_annotations(
            t, z, f"Step Response z ({z_col})", z_col,
            save_path=sp("step_z.png"), color=args.color_z, zero_color=args.zero_color,
            y_final=m_z["y_final"], t_peak=m_z["t_peak"], y_peak=m_z["y_peak"],
            band=m_z["band5"], Ts5=m_z["Ts5"], t10=m_z["t10"], t90=m_z["t90"]
        )
        plt.show()

if __name__ == "__main__":
    main()
