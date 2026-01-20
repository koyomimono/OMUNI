#!/usr/bin/env python3
# step.py
# ex（赤），ey（青），z＝psi/epsi/omega（緑）をplot表示．基準線はy=0（黒）．
# 時刻は常にt[0]を原点にして0秒スタートし，x軸の左端を0に固定する．
# 使い方：
#   表示のみ：python step.py --csv step_response.csv
#   表示＋保存：python step.py --csv step_response.csv --save out

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path: str):
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

def make_plot(t, y, title, ylabel, save_path=None, color=None, zero_color=None):
    plt.figure()
    plt.plot(t, y, label=title, color=color)
    plt.axhline(0.0, color=zero_color)  # 基準線（y=0）
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # ← ここがポイント：左端を0秒に固定
    plt.xlim(left=0.0, right=max(t) if len(t) else None)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

def main():
    ap = argparse.ArgumentParser(description="x，y，zのステップ応答（基準線0，t=0原点）をplot表示")
    ap.add_argument("--csv", default="step_response.csv", help="入力CSVパス")
    ap.add_argument("--save", default=None, help="保存ディレクトリ（指定しなければ保存しない）")
    # 既定色：x=赤，y=青，z=緑，基準線=黒（実行時に上書き可）
    ap.add_argument("--color-x", default="r")
    ap.add_argument("--color-y", default="b")
    ap.add_argument("--color-z", default="g")
    ap.add_argument("--zero-color", default="k")
    args = ap.parse_args()

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    df, z_col = load_csv(args.csv)

    # 時刻の原点合わせ（必ず0秒スタート）
    t_raw = df["time[s]"].to_numpy(dtype=float)
    t0 = t_raw[0] if len(t_raw) else 0.0
    t = np.maximum(t_raw - t0, 0.0)  # 数値誤差の負をクリップ

    x = df["ex[mm]"].to_numpy(dtype=float)
    y = df["ey[mm]"].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    sp = (lambda name: os.path.join(args.save, name) if args.save else None)
    #make_plot(t, x, "Step Response x", "ex [mm]",save_path=sp("step_x.png"), color=args.color_x, zero_color=args.zero_color)
    make_plot(t, y, "Step Response y", "ey [mm]",save_path=sp("step_y.png"), color=args.color_y, zero_color=args.zero_color)
    #make_plot(t, z, f"Step Response z ({z_col})", z_col,save_path=sp("step_z.png"), color=args.color_z, zero_color=args.zero_color)

    plt.show()

if __name__ == "__main__":
    main()
