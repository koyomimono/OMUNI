#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_num(a):
    return pd.to_numeric(a, errors="coerce").to_numpy()


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_close(path, show):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def get_labels(df):
    if "run" in df.columns:
        return df["run"].astype(str).to_list()
    return [f"run{i:02d}" for i in range(len(df))]


def boxplot_one(y, title, ylabel, out_path, show):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return
    plt.figure()
    plt.boxplot(y, vert=True, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    save_close(out_path, show)


def hist_one(y, title, xlabel, out_path, show, bins=20, rng=None):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return
    plt.figure()
    plt.hist(y, bins=bins, range=rng)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    save_close(out_path, show)


def line_per_run(y, labels, title, ylabel, out_path, show, ylim=None):
    y = np.asarray(y, float)
    x = np.arange(len(y))
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return
    plt.figure()
    plt.plot(x[ok], y[ok], marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    save_close(out_path, show)


def scatter_xy(x, y, xlabel, ylabel, title, out_path, show):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    plt.figure()
    plt.scatter(x[ok], y[ok])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_close(out_path, show)


def corr_heatmap(df_num: pd.DataFrame, title, out_path, show):
    if df_num.shape[1] < 2:
        return
    C = df_num.corr(numeric_only=True)
    if C.shape[0] < 2:
        return
    plt.figure()
    plt.imshow(C.to_numpy(), origin="lower", aspect="auto")
    plt.xticks(np.arange(C.shape[1]), C.columns, rotation=45, ha="right")
    plt.yticks(np.arange(C.shape[0]), C.index)
    plt.title(title)
    # 값 텍스트 (너무 커지면 지저분해질 수 있어 최소한만)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C.iloc[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    save_close(out_path, show)


def zscore_profile(df, cols, labels, title, out_path, show):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    # 열 단위 z-score
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, skipna=True)
    sd = sd.replace(0, np.nan)
    Z = (X - mu) / sd
    Z = Z.to_numpy()

    if np.isfinite(Z).sum() < 10:
        return

    plt.figure()
    for i in range(Z.shape[0]):
        if np.isfinite(Z[i]).sum() >= 2:
            plt.plot(np.arange(len(cols)), Z[i], marker="o", linewidth=1, alpha=0.7)
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    plt.ylabel("z-score (per feature)")
    plt.title(title)
    save_close(out_path, show)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ALL_real_summary.csv 경로")
    ap.add_argument("--out", default="summary_plots", help="그래프 저장 폴더")
    ap.add_argument("--show", type=int, default=0, help="1이면 화면 표시")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    ensure_outdir(out_dir)

    df = pd.read_csv(in_path)
    labels = get_labels(df)

    # ---- 기본 핵심 컬럼들 ----
    has = set(df.columns)

    # 1) mean speed
    if "mean_speed" in has:
        y = to_num(df["mean_speed"])
        line_per_run(y, labels, "Mean speed per run", "Mean speed (mm/s)",
                     out_dir / "mean_speed_per_run.png", args.show)
        hist_one(y, "Mean speed distribution", "Mean speed (mm/s)",
                 out_dir / "mean_speed_hist.png", args.show, bins=25)
        boxplot_one(y, "Mean speed (boxplot)", "Mean speed (mm/s)",
                    out_dir / "mean_speed_boxplot.png", args.show)

    # 2) total distance
    if "total_distance_mm" in has:
        y = to_num(df["total_distance_mm"])
        line_per_run(y, labels, "Total distance per run", "Total distance (mm)",
                     out_dir / "total_distance_per_run.png", args.show)
        hist_one(y, "Total distance distribution", "Total distance (mm)",
                 out_dir / "total_distance_hist.png", args.show, bins=25)
        boxplot_one(y, "Total distance (boxplot)", "Total distance (mm)",
                    out_dir / "total_distance_boxplot.png", args.show)

        # outlier check (log10)
        y2 = y.copy()
        ok = np.isfinite(y2) & (y2 > 0)
        if ok.sum() >= 2:
            line_per_run(np.log10(y2), labels,
                         "Total distance log10 (outlier check)", "log10(distance mm)",
                         out_dir / "total_distance_log10_outlier_check.png", args.show)

    # 3) duration check
    if "duration_s" in has:
        y = to_num(df["duration_s"])
        line_per_run(y, labels, "Duration per run (sanity check)", "Duration (s)",
                     out_dir / "duration_per_run.png", args.show)
        hist_one(y, "Duration distribution", "Duration (s)",
                 out_dir / "duration_hist.png", args.show, bins=20)
        boxplot_one(y, "Duration (boxplot)", "Duration (s)",
                    out_dir / "duration_boxplot.png", args.show)

    # 4) stop ratio + bouts
    if "stop_ratio" in has:
        y = to_num(df["stop_ratio"])
        line_per_run(y, labels, "Stop ratio per run", "Stop ratio",
                     out_dir / "stop_ratio_per_run.png", args.show, ylim=(-0.05, 1.05))
        hist_one(y, "Stop ratio distribution", "Stop ratio",
                 out_dir / "stop_ratio_hist.png", args.show, bins=20, rng=(0, 1))
        boxplot_one(y, "Stop ratio (boxplot)", "Stop ratio",
                    out_dir / "stop_ratio_boxplot.png", args.show)

    for col in ["mean_stop_bout_s", "mean_move_bout_s"]:
        if col in has:
            y = to_num(df[col])
            line_per_run(y, labels, f"{col} per run", col,
                         out_dir / f"{col}_per_run.png", args.show)
            hist_one(y, f"{col} distribution", col,
                     out_dir / f"{col}_hist.png", args.show, bins=25)
            boxplot_one(y, f"{col} (boxplot)", col,
                        out_dir / f"{col}_boxplot.png", args.show)

    # 5) thigmotaxis (0.7/0.8/0.9R)
    edge_cols = [c for c in df.columns if c.startswith("edge_ratio_r_gt_")]
    if edge_cols:
        def edge_key(c):
            try:
                return float(c.split("_")[-1].replace("R", ""))
            except:
                return 999.0
        edge_cols = sorted(edge_cols, key=edge_key)

        # per-run line (multi)
        plt.figure()
        x = np.arange(len(df))
        for c in edge_cols:
            y = to_num(df[c])
            ok = np.isfinite(y)
            if ok.sum() >= 2:
                plt.plot(x[ok], y[ok], marker="o", label=c)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylim(-0.05, 1.05)
        plt.ylabel("Edge preference ratio")
        plt.title("Thigmotaxis indices per run")
        plt.legend()
        save_close(out_dir / "thigmotaxis_per_run.png", args.show)

        # distribution hist + box for each
        for c in edge_cols:
            y = to_num(df[c])
            hist_one(y, f"{c} distribution", c,
                     out_dir / f"{c}_hist.png", args.show, bins=20, rng=(0, 1))
            boxplot_one(y, f"{c} (boxplot)", c,
                        out_dir / f"{c}_boxplot.png", args.show)

    # 6) occupancy entropy
    if "occupancy_entropy" in has:
        y = to_num(df["occupancy_entropy"])
        line_per_run(y, labels, "Occupancy entropy per run", "Occupancy entropy",
                     out_dir / "occupancy_entropy_per_run.png", args.show)
        hist_one(y, "Occupancy entropy distribution", "Occupancy entropy",
                 out_dir / "occupancy_entropy_hist.png", args.show, bins=25)
        boxplot_one(y, "Occupancy entropy (boxplot)", "Occupancy entropy",
                    out_dir / "occupancy_entropy_boxplot.png", args.show)

    # 7) 관계도(필수 추천)
    # speed vs edge0.8R, speed vs stop_ratio, distance vs stop_ratio
    if "mean_speed" in has and "stop_ratio" in has:
        scatter_xy(to_num(df["mean_speed"]), to_num(df["stop_ratio"]),
                   "Mean speed (mm/s)", "Stop ratio",
                   "Mean speed vs stop ratio",
                   out_dir / "mean_speed_vs_stop_ratio.png", args.show)

    edge08 = "edge_ratio_r_gt_0.8R"
    if "mean_speed" in has and edge08 in has:
        scatter_xy(to_num(df["mean_speed"]), to_num(df[edge08]),
                   "Mean speed (mm/s)", "Edge ratio (r > 0.8R)",
                   "Mean speed vs thigmotaxis (0.8R)",
                   out_dir / "mean_speed_vs_edge0p8R.png", args.show)

    if "total_distance_mm" in has and "stop_ratio" in has:
        scatter_xy(to_num(df["total_distance_mm"]), to_num(df["stop_ratio"]),
                   "Total distance (mm)", "Stop ratio",
                   "Total distance vs stop ratio",
                   out_dir / "total_distance_vs_stop_ratio.png", args.show)

    if "mean_speed" in has and "total_distance_mm" in has:
        scatter_xy(to_num(df["mean_speed"]), to_num(df["total_distance_mm"]),
                   "Mean speed (mm/s)", "Total distance (mm)",
                   "Mean speed vs total distance",
                   out_dir / "mean_speed_vs_total_distance.png", args.show)

    # 8) feature correlation heatmap (필수)
    corr_cols = []
    for c in [
        "duration_s", "mean_speed", "median_speed", "p90_speed", "max_speed",
        "total_distance_mm", "stop_ratio", "mean_stop_bout_s", "mean_move_bout_s",
        "occupancy_entropy", edge08
    ]:
        if c in has:
            corr_cols.append(c)

    # edge 0.7/0.9도 있으면 포함
    for c in ["edge_ratio_r_gt_0.7R", "edge_ratio_r_gt_0.9R"]:
        if c in has and c not in corr_cols:
            corr_cols.append(c)

    if len(corr_cols) >= 3:
        df_num = df[corr_cols].apply(pd.to_numeric, errors="coerce")
        corr_heatmap(df_num, "Feature correlation heatmap", out_dir / "feature_correlation_heatmap.png", args.show)

    # 9) z-score feature profile (이상치 런 탐지용, 매우 유용)
    # 너무 많으면 지저분해지니 핵심만
    prof_cols = []
    for c in ["mean_speed", "total_distance_mm", "stop_ratio", "occupancy_entropy", edge08]:
        if c in has:
            prof_cols.append(c)
    if len(prof_cols) >= 3:
        zscore_profile(df, prof_cols, labels,
                       "Run-wise z-score feature profiles (outlier-friendly)",
                       out_dir / "zscore_feature_profiles.png", args.show)

    print(f"[OK] saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
