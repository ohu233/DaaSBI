#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七天流量趋势折线图

直接画每天的流量指标（总量/峰值），按 ≥50 与 <50 两条线，
突出日间变化——这是看"流量变化"最干净的方式。

依赖: pip install pandas numpy matplotlib
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_freq_heatmap import BASE_DIR, INPUT_DIR, OUTPUT_DIR, WEEKDAY_CN

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DAYS = ["20230917", "20230918", "20230919", "20230920",
        "20230921", "20230922", "20230923"]
CATEGORIES = ["gte50", "lt50"]
LABEL = {"gte50": "≥50", "lt50": "<50"}
COLOR = {"gte50": "#d62728", "lt50": "#1f77b4"}

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "flow_trend_7days.png")


def main():
    print("读取数据...", flush=True)
    totals, peaks, medians = {}, {}, {}
    for cat in CATEGORIES:
        for day in DAYS:
            df = pd.read_csv(
                os.path.join(INPUT_DIR, f"signal_cell_counts_{day}_strict_od_{cat}.csv"),
                encoding="utf-8-sig",
            )
            totals[(cat, day)] = float(df["pass_count"].sum())
            peaks[(cat, day)] = float(df["pass_count"].max())
            medians[(cat, day)] = float(df["pass_count"].median())
            print(f"  {day} {cat}: 总量={totals[(cat,day)]:,.0f}  峰值={peaks[(cat,day)]:,.0f}", flush=True)

    x = np.arange(len(DAYS))
    xlabels = [
        f"{d[4:6]}/{d[6:8]}\n{WEEKDAY_CN[datetime.strptime(d, '%Y%m%d').weekday()]}"
        for d in DAYS
    ]
    weekend_idx = [i for i, d in enumerate(DAYS)
                   if datetime.strptime(d, "%Y%m%d").weekday() >= 5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("七天流量趋势（红：≥50；蓝：<50）", fontsize=15)

    panels = [
        (axes[0], totals, "每日通行总量（所有栅格 pass_count 之和）", "总量"),
        (axes[1], peaks, "每日峰值（最忙栅格 pass_count）", "峰值"),
        (axes[2], medians, "每日中位数 pass_count", "中位数"),
    ]
    for ax, data, title, yname in panels:
        for i in weekend_idx:  # 周末底色
            ax.axvspan(i - 0.4, i + 0.4, color="#fff2cc", zorder=0)
        for cat in CATEGORIES:
            ys = [data[(cat, d)] for d in DAYS]
            ax.plot(x, ys, "-o", color=COLOR[cat], label=LABEL[cat],
                    linewidth=2, markersize=6)
            # 数据标注
            for xi, yi in zip(x, ys):
                ax.annotate(f"{yi:,.0f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=COLOR[cat])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(yname)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\n趋势图已保存: {OUTPUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
