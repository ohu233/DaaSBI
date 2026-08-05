#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七天 × (≥50 / <50) 频次热力图拼图（带高德底图）

生成一张 PNG，2 行 7 列：
  上行 = strict_od_gte50（通行频次 ≥ 50）
  下行 = strict_od_lt50（通行频次 < 50）
  列   = 2023-09-17 ~ 2023-09-23 七天（横向排列）

所有子图共用同一空间范围（框选范围一致）、同一张高德路网底图。
栅格原始为 WGS84，先转 GCJ-02 与高德瓦片对齐，再转 Web Mercator 绘制。

依赖: pip install pandas numpy matplotlib scipy contextily
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接出 PNG
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import contextily as cx

# 中文字体（Windows 自带微软雅黑/黑体），避免中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from plot_freq_heatmap import (
    BASE_DIR, PKL_PATH, INPUT_DIR, OUTPUT_DIR, WEEKDAY_CN,
)

# ============================================================
# 参数
# ============================================================
TOP_PCT = 0.05      # 每张子图只画频次最高的 top 5% 栅格（与单图一致）
WEIGHT_MODE = "minmax"  # 权重/归一方式：minmax(最大最小线性) | zscore(均值方差) | log(对数)
WEIGHT_LABEL = {"minmax": "最大最小线性", "zscore": "均值方差(z-score)", "log": "对数"}
GRID_PIX = 500      # 栅格化分辨率（每边像素数）
SIGMA = 4.0         # 高斯平滑核（像素），模拟扩散效果
PAD_DEG = 0.01      # 框选范围外扩留白（度）
ZOOM = 11           # 高德底图瓦片层级（不动）
DPI = 300           # 输出分辨率
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "freq_heatmap_7days_gte50_lt50.png")

# 高德路网瓦片（style=8）。固定子域 01，避开 contextily 不展开 {s} 的问题
GAODE_URL = (
    "http://webrd01.is.autonavi.com/appmaptile"
    "?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
)

# 渐变：0~BLUE_FLOOR 压成纯蓝（低值统一蓝色），之上走 blue→cyan→lime→yellow→red
BLUE_FLOOR = 0.3
_span = 1.0 - BLUE_FLOOR
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "freq",
    [
        (0.0, "blue"), (BLUE_FLOOR, "blue"),
        (BLUE_FLOOR + _span * 0.25, "cyan"),
        (BLUE_FLOOR + _span * 0.50, "lime"),
        (BLUE_FLOOR + _span * 0.75, "yellow"),
        (1.0, "red"),
    ],
)
HEAT_CMAP.set_bad(alpha=0)  # 空值透明，便于叠在底图上

CATEGORIES = ["gte50", "lt50"]
CATEGORY_LABEL = {"gte50": "通行频次 ≥ 50", "lt50": "通行频次 < 50"}
# 显示的行：上=≥50，中=<50，下=两者汇总（不区分）
ROWS = ["gte50", "lt50", "total"]
ROW_LABEL = {"gte50": "通行频次 ≥ 50", "lt50": "通行频次 < 50", "total": "汇总（全部）"}
DAYS = ["20230917", "20230918", "20230919", "20230920",
        "20230921", "20230922", "20230923"]

_EE = 0.00669342162296594323
_A = 6378245.0


# ============================================================
# 坐标转换（向量化）
# ============================================================
def wgs84_to_gcj02_vec(lon, lat):
    """WGS84 经纬度 → GCJ-02（高德坐标系），numpy 向量化版本"""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    def _tlat(x, y):
        ret = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y
               + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x)))
        ret += (20.0 * np.sin(6.0 * x * np.pi)
                + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(y * np.pi)
                + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(y / 12.0 * np.pi)
                + 320.0 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _tlon(x, y):
        ret = (300.0 + x + 2.0 * y + 0.1 * x * x
               + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x)))
        ret += (20.0 * np.sin(6.0 * x * np.pi)
                + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(x * np.pi)
                + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(x / 12.0 * np.pi)
                + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
        return ret

    dlat = _tlat(lon - 105.0, lat - 35.0)
    dlon = _tlon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = np.sqrt(magic)

    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * np.pi)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * np.cos(radlat) * np.pi)
    return lon + dlon, lat + dlat


def to_webmercator(lon, lat):
    """经纬度（视为 GCJ-02）→ Web Mercator (EPSG:3857) 米"""
    mx = _A * np.radians(np.asarray(lon, dtype=np.float64))
    my = _A * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, dtype=np.float64)) / 2))
    return mx, my


# ============================================================
# 数据
# ============================================================
def load_all_points(csv_path, grid):
    """读取一个 CSV 的全部匹配栅格（原始 WGS84）"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    lons, lats, freqs = [], [], []
    for _, row in df.iterrows():
        info = grid.get((int(row["hex_x"]), int(row["hex_y"]), int(row["hex_z"])))
        if info is None:
            continue
        lons.append(info["lon"])
        lats.append(info["lat"])
        freqs.append(int(row["pass_count"]))
    return (np.array(lons), np.array(lats), np.array(freqs, dtype=np.float64))


def top_pct_indices(freqs, pct):
    n = max(1, int(len(freqs) * pct))
    return np.argsort(freqs)[-n:]


def load_merged_points(csv_paths, grid):
    """多个 CSV 合并（同一栅格 pass_count 相加），用于汇总行"""
    df = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig") for p in csv_paths], ignore_index=True
    )
    df = df.groupby(["hex_x", "hex_y", "hex_z"], as_index=False)["pass_count"].sum()
    lons, lats, freqs = [], [], []
    for _, row in df.iterrows():
        info = grid.get((int(row["hex_x"]), int(row["hex_y"]), int(row["hex_z"])))
        if info is None:
            continue
        lons.append(info["lon"])
        lats.append(info["lat"])
        freqs.append(int(row["pass_count"]))
    return (np.array(lons), np.array(lats), np.array(freqs, dtype=np.float64))


def compute_weights(freqs, lo=None, hi=None, mu=None, sd=None):
    """按 WEIGHT_MODE 把 pass_count 转成热力权重。
    minmax/zscore 传入全局统计量即「一起归一」；不传则退回各自归一。"""
    fr = np.asarray(freqs, dtype=np.float64)
    if WEIGHT_MODE == "log":
        return np.log1p(fr)
    if WEIGHT_MODE == "zscore":
        m = fr.mean() if mu is None else mu
        s = fr.std() if sd is None else sd
        return np.clip((fr - m) / (s + 1e-9), 0.0, None)
    lo = fr.min() if lo is None else lo
    hi = fr.max() if hi is None else hi
    return (fr - lo) / (hi - lo + 1e-9)


# ============================================================
# 主流程
# ============================================================
def main():
    # ---------- 1. 加载栅格字典 ----------
    print("加载栅格文件...", flush=True)
    with open(PKL_PATH, "rb") as f:
        grid = pickle.load(f)
    print(f"  栅格总数: {len(grid):,}", flush=True)

    # ---------- 2. 读 14 个 CSV，取 top5% 点的 GCJ-02 范围 ----------
    print("读取数据...", flush=True)
    data = {}
    g_lons, g_lats = [], []
    for cat in CATEGORIES:
        for day in DAYS:
            csv_path = os.path.join(INPUT_DIR, f"signal_cell_counts_{day}_strict_od_{cat}.csv")
            lons, lats, freqs = load_all_points(csv_path, grid)
            data[(cat, day)] = (lons, lats, freqs)
            idx = top_pct_indices(freqs, TOP_PCT)
            glon, glat = wgs84_to_gcj02_vec(lons[idx], lats[idx])
            g_lons.append(glon)
            g_lats.append(glat)
            print(f"  {day} {cat}: {len(lons):,} 点", flush=True)

    # 汇总行：合并 gte50 + lt50（同一栅格 pass_count 相加）
    for day in DAYS:
        lons, lats, freqs = load_merged_points([
            os.path.join(INPUT_DIR, f"signal_cell_counts_{day}_strict_od_gte50.csv"),
            os.path.join(INPUT_DIR, f"signal_cell_counts_{day}_strict_od_lt50.csv"),
        ], grid)
        data[("total", day)] = (lons, lats, freqs)
        idx = top_pct_indices(freqs, TOP_PCT)
        glon, glat = wgs84_to_gcj02_vec(lons[idx], lats[idx])
        g_lons.append(glon)
        g_lats.append(glat)
        print(f"  {day} total: {len(lons):,} 点(合并后)", flush=True)

    # GCJ-02 下 1~99 百分位定框，再外扩留白
    g_lons = np.concatenate(g_lons)
    g_lats = np.concatenate(g_lats)
    lon0 = np.percentile(g_lons, 1) - PAD_DEG
    lon1 = np.percentile(g_lons, 99) + PAD_DEG
    lat0 = np.percentile(g_lats, 1) - PAD_DEG
    lat1 = np.percentile(g_lats, 99) + PAD_DEG

    # 转 Web Mercator
    x0, _ = to_webmercator(lon0, lat0)
    x1, _ = to_webmercator(lon1, lat0)
    _, y0 = to_webmercator(lon0, lat0)
    _, y1 = to_webmercator(lon0, lat1)
    print(f"  公共范围 GCJ02 lon=[{lon0:.3f},{lon1:.3f}] lat=[{lat0:.3f},{lat1:.3f}]", flush=True)

    # ---------- 3. 拉一次高德底图（所有子图共用同一范围） ----------
    print("拉取高德底图...", flush=True)
    try:
        base_img, base_ext = cx.bounds2img(
            x0, y0, x1, y1,
            source=GAODE_URL, zoom=ZOOM, ll=False,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        # base_ext = [W, E, S, N]（米，按瓦片对齐）
        bx0, bx1, by0, by1 = base_ext
        have_basemap = True
        print(f"  底图 {base_img.shape[:2]} 像素，范围已按瓦片对齐", flush=True)
    except Exception as e:
        print(f"  底图拉取失败，退回无底图模式: {e}", flush=True)
        bx0, bx1, by0, by1 = x0, x1, y0, y1
        have_basemap = False

    # ---------- 4. 计算各子图平滑热力 H 及峰值 ----------
    print("计算热力栅格...", flush=True)
    # 全局统计量：minmax/zscore 用同一把尺子「一起归一」（所有行×日）
    all_freqs = np.concatenate([data[(row, d)][2] for row in ROWS for d in DAYS])
    g_lo, g_hi = float(all_freqs.min()), float(all_freqs.max())
    g_mu, g_sd = float(all_freqs.mean()), float(all_freqs.std())
    print(f"  全局 freq: min={g_lo:.0f} max={g_hi:.0f} mean={g_mu:.1f} std={g_sd:.1f}", flush=True)

    H_dict, peak = {}, {}
    for row in ROWS:
        for day in DAYS:
            lons, lats, freqs = data[(row, day)]
            idx = top_pct_indices(freqs, TOP_PCT)
            glon, glat = wgs84_to_gcj02_vec(lons[idx], lats[idx])
            mx, my = to_webmercator(glon, glat)
            w = compute_weights(freqs[idx], g_lo, g_hi, g_mu, g_sd)
            H, _, _ = np.histogram2d(
                my, mx, bins=GRID_PIX,
                range=[[by0, by1], [bx0, bx1]], weights=w,
            )
            H = gaussian_filter(H, sigma=SIGMA, mode="constant")
            H_dict[(row, day)] = H
            peak[(row, day)] = H.max()

    # ---------- 5. 全局共用一个 vmax（所有行×日的有信号格子 p99）----------
    VMAX_PCT = 99
    EPS = 1e-9
    all_active = np.stack([H_dict[(row, d)] for row in ROWS for d in DAYS]).ravel()
    all_active = all_active[all_active > EPS]
    vmax_global = float(np.percentile(all_active, VMAX_PCT)) if all_active.size else 1.0
    print(f"  全局 vmax(active p99)={vmax_global:.3f}")
    for row in ROWS:
        print(f"    {row:6s}: " + " ".join(f"{peak[(row, d)]:6.2f}" for d in DAYS))

    # ---------- 6. 绘制 3×7 拼图 ----------
    print("绘制拼图...", flush=True)
    fig, axes = plt.subplots(nrows=len(ROWS), ncols=7, figsize=(24, 19), constrained_layout=True)
    fig.suptitle(
        f"七天栅格通行频次热力图（上：≥50；中：<50；下：汇总）— "
        f"{WEIGHT_LABEL[WEIGHT_MODE]}加权，全局统一尺度，红色=满刻度",
        fontsize=15,
    )

    for r, row in enumerate(ROWS):
        for c, day in enumerate(DAYS):
            ax = axes[r, c]
            H = H_dict[(row, day)]
            Hn = H / vmax_global  # 三行共用全局尺度

            # 固定框，等比例（北朝上、不拉伸）
            ax.set_xlim(bx0, bx1)
            ax.set_ylim(by0, by1)
            ax.set_aspect("equal")

            # 底图（origin=upper 时 row0 在上=北）
            if have_basemap:
                ax.imshow(base_img, extent=[bx0, bx1, by0, by1])

            # 热力图：rows=纬度(北)、cols=经度(东)，origin=lower 让北上南下
            rgba = HEAT_CMAP(Hn)
            a = np.clip(Hn * 2.2, 0.0, 1.0)
            a[Hn < 0.015] = 0.0  # 极低值透明，透出底图
            rgba[..., 3] = a
            ax.imshow(rgba, extent=[bx0, bx1, by0, by1], origin="lower",
                      aspect="equal", interpolation="bilinear")

            if r == 0:
                weekday = WEEKDAY_CN[datetime.strptime(day, "%Y%m%d").weekday()]
                ax.set_title(f"{day[:4]}-{day[4:6]}-{day[6:8]} {weekday}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[r, 0].set_ylabel(ROW_LABEL[row], fontsize=12, labelpad=8)

    # 三行共用一个色标
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes.ravel().tolist(), location="right", shrink=0.85, pad=0.01,
        label=f"相对流量强度（{WEIGHT_LABEL[WEIGHT_MODE]}加权密度，红=满刻度）",
    )

    # ---------- 5. 保存 ----------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI)
    plt.close(fig)
    print(f"\n拼图已保存: {OUTPUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
