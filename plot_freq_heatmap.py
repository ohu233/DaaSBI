#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
栅格频次热力图可视化
以高德瓦片为底图，展示各六边形栅格的通行频次
只对 CSV 中有频次数据的栅格做热力图扩散，不在表里的路网不绘制

遍历 data/23cells 下所有 CSV，逐个生成热力图并输出到 heatmap 文件夹

依赖: pip install folium pandas numpy
"""

import glob
import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "23cells")
PKL_PATH = os.path.join(BASE_DIR, "data", "nanjing_metro_hex_road_epsg2434.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "heatmap")

# 星期标签：datetime.weekday() → 周一..周日
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# ============================================================
# WGS84 → GCJ-02 坐标偏移（高德瓦片使用 GCJ-02 坐标系）
# ============================================================
_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(x, y):
    ret = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y
           + 0.1 * x * y + 0.2 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * _PI)
            + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI)
            + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI)
            + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x, y):
    ret = (300.0 + x + 2.0 * y + 0.1 * x * x
           + 0.1 * x * y + 0.1 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * _PI)
            + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI)
            + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI)
            + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon, lat):
    """将 WGS84 经纬度转换为 GCJ-02（高德坐标系）"""
    if _out_of_china(lon, lat):
        return lon, lat

    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)

    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)

    return lon + dlon, lat + dlat


# ============================================================
# 单个 CSV 的热力图绘制
# ============================================================
def plot_heatmap(csv_path, output_html, grid, top_pct=0.05):
    """对单个频次 CSV 生成热力图 HTML

    Args:
        csv_path:    输入 CSV 路径
        output_html: 输出 HTML 路径
        grid:        栅格字典 {(hex_x, hex_y, hex_z): {"lon", "lat", ...}}
        top_pct:     保留频率最高的 top 比例（默认 5%）
    """
    name = os.path.basename(csv_path)
    print(f"\n=== 处理 {name} ===", flush=True)

    # ---------- 1. 加载频次数据 ----------
    freq_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n = max(1, int(len(freq_df) * top_pct))
    freq_df = freq_df.nlargest(n, "pass_count")
    print(f"  记录数(top {top_pct:.0%}): {len(freq_df):,}", flush=True)
    print(f"  频次范围: [{freq_df['pass_count'].min()}, {freq_df['pass_count'].max()}]",
          flush=True)

    # ---------- 2. 匹配经纬度 ----------
    lons, lats, frequencies = [], [], []
    missed = 0

    for _, row in freq_df.iterrows():
        key = (int(row["hex_x"]), int(row["hex_y"]), int(row["hex_z"]))
        info = grid.get(key)
        if info is None:
            missed += 1
            continue

        gcj_lon, gcj_lat = wgs84_to_gcj02(info["lon"], info["lat"])
        lons.append(gcj_lon)
        lats.append(gcj_lat)
        frequencies.append(int(row["pass_count"]))

    print(f"  匹配成功: {len(lons):,}, 未命中: {missed:,}", flush=True)

    if not lons:
        print(f"  警告: 没有匹配到任何栅格，跳过 {name}")
        return

    # ---------- 3. 计算中心点和自适应缩放 ----------
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    max_span = max(lat_span, lon_span)
    zoom = (
        6 if max_span > 2.0 else
        7 if max_span > 1.0 else
        8 if max_span > 0.5 else
        9 if max_span > 0.2 else
        10 if max_span > 0.1 else
        11 if max_span > 0.05 else
        12
    )

    print(f"  地图中心: ({center_lat:.4f}, {center_lon:.4f}), zoom={zoom}", flush=True)

    # ---------- 4. 构建热力图数据 ----------
    freq_arr = np.array(frequencies, dtype=np.float64)
    # 对数归一化：压缩低频，拉开高频，让廊道更突出
    log_freq = np.log1p(freq_arr)
    log_max = log_freq.max()
    if log_max > 0:
        norm_freq = log_freq / log_max
    else:
        norm_freq = log_freq

    # 只用 CSV 中有数据的点，不在表里的路网栅格不参与
    heat_data = [
        [lats[i], lons[i], float(norm_freq[i])]
        for i in range(len(lons))
        if norm_freq[i] > 0
    ]

    # ---------- 5. 创建 folium 地图 ----------
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    # 高德路网瓦片
    folium.TileLayer(
        tiles=(
            "http://webrd0{s}.is.autonavi.com/appmaptile"
            "?lang=zh_cn&size=1&scale=1&style=8"
            "&x={x}&y={y}&z={z}"
        ),
        attr='&copy; <a href="http://www.gaode.com/">高德地图</a>',
        name="高德地图",
        subdomains="1234",
        max_zoom=18,
        min_zoom=3,
    ).add_to(m)

    # 高德卫星瓦片（默认隐藏，可切换）
    folium.TileLayer(
        tiles=(
            "http://webst0{s}.is.autonavi.com/appmaptile"
            "?style=6&x={x}&y={y}&z={z}"
        ),
        attr='&copy; <a href="http://www.gaode.com/">高德卫星</a>',
        name="高德卫星",
        subdomains="1234",
        max_zoom=18,
        min_zoom=3,
        show=False,
    ).add_to(m)

    # 热力图图层 — 只有 CSV 中存在的栅格才参与扩散
    HeatMap(
        heat_data,
        name="栅格频次热力图",
        min_opacity=0.3,
        max_opacity=0.9,
        radius=12,
        blur=15,
        max_zoom=zoom + 3,
        gradient={
            0.0: "blue",
            0.25: "cyan",
            0.5: "lime",
            0.75: "yellow",
            1.0: "red",
        },
    ).add_to(m)

    # 图层控制开关
    folium.LayerControl().add_to(m)

    # ---------- 6. 保存 ----------
    m.save(output_html)
    print(f"  已保存: {output_html}", flush=True)
    print(f"  有效栅格: {len(heat_data):,}, 频次最大值: {int(freq_arr.max()):,}", flush=True)


# ============================================================
# 主流程
# ============================================================
def main():
    # ---------- 1. 加载栅格字典（只加载一次） ----------
    print("加载栅格文件...", flush=True)
    with open(PKL_PATH, "rb") as f:
        grid = pickle.load(f)
    print(f"  栅格总数: {len(grid):,}", flush=True)

    # ---------- 2. 收集输入 CSV ----------
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "signal_cell_counts_*.csv")))
    if not csv_files:
        print(f"错误: 在 {INPUT_DIR} 下未找到任何 CSV 文件")
        return
    print(f"\n共发现 {len(csv_files)} 个 CSV 文件", flush=True)

    # ---------- 3. 准备输出目录 ----------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 4. 逐个生成热力图 ----------
    for i, csv_path in enumerate(csv_files, 1):
        # 文件名形如 signal_cell_counts_<YYYYMMDD>_<att>.csv
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        att = stem.replace("signal_cell_counts_", "", 1)
        # att 形如 20230917_strict_od_gte50 → 解析日期得到星期几
        date_str = att[:8]
        try:
            weekday = WEEKDAY_CN[datetime.strptime(date_str, "%Y%m%d").weekday()]
        except ValueError:
            weekday = "未知"
        output_html = os.path.join(
            OUTPUT_DIR, f"freq_heatmap_{att}_{weekday}.html"
        )

        print(f"\n[{i}/{len(csv_files)}] 处理中...", flush=True)
        try:
            plot_heatmap(csv_path, output_html, grid)
        except Exception as e:
            print(f"  失败: {e}", flush=True)
            continue

    print(f"\n全部完成，输出目录: {OUTPUT_DIR}", flush=True)
    print("请在浏览器中打开 HTML 文件查看", flush=True)


if __name__ == "__main__":
    main()
