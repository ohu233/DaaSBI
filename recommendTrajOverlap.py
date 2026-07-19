"""
原始信令点 + 推荐路线重叠度匹配（无 A* 路径重建）

流程:
  1. 加载推荐路线 (traj_labeled_filtered.csv)
  2. 从 Hive 读取 with_stops 表,按 uid + index 取原始 hex 序列,去连续重复
  3. 对推荐路线做 BUFFER_DIST 格缓冲区,计算原始信令点命中率
  4. 输出匹配结果表到 Hive

用法:
  spark-submit --master yarn --deploy-mode cluster \
    --driver-memory 8g --executor-memory 4g \
    --files traj_labeled_filtered.csv \
    path_overlap_matching-buffer2.py
"""

import csv
import json
import traceback

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)
from pyspark.sql.window import Window

# ============================================================
# 常量
# ============================================================

DB = "ss_seu_df"
REF_CSV_PATH = "traj_labeled_filtered.csv"

DATES = [
    "20230917", "20230918", "20230919", "20230920",
    "20230921", "20230922", "20230923",
    "20250914", "20250915", "20250916", "20250917",
    "20250918", "20250919", "20250920",
]

BUFFER_DIST = 2


# ============================================================
# Hex 缓冲区工具
# ============================================================

def hex_buffer_offsets(buffer_dist):
    offsets = []
    for dx in range(-buffer_dist, buffer_dist + 1):
        for dy in range(-buffer_dist, buffer_dist + 1):
            dz = -dx - dy
            if abs(dz) <= buffer_dist:
                offsets.append((dx, dy, dz))
    return offsets


HEX_BUFFER_OFFSETS = hex_buffer_offsets(BUFFER_DIST)


def build_path_buffer(cells, offsets=HEX_BUFFER_OFFSETS):
    buffered = set()
    for x, y, z in cells:
        for dx, dy, dz in offsets:
            buffered.add((x + dx, y + dy, z + dz))
    return buffered


# ============================================================
# 信令点去重 + 重叠度计算
# ============================================================

def dedup_consecutive(hex_sequence):
    """去掉连续重复的 hex 格,保留首次出现。"""
    if not hex_sequence:
        return []
    result = [hex_sequence[0]]
    for h in hex_sequence[1:]:
        if h != result[-1]:
            result.append(h)
    return result


def buffered_overlap(signal_cells, reference_cells, reference_buffer=None,
                     buffer_dist=BUFFER_DIST):
    """计算原始信令点落在推荐路线缓冲区内的占比。

    分母取 min(len(signal_cells), len(reference_cells)),
    等价于 max(hit/|signal|, hit/|reference|)。
    """
    if not signal_cells or not reference_cells:
        return 0, set(), 0.0

    if not isinstance(signal_cells, set):
        signal_cells = set(signal_cells)
    if not isinstance(reference_cells, set):
        reference_cells = set(reference_cells)

    if reference_buffer is None:
        if buffer_dist == BUFFER_DIST:
            reference_buffer = build_path_buffer(reference_cells)
        else:
            reference_buffer = build_path_buffer(
                reference_cells, hex_buffer_offsets(buffer_dist)
            )

    hit_cells = signal_cells & reference_buffer
    hit_count = len(hit_cells)
    denominator = min(len(signal_cells), len(reference_cells))
    overlap = min(hit_count, denominator) / denominator if denominator else 0.0
    return hit_count, hit_cells, overlap


# ============================================================
# 推荐路线加载
# ============================================================

def load_reference_routes(csv_path):
    """从本地 CSV 加载推荐路线。"""
    print(f"[1/3] 加载推荐路线: {csv_path}")

    routes = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            traj = json.loads(row["traj"])
            path = [tuple(p) for p in traj]
            path_cells = set(path)
            routes.append({
                "path": path,
                "path_cells": path_cells,
                "buffered": build_path_buffer(path_cells),
                "mode": row.get("mode", ""),
                "travel_mode": row.get("travel_mode", ""),
                "start_city": row.get("start_city", ""),
                "start_county": row.get("start_county", ""),
                "end_city": row.get("end_city", ""),
                "end_county": row.get("end_county", ""),
            })

    print(f"  推荐路线: {len(routes)} 条")
    for i, r in enumerate(routes):
        print(f"    [{i}] {r['start_city']} -> {r['end_county']}, "
              f"{r['travel_mode']}, {len(r['path'])} 格")
    return routes


# ============================================================
# 单日期处理
# ============================================================

def process_date(spark, date_str, ref_routes, out_prefix="dataset"):
    """读取 with_stops 表,用原始信令点直接匹配推荐路线。"""
    table_to = f"{out_prefix}_{date_str}_to_gaochun_with_stops"
    table_from = f"{out_prefix}_{date_str}_from_gaochun_with_stops"

    all_results = []

    for direction, table_name in [("to_gaochun", table_to),
                                   ("from_gaochun", table_from)]:
        if table_name not in {r.tableName for r in spark.sql("SHOW TABLES").select("tableName").collect()}:
            print(f"  ⚠ 表不存在,跳过: {table_name}")
            continue

        df = spark.table(table_name).select("uid", "index", "hex_x", "hex_y", "hex_z")

        w = Window.partitionBy("uid").orderBy("index")
        sorted_df = df.withColumn("_rn", F.row_number().over(w))
        rows = sorted_df.orderBy("uid", "_rn").collect()

        trajectories = {}
        for r in rows:
            uid = r.uid
            if uid not in trajectories:
                trajectories[uid] = []
            trajectories[uid].append((r.hex_x, r.hex_y, r.hex_z))

        print(f"  [{direction}] {len(trajectories)} 条轨迹")

        for uid, hex_seq in trajectories.items():
            # 去连续重复,不做 A* 路径重建
            deduped = dedup_consecutive(hex_seq)
            if not deduped:
                continue
            signal_cells = set(deduped)

            best_ref_idx = -1
            best_overlap = -1.0
            best_hit_count = 0
            best_hit_cells = []
            best_travel_mode = ""
            best_mode = ""

            for j, ref in enumerate(ref_routes):
                hit_count, hit_cells, overlap = buffered_overlap(
                    signal_cells,
                    ref["path_cells"],
                    ref["buffered"],
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ref_idx = j
                    best_hit_count = hit_count
                    best_hit_cells = sorted(hit_cells)
                    best_travel_mode = ref["travel_mode"]
                    best_mode = ref["mode"]

            cells_str = ";".join(f"{x},{y},{z}" for x, y, z in best_hit_cells)

            all_results.append((
                date_str, direction, uid,
                best_ref_idx, best_travel_mode, best_mode,
                len(hex_seq), len(deduped),
                best_hit_count, round(best_overlap, 4),
                cells_str,
            ))

    return all_results


# ============================================================
# 主入口
# ============================================================

def main():
    spark = (
        SparkSession.builder
        .enableHiveSupport()
        .appName("path_overlap_matching_signal")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB}")
    spark.sql(f"USE {DB}")

    # 1. 加载推荐路线
    ref_routes = load_reference_routes(REF_CSV_PATH)

    # 2-3. 逐日期处理
    all_rows = []
    for date_str in DATES:
        print(f"\n[2/3] 处理日期: {date_str}")
        try:
            rows = process_date(spark, date_str, ref_routes)
            all_rows.extend(rows)
            print(f"  匹配结果: {len(rows)} 条")
        except Exception as e:
            print(f"  [ERROR] {date_str}: {e}")
            traceback.print_exc()

    if not all_rows:
        print("\n⚠ 无匹配结果,退出")
        return

    # 写出结果表
    print(f"\n[3/3] 写出结果 ({len(all_rows)} 行) ...")
    overlap_schema = StructType([
        StructField("date", StringType(), False),
        StructField("direction", StringType(), False),
        StructField("uid", StringType(), False),
        StructField("best_ref_idx", IntegerType(), False),
        StructField("travel_mode", StringType(), True),
        StructField("mode", StringType(), True),
        StructField("raw_points", IntegerType(), False),
        StructField("deduped_len", IntegerType(), False),
        StructField("hit_count", IntegerType(), False),
        StructField("overlap", DoubleType(), False),
        StructField("hit_cells", StringType(), True),
    ])
    spark.createDataFrame(all_rows, schema=overlap_schema) \
        .write.mode("overwrite").saveAsTable("path_overlap_results_buffer2_point")
    print("  ✅ path_overlap_results_buffer2_point")

    # 汇总
    print("\n汇总统计 ...")
    spark.sql("""
        SELECT date, direction, travel_mode, best_ref_idx,
               COUNT(*) as n_trajectories,
               ROUND(AVG(overlap), 4) as avg_overlap,
               ROUND(MAX(overlap), 4) as max_overlap,
               ROUND(AVG(hit_count), 1) as avg_hit_cells
        FROM path_overlap_results_buffer2_point        GROUP BY date, direction, travel_mode, best_ref_idx
        ORDER BY date, direction, travel_mode
    """).show(50, False)

    spark.stop()


if __name__ == "__main__":
    main()