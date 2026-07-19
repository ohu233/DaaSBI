#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【性能优化版】清洗后的六边形轨迹 → 降采样 → OD → 路网匹配 → 栅格频次
支持中间表落盘，便于断点续跑和调试
"""

import gc
import heapq
import math
import os
import time
import warnings
from collections import Counter, OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StructField, StructType

# 解决 pandas 2.x 兼容性问题
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

warnings.filterwarnings("ignore")

# ============================================================
# 配置区域
# ============================================================
DATE = "20230917"
DB = "ss_seu_df"

# 原始表名仅保留作说明；当前脚本实际读取下面两个清洗后 hex 表。
SRC_TABLE = f"dataset_{DATE}"

CLEAN_OUTBOUND_TABLE = f"dataset_{DATE}_nanjing_to_gaochun_lishui_hex"
CLEAN_INBOUND_TABLE = f"dataset_{DATE}_gaochun_lishui_to_nanjing_hex"

ROAD_TABLE = "hex_grid_nanjing"

# 最终输出表
OUTBOUND_FREQ_TABLE = f"dataset_{DATE}_outbound_grid_frequency"
INBOUND_FREQ_TABLE = f"dataset_{DATE}_inbound_grid_frequency"

# 中间表（直接写在 ss_seu_df 库中）
OUTBOUND_DOWNSAMPLED_TABLE = f"dataset_{DATE}_outbound_downsampled"
INBOUND_DOWNSAMPLED_TABLE = f"dataset_{DATE}_inbound_downsampled"
OUTBOUND_OD_TABLE = f"dataset_{DATE}_outbound_od"
INBOUND_OD_TABLE = f"dataset_{DATE}_inbound_od"

# 降采样参数
STAY_MIN_DUR = 600
STAY_KEEP_INTERVAL = 3600
MOVE_MIN_DT = 180
MOVE_MIN_HEX = 8

# OD 切分阈值，保持原逻辑：cube 三坐标绝对差之和
OD_DISTANCE_THRESHOLD = 36

# 路网匹配参数
K = 12
MAX_SNAP_RADIUS = 30
MAX_ASTAR_EXPANSIONS = 80000

# 缓存大小
CANDIDATE_CACHE_SIZE = 200000
ROUTE_CACHE_SIZE = 20000

# 日志输出间隔
PROGRESS_EVERY = 10

# 清洗后轨迹表实际需要的字段
TRACK_COLUMNS = [
    "uid",
    "stime",
    "hex_coord",
    "time_value",
    "dist_value",
    "lat",
    "lon",
]

# ============================================================
# 六边形与交通模式
# ============================================================
MODE_LIST = ("GSD", "GG", "TS", "TG")
MODE_ORDER = {mode: i for i, mode in enumerate(MODE_LIST)}

HEX_DIRECTIONS = (
    (0, -1, +1),
    (-1, 0, +1),
    (-1, +1, 0),
    (0, +1, -1),
    (+1, 0, -1),
    (+1, -1, 0),
)


def hex_distance(c1, c2):
    return max(
        abs(c1[0] - c2[0]),
        abs(c1[1] - c2[1]),
        abs(c1[2] - c2[2]),
    )


@lru_cache(maxsize=128)
def hex_ring_offsets(radius):
    """缓存每个半径对应的六边形环偏移。"""
    if radius == 0:
        return ((0, 0, 0),)

    offsets = []
    q, r, s = 0, -radius, radius
    edge_order = (2, 3, 4, 5, 0, 1)

    for direction_index in edge_order:
        dq, dr, ds = HEX_DIRECTIONS[direction_index]
        for _ in range(radius):
            offsets.append((q, r, s))
            q += dq
            r += dr
            s += ds

    return tuple(offsets)


def normalize_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def code_to_modes(code):
    modes = []
    if ((code >> 2) & 1) == 1 or ((code >> 5) & 1) == 1:
        modes.append("GSD")
    if ((code >> 3) & 1) == 1 or ((code >> 4) & 1) == 1:
        modes.append("GG")
    if ((code >> 1) & 1) == 1:
        modes.append("TS")
    if ((code >> 6) & 1) == 1:
        modes.append("TG")
    return modes


def mode_switch_penalty(mode_a, mode_b):
    if mode_a == mode_b:
        return 0.0

    pair = {mode_a, mode_b}
    if pair == {"GSD", "GG"}:
        return 15.0
    if pair == {"TS", "TG"}:
        return 25.0

    road_modes = {"GSD", "GG"}
    transit_modes = {"TS", "TG"}
    if (
        mode_a in road_modes
        and mode_b in transit_modes
        or mode_a in transit_modes
        and mode_b in road_modes
    ):
        return 20.0

    return 30.0


MODE_SWITCH_COST = {
    (mode_a, mode_b): mode_switch_penalty(mode_a, mode_b)
    for mode_a in MODE_LIST
    for mode_b in MODE_LIST
}

# ============================================================
# 有限大小 LRU 缓存
# ============================================================
_CACHE_MISS = object()


class LRUCache:
    def __init__(self, maxsize):
        self.maxsize = max(0, int(maxsize))
        self._data = OrderedDict()

    def get(self, key, default=_CACHE_MISS):
        if self.maxsize <= 0:
            return default

        try:
            value = self._data.pop(key)
        except KeyError:
            return default

        self._data[key] = value
        return value

    def set(self, key, value):
        if self.maxsize <= 0:
            return

        if key in self._data:
            self._data.pop(key)
        self._data[key] = value

        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)

# ============================================================
# Hive 数据加载
# ============================================================
def qualified_table(table_name):
    if "." in table_name:
        return table_name
    return f"{DB}.{table_name}"


def table_exists(spark, table_name):
    """兼容旧版本Spark的表存在性检查"""
    full_name = qualified_table(table_name)
    try:
        spark.table(full_name).limit(0)
        return True
    except Exception:
        return False


def load_clean_hex_table(spark, table_name):
    full_name = qualified_table(table_name)
    
    if not table_exists(spark, table_name):
        raise RuntimeError(f"清洗后的 hex 表不存在: {full_name}")

    print(f"  读取 {full_name} ...", flush=True)
    spark_df = spark.table(full_name)

    missing = [column for column in TRACK_COLUMNS if column not in spark_df.columns]
    if missing:
        raise RuntimeError(
            f"表 {full_name} 缺少必要字段: {missing}; "
            f"现有字段: {spark_df.columns}"
        )

    return spark_df.select(*TRACK_COLUMNS).toPandas()


def load_road_from_hive(spark, road_table):
    """只加载匹配所需字段"""
    full_name = qualified_table(road_table)
    
    if not table_exists(spark, road_table):
        raise RuntimeError(f"路网表不存在: {full_name}")

    started = time.perf_counter()
    road_df = (
        spark.table(full_name)
        .select("x", "y", "z", "code")
        .toPandas()
    )

    mode_cells = {mode: set() for mode in MODE_LIST}
    cell_modes = {}

    for row in road_df.itertuples(index=False):
        cell = (int(row.x), int(row.y), int(row.z))
        modes = code_to_modes(int(row.code))
        if not modes:
            continue

        ordered_modes = tuple(sorted(modes, key=MODE_ORDER.__getitem__))
        cell_modes[cell] = ordered_modes
        for mode in ordered_modes:
            mode_cells[mode].add(cell)

    elapsed = time.perf_counter() - started
    print(
        f"  路网加载完成: 有效栅格 {len(cell_modes):,}, "
        f"耗时 {elapsed:.1f}s",
        flush=True,
    )

    del road_df
    gc.collect()
    return mode_cells, cell_modes


def save_to_hive(df, spark, table_name):
    """保存 DataFrame 到 Hive 表（兼容 pandas 2.x）"""
    if df.empty:
        print(f"  警告: DataFrame 为空，不保存 {table_name}")
        return
    
    full_name = qualified_table(table_name)
    
    # 重置索引，避免索引列问题
    df_to_save = df.reset_index(drop=True)
    
    # 处理时间戳列和复杂类型
    for col in df_to_save.columns:
        if df_to_save[col].dtype.name == 'datetime64[ns]':
            df_to_save[col] = df_to_save[col].astype(str)
        elif df_to_save[col].dtype.name == 'object':
            # 将对象类型转换为字符串，避免复杂类型
            df_to_save[col] = df_to_save[col].astype(str)
    
    # 使用 Arrow 创建，失败时自动回退
    try:
        spark_df = spark.createDataFrame(df_to_save)
    except (AttributeError, TypeError) as e:
        if 'iteritems' in str(e):
            print(f"  使用兼容模式转换数据...")
            # 手动转换：转换为列表元组
            data = [tuple(row) for row in df_to_save.to_numpy()]
            spark_df = spark.createDataFrame(data, schema=list(df_to_save.columns))
        else:
            raise
    
    spark_df.write.mode("overwrite").saveAsTable(full_name)
    print(f"  表已保存: {full_name}, 记录数: {len(df):,}", flush=True)


def load_from_hive(spark, table_name):
    """从 Hive 表加载数据为 pandas DataFrame"""
    if not table_exists(spark, table_name):
        return None
    
    full_name = qualified_table(table_name)
    print(f"  从中间表读取: {full_name}", flush=True)
    return spark.table(full_name).toPandas()

# ============================================================
# 路网匹配器
# ============================================================
class CompositeMatcher:
    def __init__(
        self,
        mode_cells,
        cell_modes,
        k=K,
        max_snap_radius=MAX_SNAP_RADIUS,
        max_expansions=MAX_ASTAR_EXPANSIONS,
        route_cache_size=ROUTE_CACHE_SIZE,
    ):
        self.mode_cells = mode_cells
        self.cell_modes = cell_modes
        self.k = int(k)
        self.max_snap_radius = int(max_snap_radius)
        self.max_expansions = int(max_expansions)
        self.route_cache = LRUCache(route_cache_size)

        self.multi_astar_calls = 0
        self.route_cache_hits = 0
        self.route_cache_misses = 0

    @lru_cache(maxsize=CANDIDATE_CACHE_SIZE)
    def find_candidates(self, point):
        """对相同信号栅格复用候选点结果"""
        candidates = []
        seen = set()
        px, py, pz = point

        for radius in range(self.max_snap_radius + 1):
            for dq, dr, ds in hex_ring_offsets(radius):
                cell = (px + dq, py + dr, pz + ds)
                modes = self.cell_modes.get(cell)
                if not modes:
                    continue

                for mode in modes:
                    node = (cell[0], cell[1], cell[2], mode)
                    if node in seen:
                        continue
                    seen.add(node)
                    candidates.append((node, radius))

            if len(candidates) >= self.k:
                break

        candidates.sort(
            key=lambda item: (
                item[1],
                MODE_ORDER[item[0][3]],
                item[0][0],
                item[0][1],
                item[0][2],
            )
        )
        return tuple(candidates[: self.k])

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return tuple(path)

    @staticmethod
    def _path_stats(path):
        if not path:
            return 0, 0

        move_len = 0
        switch_count = 0
        previous = path[0]

        for current in path[1:]:
            if previous[:3] != current[:3]:
                move_len += hex_distance(previous[:3], current[:3])
            if previous[3] != current[3]:
                switch_count += 1
            previous = current

        return move_len, switch_count

    def _multi_target_astar(self, start, goals):
        """从一个上一候选点，一次搜索多个当前候选点"""
        unique_goals = tuple(dict.fromkeys(goals))
        if not unique_goals:
            return {}

        if len(unique_goals) == 1 and unique_goals[0] == start:
            return {
                start: {
                    "cost": 0.0,
                    "path": (start,),
                    "move_len": 0,
                    "switch_count": 0,
                }
            }

        self.multi_astar_calls += 1

        start_cell = start[:3]
        goal_cells = tuple(goal[:3] for goal in unique_goals)
        remaining = set(unique_goals)
        results = {}

        max_straight = max(
            hex_distance(start_cell, goal_cell)
            for goal_cell in goal_cells
        )
        corridor_margin = max(
            20,
            min(90, int(max_straight * 1.5) + 10),
        )

        all_cells = (start_cell,) + goal_cells
        lower = tuple(
            min(cell[index] for cell in all_cells) - corridor_margin
            for index in range(3)
        )
        upper = tuple(
            max(cell[index] for cell in all_cells) + corridor_margin
            for index in range(3)
        )

        def in_corridor(cell):
            return (
                lower[0] <= cell[0] <= upper[0]
                and lower[1] <= cell[1] <= upper[1]
                and lower[2] <= cell[2] <= upper[2]
            )

        def heuristic(cell):
            return min(hex_distance(cell, goal_cell) for goal_cell in goal_cells)

        open_set = [(heuristic(start_cell), 0.0, start)]
        came_from = {}
        best_g = {start: 0.0}
        closed = set()
        expansions = 0

        mode_cells = self.mode_cells
        cell_modes = self.cell_modes
        directions = HEX_DIRECTIONS
        switch_cost = MODE_SWITCH_COST

        while open_set and remaining and expansions < self.max_expansions:
            _, g_cost, current = heapq.heappop(open_set)
            if current in closed:
                continue

            closed.add(current)
            expansions += 1

            if current in remaining:
                path = self._reconstruct_path(came_from, current)
                move_len, switch_count = self._path_stats(path)
                results[current] = {
                    "cost": g_cost,
                    "path": path,
                    "move_len": move_len,
                    "switch_count": switch_count,
                }
                remaining.remove(current)
                if not remaining:
                    break

            q, r, s, mode = current
            cell = (q, r, s)

            # 1. 同模式移动到相邻栅格
            current_mode_cells = mode_cells[mode]
            for dq, dr, ds in directions:
                nb_cell = (q + dq, r + dr, s + ds)
                if not in_corridor(nb_cell):
                    continue
                if nb_cell not in current_mode_cells:
                    continue

                neighbor = (nb_cell[0], nb_cell[1], nb_cell[2], mode)
                if neighbor in closed:
                    continue

                new_g = g_cost + 1.0
                if new_g < best_g.get(neighbor, math.inf):
                    best_g[neighbor] = new_g
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_set,
                        (new_g + heuristic(nb_cell), new_g, neighbor),
                    )

            # 2. 原地换模式
            for other_mode in cell_modes.get(cell, ()):
                if other_mode == mode:
                    continue

                neighbor = (q, r, s, other_mode)
                if neighbor in closed:
                    continue

                new_g = g_cost + switch_cost[(mode, other_mode)]
                if new_g < best_g.get(neighbor, math.inf):
                    best_g[neighbor] = new_g
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_set,
                        (new_g + heuristic(cell), new_g, neighbor),
                    )

            # 3. 移动到相邻栅格并换模式
            for dq, dr, ds in directions:
                nb_cell = (q + dq, r + dr, s + ds)
                if not in_corridor(nb_cell):
                    continue

                for other_mode in cell_modes.get(nb_cell, ()):
                    if other_mode == mode:
                        continue

                    neighbor = (
                        nb_cell[0],
                        nb_cell[1],
                        nb_cell[2],
                        other_mode,
                    )
                    if neighbor in closed:
                        continue

                    new_g = (
                        g_cost
                        + 1.0
                        + switch_cost[(mode, other_mode)]
                    )
                    if new_g < best_g.get(neighbor, math.inf):
                        best_g[neighbor] = new_g
                        came_from[neighbor] = current
                        heapq.heappush(
                            open_set,
                            (new_g + heuristic(nb_cell), new_g, neighbor),
                        )

        return results

    def _get_base_routes(self, start, goals):
        """先查有限 LRU 缓存，再对缺失目标执行一次多目标 A*"""
        routes = {}
        missing_goals = []

        for goal in goals:
            if start == goal:
                routes[goal] = {
                    "cost": 0.0,
                    "path": (start,),
                    "move_len": 0,
                    "switch_count": 0,
                }
                continue

            cached = self.route_cache.get((start, goal))
            if cached is _CACHE_MISS:
                self.route_cache_misses += 1
                missing_goals.append(goal)
            else:
                self.route_cache_hits += 1
                routes[goal] = cached

        if missing_goals:
            searched = self._multi_target_astar(start, tuple(missing_goals))

            for goal, result in searched.items():
                routes[goal] = result
                self.route_cache.set((start, goal), result)

                # 同时缓存反向路径
                reverse_result = {
                    "cost": result["cost"],
                    "path": tuple(reversed(result["path"])),
                    "move_len": result["move_len"],
                    "switch_count": result["switch_count"],
                }
                self.route_cache.set((goal, start), reverse_result)

        return routes

    def match_signal_points(self, signal_points):
        if not signal_points:
            return []

        candidates_by_point = []
        for point in signal_points:
            candidates = self.find_candidates(point)
            if not candidates:
                raise RuntimeError(f"信号点没有路网候选: {point}")
            candidates_by_point.append(candidates)

        point_count = len(signal_points)
        if point_count == 1:
            return []

        previous_scores = [
            float(snap_distance)
            for _, snap_distance in candidates_by_point[0]
        ]
        back_pointers = [[None] * len(candidates_by_point[0])]
        chosen_paths = [[None] * len(candidates_by_point[0])]

        for point_index in range(1, point_count):
            previous_candidates = candidates_by_point[point_index - 1]
            current_candidates = candidates_by_point[point_index]
            current_nodes = tuple(node for node, _ in current_candidates)

            current_scores = [math.inf] * len(current_candidates)
            current_back = [None] * len(current_candidates)
            current_paths = [None] * len(current_candidates)

            signal_distance = max(
                hex_distance(
                    signal_points[point_index - 1],
                    signal_points[point_index],
                ),
                1,
            )

            for previous_index, (previous_node, _) in enumerate(
                previous_candidates
            ):
                previous_score = previous_scores[previous_index]
                if math.isinf(previous_score):
                    continue

                base_routes = self._get_base_routes(
                    previous_node,
                    current_nodes,
                )

                for current_index, (current_node, snap_distance) in enumerate(
                    current_candidates
                ):
                    route = base_routes.get(current_node)
                    if route is None:
                        continue

                    excess = max(
                        0.0,
                        route["move_len"] - signal_distance * 2.5,
                    )
                    transition_cost = route["cost"] + excess
                    score = (
                        previous_score
                        + transition_cost
                        + float(snap_distance)
                    )

                    if score < current_scores[current_index]:
                        current_scores[current_index] = score
                        current_back[current_index] = previous_index
                        current_paths[current_index] = route["path"]

            if all(math.isinf(score) for score in current_scores):
                raise RuntimeError(
                    f"第 {point_index} 个信号点与上一点之间没有可连接候选"
                )

            previous_scores = current_scores
            back_pointers.append(current_back)
            chosen_paths.append(current_paths)

        best_last_index = min(
            range(len(previous_scores)),
            key=previous_scores.__getitem__,
        )
        if math.isinf(previous_scores[best_last_index]):
            raise RuntimeError("没有找到完整连通的候选序列")

        matched_indices = [None] * point_count
        matched_indices[-1] = best_last_index

        for point_index in range(point_count - 1, 0, -1):
            previous_index = back_pointers[point_index][
                matched_indices[point_index]
            ]
            if previous_index is None:
                raise RuntimeError(
                    f"候选序列回溯失败，位置: {point_index}"
                )
            matched_indices[point_index - 1] = previous_index

        route_cells = []
        for point_index in range(1, point_count):
            path = chosen_paths[point_index][matched_indices[point_index]]
            if not path:
                continue

            for node in path:
                cell = node[:3]
                if not route_cells or route_cells[-1] != cell:
                    route_cells.append(cell)

        return route_cells

    def print_stats(self):
        candidate_info = self.find_candidates.cache_info()
        total_route_cache_queries = self.route_cache_hits + self.route_cache_misses
        hit_rate = (
            self.route_cache_hits / total_route_cache_queries * 100.0
            if total_route_cache_queries
            else 0.0
        )

        print(
            "  匹配器统计: "
            f"多目标A*调用={self.multi_astar_calls:,}, "
            f"候选缓存命中={candidate_info.hits:,}, "
            f"候选缓存未命中={candidate_info.misses:,}, "
            f"路径缓存命中率={hit_rate:.1f}%, "
            f"路径缓存当前条目={len(self.route_cache):,}",
            flush=True,
        )

# ============================================================
# 轨迹转换、降采样、OD
# ============================================================
def parse_hex_coord(df):
    work = df.copy()
    parts = work["hex_coord"].astype(str).str.split(",", expand=True)

    if parts.shape[1] != 3:
        raise ValueError("hex_coord 必须是形如 'x,y,z' 的三段坐标")

    work["hex_x"] = pd.to_numeric(parts[0], errors="raise").astype(np.int32)
    work["hex_y"] = pd.to_numeric(parts[1], errors="raise").astype(np.int32)
    work["hex_z"] = pd.to_numeric(parts[2], errors="raise").astype(np.int32)

    bad = (work["hex_x"] + work["hex_y"] + work["hex_z"]) != 0
    if bad.any():
        work.loc[bad, "hex_z"] = (
            -work.loc[bad, "hex_x"] - work.loc[bad, "hex_y"]
        )

    return work


def add_unix_time(df):
    work = df.copy()
    work["stime_dt"] = pd.to_datetime(work["stime"], errors="coerce")

    bad_time = work["stime_dt"].isna()
    if bad_time.any():
        print(
            f"  警告: 删除无法解析 stime 的记录 {int(bad_time.sum()):,} 条",
            flush=True,
        )
        work = work.loc[~bad_time].copy()

    work["stime_ts"] = (
        work["stime_dt"].astype("int64") // 10**9
    ).astype(np.int64)

    work["time_value"] = pd.to_numeric(
        work["time_value"], errors="coerce"
    ).fillna(0)
    work["dist_value"] = pd.to_numeric(
        work["dist_value"], errors="coerce"
    ).fillna(0.0)

    return work


def downsample(df):
    if df.empty:
        return df.copy()

    work = df.copy()
    work["_date"] = work["stime_dt"].dt.strftime("%Y%m%d")
    work["_group"] = work["_date"] + "_" + work["uid"].astype(str)

    output_frames = []

    for _, group in work.groupby("_group", sort=False):
        g = group.sort_values("stime_ts", kind="mergesort").reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        timestamps = g["stime_ts"].to_numpy(dtype=np.int64, copy=False)
        hex_x = g["hex_x"].to_numpy(dtype=np.int64, copy=False)
        hex_y = g["hex_y"].to_numpy(dtype=np.int64, copy=False)
        hex_z = g["hex_z"].to_numpy(dtype=np.int64, copy=False)

        changed = np.empty(n, dtype=bool)
        changed[0] = True
        if n > 1:
            changed[1:] = (
                (hex_x[1:] != hex_x[:-1])
                | (hex_y[1:] != hex_y[:-1])
                | (hex_z[1:] != hex_z[:-1])
            )

        run_starts = np.flatnonzero(changed)
        run_ends = np.r_[run_starts[1:] - 1, n - 1]

        phase1_indices = []
        for start, end in zip(run_starts, run_ends):
            duration = int(timestamps[end] - timestamps[start])

            if duration < STAY_MIN_DUR:
                phase1_indices.extend(range(int(start), int(end) + 1))
                continue

            phase1_indices.append(int(start))
            phase1_indices.append(int(end))

            interval_count = int(duration / STAY_KEEP_INTERVAL)
            if interval_count > 1:
                target_offsets = (
                    np.arange(1, interval_count, dtype=np.int64)
                    * STAY_KEEP_INTERVAL
                )
                if target_offsets.size:
                    run_times = timestamps[start : end + 1]
                    positions = np.searchsorted(
                        run_times,
                        timestamps[start] + target_offsets,
                        side="left",
                    )
                    positions = np.minimum(positions, end - start)
                    phase1_indices.extend(
                        (positions + start).astype(int).tolist()
                    )

        phase1 = np.unique(np.asarray(phase1_indices, dtype=np.int64))
        if phase1.size == 0:
            continue

        kept = [int(phase1[0])]
        for index in phase1[1:]:
            index = int(index)
            last = kept[-1]
            delta_time = timestamps[index] - timestamps[last]
            delta_hex = (
                abs(hex_x[index] - hex_x[last])
                + abs(hex_y[index] - hex_y[last])
                + abs(hex_z[index] - hex_z[last])
            )

            if delta_time >= MOVE_MIN_DT or delta_hex >= MOVE_MIN_HEX:
                kept.append(index)

        final_phase1 = int(phase1[-1])
        if kept[-1] != final_phase1:
            kept.append(final_phase1)

        kept_array = np.asarray(kept, dtype=np.int64)
        previous_kept = np.r_[-1, kept_array[:-1]]

        time_values = g["time_value"].to_numpy(dtype=np.float64, copy=False)
        dist_values = g["dist_value"].to_numpy(dtype=np.float64, copy=False)
        cumulative_time = np.cumsum(time_values)
        cumulative_dist = np.cumsum(dist_values)

        previous_time_sum = np.where(
            previous_kept >= 0,
            cumulative_time[np.maximum(previous_kept, 0)],
            0.0,
        )
        previous_dist_sum = np.where(
            previous_kept >= 0,
            cumulative_dist[np.maximum(previous_kept, 0)],
            0.0,
        )

        aggregated_time = cumulative_time[kept_array] - previous_time_sum
        aggregated_dist = cumulative_dist[kept_array] - previous_dist_sum

        selected = g.iloc[kept_array].copy()
        selected["time_value"] = aggregated_time.astype(np.int64)
        selected["dist_value"] = aggregated_dist
        selected["velocity"] = np.divide(
            aggregated_dist * 3.6,
            aggregated_time,
            out=np.zeros_like(aggregated_dist, dtype=np.float64),
            where=aggregated_time > 0,
        )
        output_frames.append(selected)

    if not output_frames:
        return work.iloc[0:0].copy()

    output = pd.concat(output_frames, ignore_index=True)
    output.drop(
        columns=[column for column in ("_date", "_group") if column in output],
        inplace=True,
    )
    return output


def generate_od(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ID", "stime_o", "stime_d", "lat_o", "lon_o",
                "lat_d", "lon_d", "locxo", "locyo", "loczo",
                "locxd", "locyd", "loczd", "mode", "time", "distance",
            ]
        )

    work = df.copy()
    work["_date"] = work["stime_dt"].dt.strftime("%Y%m%d")
    rows = []

    for (date_text, uid), group in work.groupby(
        ["_date", "uid"], sort=False
    ):
        g = group.sort_values("stime_ts", kind="mergesort").reset_index(drop=True)
        if len(g) < 2:
            continue

        timestamps = g["stime"].to_numpy(copy=False)
        latitudes = g["lat"].to_numpy(copy=False)
        longitudes = g["lon"].to_numpy(copy=False)
        hex_x = g["hex_x"].to_numpy(dtype=np.int64, copy=False)
        hex_y = g["hex_y"].to_numpy(dtype=np.int64, copy=False)
        hex_z = g["hex_z"].to_numpy(dtype=np.int64, copy=False)
        time_values = g["time_value"].to_numpy(dtype=np.float64, copy=False)
        dist_values = g["dist_value"].to_numpy(dtype=np.float64, copy=False)

        group_id = f"{date_text}_{uid}"
        anchor_index = 0
        accumulated_time = 0.0
        accumulated_distance = 0.0

        for current_index in range(1, len(g)):
            accumulated_time += time_values[current_index]
            accumulated_distance += dist_values[current_index]

            manhattan_hex = (
                abs(hex_x[current_index] - hex_x[anchor_index])
                + abs(hex_y[current_index] - hex_y[anchor_index])
                + abs(hex_z[current_index] - hex_z[anchor_index])
            )

            if manhattan_hex < OD_DISTANCE_THRESHOLD:
                continue

            rows.append(
                {
                    "ID": group_id,
                    "stime_o": timestamps[anchor_index],
                    "stime_d": timestamps[current_index],
                    "lat_o": latitudes[anchor_index],
                    "lon_o": longitudes[anchor_index],
                    "lat_d": latitudes[current_index],
                    "lon_d": longitudes[current_index],
                    "locxo": int(hex_x[anchor_index]),
                    "locyo": int(hex_y[anchor_index]),
                    "loczo": int(hex_z[anchor_index]),
                    "locxd": int(hex_x[current_index]),
                    "locyd": int(hex_y[current_index]),
                    "loczd": int(hex_z[current_index]),
                    "mode": "GSD",
                    "time": int(accumulated_time),
                    "distance": float(accumulated_distance),
                }
            )

            anchor_index = current_index
            accumulated_time = 0.0
            accumulated_distance = 0.0

    order = [
        "ID", "stime_o", "stime_d", "lat_o", "lon_o", "lat_d", "lon_d",
        "locxo", "locyo", "loczo", "locxd", "locyd", "loczd",
        "mode", "time", "distance",
    ]
    return pd.DataFrame(rows, columns=order)


def build_signal_points_by_id(od_df):
    """一次性构建 ID → 信号点列表"""
    if od_df.empty:
        return []

    work = od_df.copy()
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    work["_norm_id"] = work["ID"].map(normalize_id)

    grouped_points = []

    for target_id, group in work.groupby("_norm_id", sort=False):
        g = group.sort_values("_row_order", kind="mergesort")

        origins = list(
            zip(
                g["locxo"].astype(int),
                g["locyo"].astype(int),
                g["loczo"].astype(int),
            )
        )
        last = g.iloc[-1]
        origins.append(
            (
                int(last["locxd"]),
                int(last["locyd"]),
                int(last["loczd"]),
            )
        )

        compact_points = []
        for point in origins:
            if not compact_points or compact_points[-1] != point:
                compact_points.append(point)

        grouped_points.append((target_id, tuple(compact_points)))

    return grouped_points

# ============================================================
# 频次写入与方向处理
# ============================================================
def write_frequency(freq, output_table, spark):
    rows = [
        (int(x), int(y), int(z), int(count))
        for (x, y, z), count in freq.items()
    ]

    schema = StructType(
        [
            StructField("locx", LongType(), False),
            StructField("locy", LongType(), False),
            StructField("locz", LongType(), False),
            StructField("frequency", LongType(), False),
        ]
    )

    spark_df = spark.createDataFrame(rows, schema=schema)
    full_name = qualified_table(output_table)
    spark_df.write.mode("overwrite").saveAsTable(full_name)

    print(
        f"  频次表已写入 Hive: {full_name}, 栅格数: {len(rows):,}",
        flush=True,
    )


def match_and_write_frequency(
    od_df,
    direction_label,
    matcher,
    spark,
    output_table,
):
    print(f"  [{direction_label}] 开始路网匹配...", flush=True)

    grouped_points = build_signal_points_by_id(od_df)
    total_ids = len(grouped_points)
    if total_ids == 0:
        print("  无 OD 对，写入空频次表", flush=True)
        write_frequency(Counter(), output_table, spark)
        return

    frequency = Counter()
    failed = 0
    started = time.perf_counter()

    for index, (target_id, signal_points) in enumerate(grouped_points, start=1):
        try:
            route_cells = matcher.match_signal_points(signal_points)
            frequency.update(route_cells)
        except Exception as error:
            failed += 1
            print(
                f"    [{index}/{total_ids}] {target_id} [FAILED] {error}",
                flush=True,
            )

        if (
            index == 1
            or index % PROGRESS_EVERY == 0
            or index == total_ids
        ):
            elapsed = time.perf_counter() - started
            speed = index / elapsed if elapsed > 0 else 0.0
            remaining_seconds = (
                (total_ids - index) / speed if speed > 0 else math.inf
            )
            print(
                f"    进度 {index:,}/{total_ids:,}, "
                f"失败 {failed:,}, "
                f"速度 {speed:.2f} ID/s, "
                f"预计剩余 {remaining_seconds / 60:.1f} 分钟",
                flush=True,
            )

    write_frequency(frequency, output_table, spark)

    elapsed = time.perf_counter() - started
    print(
        f"  [{direction_label}] 匹配完成: "
        f"成功 {total_ids - failed:,}, 失败 {failed:,}, "
        f"耗时 {elapsed / 60:.1f} 分钟",
        flush=True,
    )
    matcher.print_stats()

# ============================================================
# 主程序
# ============================================================
def process_direction(
    spark,
    clean_table,
    direction_label,
    downsampled_table,
    od_table,
    output_table,
    matcher,
):
    direction_started = time.perf_counter()
    print(f"\n===== 处理 {direction_label} =====", flush=True)
    
    # ========== 1. 降采样表 ==========
    downsampled_df = load_from_hive(spark, downsampled_table)
    
    if downsampled_df is None:
        print("  降采样表不存在，开始生成...", flush=True)
        
        raw_df = load_clean_hex_table(spark, clean_table)
        print(f"  原始记录: {len(raw_df):,}", flush=True)
        
        prepared_df = parse_hex_coord(raw_df)
        del raw_df
        gc.collect()
        
        prepared_df = add_unix_time(prepared_df)
        
        step_started = time.perf_counter()
        print("  降采样...", flush=True)
        downsampled_df = downsample(prepared_df)
        print(
            f"  降采样后: {len(downsampled_df):,} 行, "
            f"耗时 {time.perf_counter() - step_started:.1f}s",
            flush=True,
        )
        
        # 保存降采样表
        save_to_hive(downsampled_df, spark, downsampled_table)
        
        del prepared_df
        gc.collect()
    else:
        print(f"  降采样表已存在，直接使用: {len(downsampled_df):,} 行", flush=True)
    
    # ========== 2. OD 表 ==========
    od_df = load_from_hive(spark, od_table)
    
    if od_df is None:
        print("  OD 表不存在，开始生成...", flush=True)
        
        step_started = time.perf_counter()
        print("  生成 OD...", flush=True)
        od_df = generate_od(downsampled_df)
        print(
            f"  OD 对数: {len(od_df):,}, "
            f"耗时 {time.perf_counter() - step_started:.1f}s",
            flush=True,
        )
        
        # 保存 OD 表
        save_to_hive(od_df, spark, od_table)
        
        del downsampled_df
        gc.collect()
    else:
        print(f"  OD 表已存在，直接使用: {len(od_df):,} 条记录", flush=True)
    
    # ========== 3. 路网匹配并写入频次表 ==========
    match_and_write_frequency(
        od_df=od_df,
        direction_label=direction_label,
        matcher=matcher,
        spark=spark,
        output_table=output_table,
    )
    
    del od_df
    gc.collect()
    
    print(
        f"===== {direction_label} 总耗时: "
        f"{(time.perf_counter() - direction_started) / 60:.1f} 分钟 =====",
        flush=True,
    )


def main():
    total_started = time.perf_counter()

    spark = (
        SparkSession.builder
        .appName(f"Full_Pipeline_Optimized_{DATE}")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        spark.sql(f"USE {DB}")

        print("加载路网...", flush=True)
        mode_cells, cell_modes = load_road_from_hive(spark, ROAD_TABLE)

        # 两个方向共用候选缓存和路径缓存
        matcher = CompositeMatcher(
            mode_cells=mode_cells,
            cell_modes=cell_modes,
            k=K,
            max_snap_radius=MAX_SNAP_RADIUS,
            max_expansions=MAX_ASTAR_EXPANSIONS,
            route_cache_size=ROUTE_CACHE_SIZE,
        )

        # 处理出城方向
        process_direction(
            spark=spark,
            clean_table=CLEAN_OUTBOUND_TABLE,
            direction_label="OUTBOUND",
            downsampled_table=OUTBOUND_DOWNSAMPLED_TABLE,
            od_table=OUTBOUND_OD_TABLE,
            output_table=OUTBOUND_FREQ_TABLE,
            matcher=matcher,
        )

        # 处理进城方向
        process_direction(
            spark=spark,
            clean_table=CLEAN_INBOUND_TABLE,
            direction_label="INBOUND",
            downsampled_table=INBOUND_DOWNSAMPLED_TABLE,
            od_table=INBOUND_OD_TABLE,
            output_table=INBOUND_FREQ_TABLE,
            matcher=matcher,
        )

        print(
            f"\n全部完成，总耗时 "
            f"{(time.perf_counter() - total_started) / 60:.1f} 分钟",
            flush=True,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()