'''
输入：dataset_YYYYMMDD（YYYYMMDD为日期，20230917-20230923和20250914-20250920），包含以下列：
- uid: 用户唯一id(eg: 1)
- index: 轨迹点在用户轨迹中的索引，从0开始(eg: 1)
- stime: 轨迹点的时间戳(eg: 1694951291)
- lat: 轨迹点的纬度(eg: 31.946001)
- lon: 轨迹点的经度(eg: 120.601)

对dataset_YYYYMMDD进行处理：
1. 预筛选：先筛选南京到高淳、溧水的OD数据，大幅减少后续处理的数据量
    - 南京：lat_min 31.88, lon_min 118.62, lat_max 32.15, lon_max 118.95
    - 高淳：lat_min 31.23, lon_min 118.78, lat_max 31.43, lon_max 119.08
    - 溧水：lat_min 31.39, lon_min 118.88, lat_max 31.70, lon_max 119.22

2. 乒乓数据，漂移数据处理：参考An adaptive staying point recognition algorithm based on spatiotemporal characteristics using cellular signaling data
处理方法：
    · 删除重复记录（dropDuplicates）
    · 汇聚连续相同坐标记录（同一uid内连续相同lat/lon合并为一行，时间取平均）
    · 汇聚连续相同时间记录（同一uid内连续相同stime合并为一行，坐标取平均）
    · 乒乓数据：检测基站切换回跳（A→B→A...），用AB点平均坐标与时间替代，attribution标记为pingpong
    · 漂移数据：计算相邻记录速度，标记超过600km/h的超速记录，超速记录删除

3. 后筛选：merge/pingpong平均坐标后个别点可能偏移出范围，再次筛选

4. 如果筛选后轨迹只有一个点，则丢弃

5. 计算相邻轨迹点之间的时间差和空间距离（使用haversine公式计算地理距离），保存在后一行的time_value和dist_value列中（每个uid第一行无差分）

6. 输出表格，形式为dataset_YYYYMMDD_NanJing_to_GaoChun_LiShui，包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引，从0开始
- stime: 轨迹点的时间戳
- lat: 轨迹点的纬度
- lon: 轨迹点的经度
- time_value: 与上一轨迹点的时间差（单位：秒），如果没有上一点则为0
- dist_value: 与上一轨迹点的空间距离（单位：米），如果没有上一点则为0
- velocity: 速度（单位：千米/时），time_value为0时填0
- attribution: 处理标记（origin/drift/pingpong）
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import traceback
import os


DEFAULT_DATES = [
    "20230917",
    "20230918",
    "20230919",
    "20230920",
    "20230921",
    "20230922",
    "20230923",
    "20250914",
    "20250915",
    "20250916",
    "20250917",
    "20250918",
    "20250919",
    "20250920",
]

class HiveTable:
    MAX_SPEED_KMH = 600            # 600 km/h drift threshold
    PINGPONG_TIME_THRESHOLD = 300  # seconds

    # City bounding boxes for OD filtering
    NANJING_BOX = dict(lat_min=30.70, lon_min=118.00, lat_max=32.45, lon_max=119.25)
    GAOCHUN_BOX = dict(lat_min=31.23, lon_min=118.78, lat_max=31.43, lon_max=119.08)
    LISHUI_BOX = dict(lat_min=31.39, lon_min=118.88, lat_max=31.70, lon_max=119.22)

    def __init__(self, db="ss_seu_df", local=False):
        builder = SparkSession.builder.enableHiveSupport()

        if local:
            warehouse = f"file://{os.path.expanduser('~/hive/warehouse')}"
            builder = (
                builder
                .appName("filter_nanjing_od")
                .master("local[*]")
                .config("spark.hadoop.hive.metastore.uris", "thrift://localhost:9083")
                .config("spark.sql.warehouse.dir", warehouse)
                .config("spark.sql.hive.metastore.version", "4.1.0")
                .config("spark.sql.hive.metastore.jars", "maven")
                .config("spark.sql.ansi.enabled", "false")
            )

        session = builder.getOrCreate()
        session.sparkContext.setLogLevel("WARN")
        session.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
        session.sql(f"USE {db}")
        self.__session = session
        self.__local = local

    def stop(self):
        self.__session.stop()

    @staticmethod
    def _pick_first_existing(columns, candidates):
        for col_name in candidates:
            if col_name in columns:
                return col_name
        return None

    @staticmethod
    def parse_time_col(col_name="stime"):
        return F.from_unixtime(F.col(col_name).cast("long")).cast("timestamp")

    def _table_exists(self, table_name):
        table_names = {
            row["tableName"]
            for row in self.__session.sql("SHOW TABLES").select("tableName").collect()
        }
        return table_name in table_names

    def _add_missing_optional_columns(self, df):
        cols = set(df.columns)
        if "cid" not in cols:
            df = df.withColumn("cid", F.lit(None).cast("string"))
        if "city" not in cols:
            df = df.withColumn("city", F.lit(None).cast("string"))
        if "province" not in cols:
            df = df.withColumn("province", F.lit(None).cast("string"))
        return df

    def _add_time_dist_columns(self, df):
        w = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))

        with_prev = (
            df
            .withColumn("prev_stime_ts", F.lag("stime_ts").over(w))
            .withColumn("prev_lat", F.lag("lat_d").over(w))
            .withColumn("prev_lon", F.lag("lon_d").over(w))
        )

        time_diff = (
            F.col("stime_ts").cast("long") - F.col("prev_stime_ts").cast("long")
        ).cast("double")

        lat1 = F.radians(F.col("prev_lat"))
        lon1 = F.radians(F.col("prev_lon"))
        lat2 = F.radians(F.col("lat_d"))
        lon2 = F.radians(F.col("lon_d"))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            F.pow(F.sin(dlat / F.lit(2.0)), F.lit(2.0))
            + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / F.lit(2.0)), F.lit(2.0))
        )
        c = F.lit(2.0) * F.atan2(F.sqrt(a), F.sqrt(F.lit(1.0) - a))
        haversine_dist = F.lit(6371000.0) * c

        return (
            with_prev
            .withColumn(
                "time_value",
                F.when(F.col("prev_stime_ts").isNull(), F.lit(0.0))
                .otherwise(F.greatest(time_diff, F.lit(0.0))),
            )
            .withColumn(
                "dist_value",
                F.when(F.col("prev_stime_ts").isNull(), F.lit(0.0))
                .otherwise(F.coalesce(haversine_dist.cast("double"), F.lit(0.0))),
            )
            .drop("prev_stime_ts", "prev_lat", "prev_lon")
        )

    def _merge_same_coord(self, df):
        w = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))

        coord_changed = (
            F.lag("lat_d").over(w).isNull()
            | (F.col("lat_d") != F.lag("lat_d").over(w))
            | (F.col("lon_d") != F.lag("lon_d").over(w))
        ).cast("int")

        stay_group = F.sum(coord_changed).over(
            w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        return (
            df.withColumn("stay_group", stay_group)
            .groupBy("uid", "stay_group")
            .agg(
                F.first("index_i").alias("index_i"),
                F.from_unixtime(F.avg(F.col("stime_ts").cast("long"))).cast("timestamp").alias("stime_ts"),
                F.first("cid").alias("cid"),
                F.first("lat_d").alias("lat_d"),
                F.first("lon_d").alias("lon_d"),
                F.first("city").alias("city"),
                F.first("province").alias("province"),
                F.count("*").alias("_merge_cnt"),
            )
            .drop("stay_group")
        )

    def _merge_same_time(self, df):
        return (
            df.groupBy("uid", "stime_ts")
            .agg(
                F.min("index_i").alias("index_i"),
                F.first("cid").alias("cid"),
                F.avg("lat_d").alias("lat_d"),
                F.avg("lon_d").alias("lon_d"),
                F.first("city").alias("city"),
                F.first("province").alias("province"),
                F.first("attribution").alias("attribution"),
                F.count("*").alias("_merge_time_cnt"),
            )
        )

    def _remove_drift(self, df):
        is_overspeed = (
            (F.col("time_value") > 0)
            & (F.col("dist_value") > 0)
            & ((F.col("dist_value") / F.col("time_value") * F.lit(3.6)) > F.lit(self.MAX_SPEED_KMH))
        )
        return df.where(~is_overspeed).drop("time_value", "dist_value")

    def _fix_pingpong(self, df):
        """Detect longest A⇆B oscillation chains (min ABA) and merge each chain into one row."""
        w = Window.partitionBy("uid").orderBy("index_i", "stime_ts")

        with_ctx = (
            df
            .withColumn("_p1_lat", F.lag("lat_d", 1).over(w))
            .withColumn("_p1_lon", F.lag("lon_d", 1).over(w))
            .withColumn("_p2_lat", F.lag("lat_d", 2).over(w))
            .withColumn("_p2_lon", F.lag("lon_d", 2).over(w))
            .withColumn("_p2_ts",  F.lag("stime_ts", 2).over(w))
            # materialize is_echo
            .withColumn(
                "_is_echo",
                F.col("_p2_lat").isNotNull()
                & (F.col("lat_d") == F.col("_p2_lat"))
                & (F.col("lon_d") == F.col("_p2_lon"))
                & ((F.col("lat_d") != F.col("_p1_lat")) | (F.col("lon_d") != F.col("_p1_lon")))
                & (
                    (F.col("stime_ts").cast("long") - F.col("_p2_ts").cast("long"))
                    < F.lit(self.PINGPONG_TIME_THRESHOLD)
                ),
            )
            # materialize in_osc (echo self, or prev will echo, or prev-prev will echo)
            .withColumn(
                "_in_osc",
                F.coalesce(F.col("_is_echo"), F.lit(False))
                | F.coalesce(F.lead("_is_echo", 1).over(w), F.lit(False))
                | F.coalesce(F.lead("_is_echo", 2).over(w), F.lit(False)),
            )
            # materialize osc_grp
            .withColumn(
                "_prev_in_osc",
                F.lag("_in_osc", 1, False).over(w),
            )
            .withColumn(
                "_boundary",
                ((F.col("_in_osc") != F.col("_prev_in_osc")) | (~F.col("_in_osc"))).cast("int"),
            )
            .withColumn(
                "osc_grp",
                F.sum("_boundary").over(
                    w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
                ),
            )
        )

        return (
            with_ctx
            .groupBy("uid", "osc_grp")
            .agg(
                F.min("index_i").alias("index_i"),
                F.from_unixtime(F.avg(F.col("stime_ts").cast("long"))).cast("timestamp").alias("stime_ts"),
                F.first("cid").alias("cid"),
                F.avg("lat_d").alias("lat_d"),
                F.avg("lon_d").alias("lon_d"),
                F.first("city").alias("city"),
                F.first("province").alias("province"),
                F.max("_in_osc").alias("_is_osc"),
                F.first("attribution").alias("_orig_attr"),
            )
            .withColumn(
                "attribution",
                F.when(F.col("_is_osc"), F.lit("pingpong"))
                .otherwise(F.col("_orig_attr")),
            )
            .drop("_is_osc", "_orig_attr", "osc_grp")
        )

    @staticmethod
    def _in_city(lat_col, lon_col, box):
        return (
            (lat_col >= F.lit(box["lat_min"]))
            & (lat_col <= F.lit(box["lat_max"]))
            & (lon_col >= F.lit(box["lon_min"]))
            & (lon_col <= F.lit(box["lon_max"]))
        )

    def _filter_nanjing_od(self, df):
        """Keep users whose origin is in Nanjing and either:
        - destination is in Gaochun/Lishui, or
        - destination is also Nanjing but passed through Gaochun/Lishui (round trip)."""
        tagged = (
            df
            .withColumn("_in_nj", self._in_city(F.col("lat_d"), F.col("lon_d"), self.NANJING_BOX))
            .withColumn(
                "_in_dest",
                self._in_city(F.col("lat_d"), F.col("lon_d"), self.GAOCHUN_BOX)
                | self._in_city(F.col("lat_d"), F.col("lon_d"), self.LISHUI_BOX),
            )
        )

        w = Window.partitionBy("uid").orderBy("index_i")
        w_rev = Window.partitionBy("uid").orderBy(F.col("index_i").desc())
        w_all = Window.partitionBy("uid")

        return (
            tagged
            .withColumn("_first_in_nj", F.first("_in_nj").over(w))
            .withColumn("_last_in_dest", F.first("_in_dest").over(w_rev))
            .withColumn("_last_in_nj", F.first("_in_nj").over(w_rev))
            .withColumn("_has_any_dest", F.max(F.col("_in_dest").cast("int")).over(w_all) == 1)
            .where(
                F.col("_first_in_nj")
                & (F.col("_last_in_dest") | (F.col("_last_in_nj") & F.col("_has_any_dest")))
            )
            .drop("_in_nj", "_in_dest", "_first_in_nj", "_last_in_dest", "_last_in_nj", "_has_any_dest")
        )

    def _resolve_src_table(self, date_str, src_prefix="dataset"):
        candidates = [
            f"{src_prefix}_{date_str}",
            f"{src_prefix}__{date_str}",
        ]
        for table_name in candidates:
            if self._table_exists(table_name):
                return table_name
        raise ValueError(
            f"Cannot find source table for date {date_str}. Tried: {', '.join(candidates)}"
        )

    def _build_multicity_detail_df(self, src_table):
        df = self.__session.table(src_table)
        columns = set(df.columns)

        uid_col = self._pick_first_existing(columns, ["uid", "user_id"])
        if uid_col is None:
            raise ValueError(f"Table {src_table} does not have uid/user_id column")

        df = self._add_missing_optional_columns(df)

        required_cols = ["index", "stime", "lat", "lon"]
        missing = [c for c in required_cols if c not in set(df.columns)]
        if missing:
            raise ValueError(f"Table {src_table} missing required columns: {', '.join(missing)}")

        detail_df = (
            df.where(
                F.col(uid_col).isNotNull()
                & F.col("index").isNotNull()
                & F.col("stime").isNotNull()
                & F.col("lat").isNotNull()
                & F.col("lon").isNotNull()
            )
            .withColumn("uid", F.col(uid_col).cast("string"))
            .withColumn("index_i", F.col("index").cast("long"))
            .withColumn("stime_ts", self.parse_time_col("stime"))
            .withColumn("lat_d", F.col("lat").cast("double"))
            .withColumn("lon_d", F.col("lon").cast("double"))
            .where(
                F.col("index_i").isNotNull()
                & F.col("stime_ts").isNotNull()
                & F.col("lat_d").isNotNull()
                & F.col("lon_d").isNotNull()
            )
        )

        # Pre-filter: keep only users in Nanjing & Gaochun/Lishui before expensive processing
        result_df = self._filter_nanjing_od(detail_df)

        # Step 1: drop duplicates
        result_df = result_df.dropDuplicates()

        # Step 2: merge consecutive same-coordinate records
        result_df = (
            self._merge_same_coord(result_df)
            .withColumn(
                "attribution",
                F.when(F.col("_merge_cnt") > 1, F.lit("merge"))
                .otherwise(F.lit("origin")),
            )
            .drop("_merge_cnt")
        )

        # Step 2b: merge same-time records (avg coordinates)
        result_df = (
            self._merge_same_time(result_df)
            .withColumn(
                "attribution",
                F.when(F.col("_merge_time_cnt") > 1, F.lit("merge"))
                .otherwise(F.col("attribution")),
            )
            .drop("_merge_time_cnt")
        )

        # Step 3: handle ping-pong (detect and merge oscillation chains)
        result_df = self._fix_pingpong(result_df)

        # Step 4: drift removal loop (delete >600km/h, recalc, repeat until clean)
        result_df = result_df.localCheckpoint()
        while True:
            result_df = self._add_time_dist_columns(result_df).localCheckpoint()
            before = result_df.count()
            result_df = self._remove_drift(result_df).orderBy("uid", "index_i").localCheckpoint()
            if result_df.count() == before:
                break

        # Step 5: post-filter (re-filter after merge/pingpong may shift coords out of target area)
        result_df = self._filter_nanjing_od(result_df)

        # Step 6: drop users with only 1 trajectory point
        uid_counts = result_df.groupBy("uid").count().where(F.col("count") >= 2).select("uid")
        result_df = result_df.join(uid_counts, "uid", "inner")

        # Final time/dist calculation
        result_df = self._add_time_dist_columns(result_df)

        # Re-number idx from 1 per uid after processing
        idx_window = Window.partitionBy("uid").orderBy("index_i")
        result_df = result_df.withColumn("idx", F.row_number().over(idx_window))

        return result_df.select(
            "uid",
            F.col("index_i").cast("long").alias("index"),
            "idx",
            F.date_format("stime_ts", "yyyy-MM-dd HH:mm:ss").alias("stime"),
            F.col("lat_d").alias("lat"),
            F.col("lon_d").alias("lon"),
            F.coalesce(F.col("time_value"), F.lit(0.0)).alias("time_value"),
            F.coalesce(F.col("dist_value"), F.lit(0.0)).alias("dist_value"),
            F.when(
                F.col("time_value") > 0,
                F.col("dist_value") / F.col("time_value") * F.lit(3.6),
            ).otherwise(F.lit(0.0)).alias("velocity"),
            "attribution",
        )

    def _build_single_day_multicity_table(
        self,
        date_str,
        src_prefix="dataset",
        out_prefix="dataset",
    ):
        src_table = self._resolve_src_table(date_str=date_str, src_prefix=src_prefix)
        out_table = f"{out_prefix}_{date_str}_NanJing_to_GaoChun_LiShui"
        detail_df = self._build_multicity_detail_df(src_table)
        row_count = detail_df.count()
        detail_df.write.mode("overwrite").saveAsTable(out_table)
        print(f"Saved table: {out_table} rows={row_count}, from: {src_table}")
        return out_table

    def run_processing(
        self,
        date_list=None,
        src_prefix="dataset",
        out_prefix="dataset",
    ):
        if date_list is None:
            date_list = DEFAULT_DATES

        failed_dates = []
        for date_str in date_list:
            print(f"[INFO] Start processing date: {date_str}")
            try:
                self._build_single_day_multicity_table(
                    date_str=date_str,
                    src_prefix=src_prefix,
                    out_prefix=out_prefix,
                )
            except Exception as exc:
                failed_dates.append((date_str, str(exc)))
                print(f"[WARN] Skip date {date_str}: {exc}")
                print(traceback.format_exc())

        if failed_dates:
            print("[WARN] Failed dates summary:")
            for date_str, reason in failed_dates:
                print(f"  - {date_str}: {reason}")


if __name__ == "__main__":
    # Set local=True for local Hive testing; local=False (default) for YARN cluster
    table = HiveTable(db="ss_seu_df", local=False)
    try:
        table.run_processing(
            date_list=DEFAULT_DATES,
            src_prefix="dataset",
            out_prefix="dataset",
        )
    finally:
        table.stop()
# spark-submit: --master yarn --deploy-mode cluster