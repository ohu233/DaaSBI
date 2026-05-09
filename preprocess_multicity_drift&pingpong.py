'''
输入：dataset_multicity_YYYYMMDD（YYYYMMDD为日期，20230917-20230923和20250914-20250920），包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引，从0开始
- stime: 轨迹点的时间戳
- lat: 轨迹点的纬度
- lon: 轨迹点的经度

对dataset_multicity_YYYYMMDD进行处理：
1. 漂移数据，乒乓数据处理：参考An adaptive staying point recognition algorithm based on spatiotemporal characteristics using cellular signaling data
处理方法：
    · 删除重复记录（dropDuplicates）
    · 汇聚连续相同坐标记录（同一uid内连续相同lat/lon合并为一行，时间取平均）
    · 漂移数据：计算相邻记录速度，删除超过300km/h的超速记录
    · 乒乓数据：检测基站切换回跳（A→B→A），用AB点平均坐标与时间替代
2. 计算相邻轨迹点之间的时间差和空间距离（使用haversine公式计算地理距离），保存在后一行的time_value和dist_value列中（每个uid第一行无差分）
3. 输出表格，形式为dataset_multicity_YYYYMMDD，包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引，从0开始
- stime: 轨迹点的时间戳
- lat: 轨迹点的纬度
- lon: 轨迹点的经度
- time_value: 与下一轨迹点的时间差（单位：秒），如果没有下一点则为0
- dist_value: 与下一轨迹点的空间距离（单位：米），如果没有下一点则为0
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window
import traceback
import os


DEFAULT_DATES = ["20230917"]

class HiveTable:
    MAX_SPEED_MPS = 83.33          # 300 km/h
    PINGPONG_TIME_THRESHOLD = 300  # seconds

    def __init__(self, db="ss_seu_df", local=False):
        builder = SparkSession.builder.enableHiveSupport()

        if local:
            warehouse = f"file://{os.path.expanduser('~/hive/warehouse')}"
            builder = (
                builder
                .appName("preprocess_multicity")
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
    def _table_date(table_name):
        return table_name.rsplit("_", 1)[-1]

    @staticmethod
    def _pick_first_existing(columns, candidates):
        for col_name in candidates:
            if col_name in columns:
                return col_name
        return None

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
            )
            .drop("stay_group")
        )

    def _remove_drift(self, df):
        is_overspeed = (
            (F.col("time_value") > 0)
            & (F.col("dist_value") > 0)
            & ((F.col("dist_value") / F.col("time_value")) > F.lit(self.MAX_SPEED_MPS))
        )
        return (
            df.where(~is_overspeed)
            .drop("time_value", "dist_value")
        )

    def _fix_pingpong(self, df):
        w = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))

        with_neighbors = (
            df
            .withColumn("_prev_cid", F.lag("cid").over(w))
            .withColumn("_next_cid", F.lead("cid").over(w))
            .withColumn("_prev_lat", F.lag("lat_d").over(w))
            .withColumn("_next_lat", F.lead("lat_d").over(w))
            .withColumn("_prev_lon", F.lag("lon_d").over(w))
            .withColumn("_next_lon", F.lead("lon_d").over(w))
            .withColumn("_prev_ts", F.lag("stime_ts").over(w))
            .withColumn("_next_ts", F.lead("stime_ts").over(w))
        )

        is_pingpong = (
            F.col("_prev_cid").isNotNull()
            & F.col("_next_cid").isNotNull()
            & (F.col("_prev_cid") == F.col("_next_cid"))
            & (F.col("_prev_cid") != F.col("cid"))
            & (
                (F.col("_next_ts").cast("long") - F.col("_prev_ts").cast("long"))
                < F.lit(self.PINGPONG_TIME_THRESHOLD)
            )
        )

        avg_lat = (F.col("_prev_lat") + F.col("_next_lat")) / F.lit(2.0)
        avg_lon = (F.col("_prev_lon") + F.col("_next_lon")) / F.lit(2.0)
        avg_ts = (
            F.col("_prev_ts").cast("long") + F.col("_next_ts").cast("long")
        ) / F.lit(2.0)

        return (
            with_neighbors
            .withColumn("lat_d", F.when(is_pingpong, avg_lat).otherwise(F.col("lat_d")))
            .withColumn("lon_d", F.when(is_pingpong, avg_lon).otherwise(F.col("lon_d")))
            .withColumn(
                "stime_ts",
                F.when(is_pingpong, F.from_unixtime(avg_ts).cast("timestamp"))
                .otherwise(F.col("stime_ts")),
            )
            .drop(
                "_prev_cid", "_next_cid",
                "_prev_lat", "_next_lat",
                "_prev_lon", "_next_lon",
                "_prev_ts", "_next_ts",
                "time_value", "dist_value",
            )
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

    def _build_multicity_detail_df(self, src_table, filter_multicity_only=True):
        df = self.__session.table(src_table)
        columns = set(df.columns)

        # uid column: support "uid" or "user_id"
        uid_col = self._pick_first_existing(columns, ["uid", "user_id"])
        if uid_col is None:
            raise ValueError(f"Table {src_table} does not have uid/user_id column")

        # add optional columns if missing
        df = self._add_missing_optional_columns(df)

        required_cols = ["index", "stime", "lat", "lon", "city"]
        missing = [c for c in required_cols if c not in set(df.columns)]
        if missing:
            raise ValueError(f"Table {src_table} missing required columns: {', '.join(missing)}")

        base_df = (
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

        if filter_multicity_only:
            uid_city_df = (
                base_df.where(F.col("city").isNotNull())
                .groupBy("uid")
                .agg(F.countDistinct("city").alias("city_cnt"))
                .where(F.col("city_cnt") > 1)
                .select("uid")
            )
            detail_df = base_df.join(uid_city_df, on="uid", how="inner")
        else:
            detail_df = base_df

        # Step 1: drop duplicates
        dedup_df = detail_df.dropDuplicates()

        # Step 2: merge consecutive same-coordinate records
        merged_df = self._merge_same_coord(dedup_df)

        # Step 3: handle drift (overspeed removal)
        with_dist = self._add_time_dist_columns(merged_df)
        drift_removed = self._remove_drift(with_dist)

        # Step 4: handle ping-pong (oscillation between two base stations)
        with_dist2 = self._add_time_dist_columns(drift_removed)
        pingpong_fixed = self._fix_pingpong(with_dist2)

        # Final time/dist calculation
        result_df = self._add_time_dist_columns(pingpong_fixed)

        # Re-number index sequentially from 0 per uid
        idx_window = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))
        result_df = result_df.withColumn("index_i", F.row_number().over(idx_window) - F.lit(1))

        return result_df.select(
            "uid",
            F.col("index_i").alias("index"),
            F.date_format("stime_ts", "yyyy-MM-dd HH:mm:ss").alias("stime"),
            "cid",
            F.col("lat_d").alias("lat"),
            F.col("lon_d").alias("lon"),
            "city",
            "province",
            F.coalesce(F.col("time_value"), F.lit(0.0)).alias("time_value"),
            F.coalesce(F.col("dist_value"), F.lit(0.0)).alias("dist_value"),
        )

    def _build_uid_metric_df(self, multicity_table):
        df = self.__session.table(multicity_table)
        uid_metric_df = (
            df.where(F.col("uid").isNotNull())
            .groupBy("uid")
            .agg(
                F.sum(F.coalesce(F.col("time_value"), F.lit(0.0))).alias("time"),
                F.sum(F.coalesce(F.col("dist_value"), F.lit(0.0))).alias("distance"),
                F.count(F.lit(1)).alias("cnt"),
            )
        )
        return uid_metric_df

    def _calc_single_table_rows(self, multicity_table):
        uid_metric_df = self._build_uid_metric_df(multicity_table)
        table_date = self._table_date(multicity_table)

        if uid_metric_df.limit(1).count() == 0:
            return [
                (table_date, "time", None, None, None, None),
                (table_date, "distance", None, None, None, None),
                (table_date, "count", 0.0, 0.0, 0.0, 0.0),
            ]

        stats_row = (
            uid_metric_df
            .agg(
                F.max("time").alias("time_max"),
                F.min("time").alias("time_min"),
                F.avg("time").alias("time_avg"),
                F.expr("percentile_approx(time, 0.5)").alias("time_median"),
                F.max("distance").alias("distance_max"),
                F.min("distance").alias("distance_min"),
                F.avg("distance").alias("distance_avg"),
                F.expr("percentile_approx(distance, 0.5)").alias("distance_median"),
                F.max("cnt").alias("count_max"),
                F.min("cnt").alias("count_min"),
                F.avg("cnt").alias("count_avg"),
                F.expr("percentile_approx(cnt, 0.5)").alias("count_median"),
            )
            .collect()[0]
        )

        rows = [
            (
                table_date,
                "time",
                float(stats_row["time_max"]) if stats_row["time_max"] is not None else None,
                float(stats_row["time_min"]) if stats_row["time_min"] is not None else None,
                float(stats_row["time_avg"]) if stats_row["time_avg"] is not None else None,
                float(stats_row["time_median"]) if stats_row["time_median"] is not None else None,
            ),
            (
                table_date,
                "distance",
                float(stats_row["distance_max"]) if stats_row["distance_max"] is not None else None,
                float(stats_row["distance_min"]) if stats_row["distance_min"] is not None else None,
                float(stats_row["distance_avg"]) if stats_row["distance_avg"] is not None else None,
                float(stats_row["distance_median"]) if stats_row["distance_median"] is not None else None,
            ),
            (
                table_date,
                "count",
                float(stats_row["count_max"]) if stats_row["count_max"] is not None else None,
                float(stats_row["count_min"]) if stats_row["count_min"] is not None else None,
                float(stats_row["count_avg"]) if stats_row["count_avg"] is not None else None,
                float(stats_row["count_median"]) if stats_row["count_median"] is not None else None,
            ),
        ]
        return rows

    def _build_single_day_multicity_table(
        self,
        date_str,
        src_prefix="dataset",
        out_prefix="dataset_multicity",
        filter_multicity_only=True,
    ):
        src_table = self._resolve_src_table(date_str=date_str, src_prefix=src_prefix)
        out_table = f"{out_prefix}_{date_str}"
        detail_df = self._build_multicity_detail_df(
            src_table, filter_multicity_only=filter_multicity_only
        )
        detail_df.write.mode("overwrite").saveAsTable(out_table)
        print(f"Saved table: {out_table} rows={detail_df.count()}, from: {src_table}")
        return out_table

    def run_14days_stats(
        self,
        date_list=None,
        src_prefix="dataset",
        multicity_prefix="dataset_multicity",
        out_table="dataset_multicity_14days_stats",
        filter_multicity_only=True,
    ):
        if date_list is None:
            date_list = DEFAULT_DATES

        all_rows = []
        multicity_tables = []
        failed_dates = []
        for date_str in date_list:
            print(f"[INFO] Start processing date: {date_str}")
            try:
                multicity_table = self._build_single_day_multicity_table(
                    date_str=date_str,
                    src_prefix=src_prefix,
                    out_prefix=multicity_prefix,
                    filter_multicity_only=filter_multicity_only,
                )
                multicity_tables.append(multicity_table)
            except Exception as exc:
                failed_dates.append((date_str, str(exc)))
                print(f"[WARN] Skip date {date_str}: {exc}")
                print(traceback.format_exc())

        if not multicity_tables:
            raise RuntimeError(
                "No daily tables were generated successfully. "
                "Please check failed-date logs above."
            )

        for table_name in multicity_tables:
            rows = self._calc_single_table_rows(table_name)
            all_rows.extend(rows)
            print(f"Finished stats for table: {table_name}")

        schema = StructType([
            StructField("stat_date", StringType(), False),
            StructField("metric", StringType(), False),
            StructField("max_value", DoubleType(), True),
            StructField("min_value", DoubleType(), True),
            StructField("avg_value", DoubleType(), True),
            StructField("median_value", DoubleType(), True),
        ])

        result_df = self.__session.createDataFrame(all_rows, schema=schema)

        metric_order = F.when(F.col("metric") == "time", F.lit(1)) \
            .when(F.col("metric") == "distance", F.lit(2)) \
            .when(F.col("metric") == "count", F.lit(3)) \
            .otherwise(F.lit(99))

        result_df = result_df.orderBy(F.col("stat_date"), metric_order)
        result_df.write.mode("overwrite").saveAsTable(out_table)

        print(f"Saved table: {out_table}, rows: {result_df.count()}")
        if failed_dates:
            print("[WARN] Failed dates summary:")
            for date_str, reason in failed_dates:
                print(f"  - {date_str}: {reason}")


if __name__ == "__main__":
    # Set local=True for local Hive testing; local=False (default) for YARN cluster
    table = HiveTable(db="ss_seu_df", local=True)
    try:
        table.run_14days_stats(
            date_list=DEFAULT_DATES,
            src_prefix="dataset",
            multicity_prefix="dataset_multicity",
            out_table="dataset_multicity_14days_stats",
            filter_multicity_only=True,
        )
    finally:
        table.stop()
# spark-submit: --master yarn --deploy-mode cluster