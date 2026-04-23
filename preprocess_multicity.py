'''
输入：dataset__YYYYMMDD（YYYYMMDD为日期，20230917-20230923和20250914-20250920），包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引，从0开始
- stime: 轨迹点的时间戳
- cid: 基站编号
- lat: 轨迹点的纬度
- lon: 轨迹点的经度
- city: 轨迹点所属城市
- province: 轨迹点所属省份
- date: 轨迹点所属日期，格式为YYYYMMDD

对dataset进行处理（dataset为原始数据）：
1. 筛选跨城市用户（uid）及其轨迹记录
2. 对筛选后的数据进行排序，按照uid分组，组内按照stime排序
3. 计算相邻轨迹点之间的时间差和空间距离（使用haversine公式计算地理距离），保存在后一行的time_value和dist_value列中（每个uid第一行无差分）
TODO:漂移数据，乒乓数据处理：参考An adaptive staying point recognition algorithm based on spatiotemporal characteristics using cellular signaling data
处理方法：
    · 删除/融合重复记录
    · 漂移数据：超速删除
    · 乒乓数据：时间坐标取平均
4. 输出表格，形式为dataset_multicity_YYYYMMDD，包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引，从0开始
- stime: 轨迹点的时间戳
- cid: 基站编号
- lat: 轨迹点的纬度
- lon: 轨迹点的经度
- city: 轨迹点所属城市
- province: 轨迹点所属省份
- time_value: 与下一轨迹点的时间差（单位：秒），如果没有下一点则为0
- dist_value: 与下一轨迹点的空间距离（单位：米），如果没有下一点则为0
5. 对每个用户的time_value和dist_value进行统计，计算最大值、最小值、平均值和中位数，输出表格dataset_multicity_14days_stats，包含以下列：
- stat_date: 统计日期，格式为YYYYMMDD
- metric: 统计指标，取值为"time"，"distance"或"count"，分别表示时间差、空间距离和轨迹点数量
- max_value: 统计指标的最大值
- min_value: 统计指标的最小值
- avg_value: 统计指标的平均值
- median_value: 统计指标的中位数

最后生成表有：14+1=15张，分别为：
- dataset_multicity_YYYYMMDD（14张）
- dataset_multicity_14days_stats（1张）
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window
import traceback


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
    def __init__(self, db="ss_seu_df"):
        session = (
            SparkSession.builder
            .enableHiveSupport()
            .getOrCreate()
        )
        session.sql(f"USE {db}")
        self.__session = session

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
        # Strict generic check: query metastore tables once and do exact-name match.
        table_names = {
            row["tableName"]
            for row in self.__session.sql("SHOW TABLES").select("tableName").collect()
        }
        return table_name in table_names

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

        required_cols = ["index", "stime", "cid", "lat", "lon", "city", "province"]
        missing = [col_name for col_name in required_cols if col_name not in columns]
        if missing:
            raise ValueError(f"Table {src_table} missing required columns: {', '.join(missing)}")

        base_df = (
            df.where(
                F.col(uid_col).isNotNull()
                & F.col("index").isNotNull()
                & F.col("stime").isNotNull()
            )
            .withColumn("index_i", F.col("index").cast("long"))
            .withColumn("lat_d", F.col("lat").cast("double"))
            .withColumn("lon_d", F.col("lon").cast("double"))
            .where(F.col("index_i").isNotNull())
        )

        uid_city_df = (
            base_df.where(F.col("city").isNotNull())
            .groupBy(F.col(uid_col).alias("uid"))
            .agg(F.countDistinct("city").alias("city_cnt"))
            .where(F.col("city_cnt") > 1)
            .select("uid")
        )

        detail_df = (
            base_df
            .join(uid_city_df, base_df[uid_col] == uid_city_df["uid"], how="inner")
            .drop(uid_city_df["uid"])
            .withColumn("uid", F.col(uid_col).cast("string"))
        )

        w = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime"))

        with_next_df = (
            detail_df
            .withColumn("prev_stime", F.lag("stime").over(w))
            .withColumn("prev_lat", F.lag("lat_d").over(w))
            .withColumn("prev_lon", F.lag("lon_d").over(w))
        )

        time_diff_expr = (
            F.unix_timestamp(F.col("stime"))
            - F.unix_timestamp(F.col("prev_stime"))
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

        metric_df = (
            with_next_df
            .withColumn(
                "time_value",
                F.when(F.col("prev_stime").isNull(), F.lit(None).cast("double"))
                .otherwise(F.greatest(time_diff_expr, F.lit(0.0))),
            )
            .withColumn(
                "dist_value",
                F.when(
                    F.col("prev_stime").isNull(),
                    F.lit(None).cast("double"),
                ).otherwise(F.coalesce(haversine_dist.cast("double"), F.lit(0.0))),
            )
        )

        return metric_df.select(
            "uid",
            F.col("index_i").alias("index"),
            "stime",
            "cid",
            F.col("lat_d").alias("lat"),
            F.col("lon_d").alias("lon"),
            "city",
            "province",
            "time_value",
            "dist_value",
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

        table_date = self._table_date(multicity_table)
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
    ):
        src_table = self._resolve_src_table(date_str=date_str, src_prefix=src_prefix)
        out_table = f"{out_prefix}_{date_str}"
        detail_df = self._build_multicity_detail_df(src_table)
        detail_df.write.mode("overwrite").saveAsTable(out_table)
        print(f"Saved table: {out_table} rows={detail_df.count()}, from: {src_table}")
        return out_table

    def run_14days_stats(
        self,
        date_list=None,
        src_prefix="dataset",
        multicity_prefix="dataset_multicity",
        out_table="dataset_multicity_14days_stats",
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
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_14days_stats(
            date_list=DEFAULT_DATES,
            src_prefix="dataset",
            multicity_prefix="dataset_multicity",
            out_table="dataset_multicity_14days_stats",
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster
