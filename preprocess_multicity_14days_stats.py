from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window


DEFAULT_TABLES = [
    "dataset_multicity_20230917",
    "dataset_multicity_20230918",
    "dataset_multicity_20230919",
    "dataset_multicity_20230920",
    "dataset_multicity_20230921",
    "dataset_multicity_20230922",
    "dataset_multicity_20230923",
    "dataset_multicity_20250914",
    "dataset_multicity_20250915",
    "dataset_multicity_20250916",
    "dataset_multicity_20250917",
    "dataset_multicity_20250918",
    "dataset_multicity_20250919",
    "dataset_multicity_20250920",
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

    def _build_uid_metric_df(self, src_table):
        df = self.__session.table(src_table)
        columns = set(df.columns)

        uid_col = self._pick_first_existing(columns, ["uid", "user_id"])

        if uid_col is None:
            raise ValueError(f"Table {src_table} does not have uid/user_id column")

        required_cols = ["stime", "lat", "lon"]
        missing = [col_name for col_name in required_cols if col_name not in columns]
        if missing:
            raise ValueError(f"Table {src_table} missing required columns: {', '.join(missing)}")

        partition_cols = [uid_col]
        if "date" in columns:
            partition_cols.append("date")

        w = Window.partitionBy(*partition_cols).orderBy(F.col("stime"))

        enriched_df = (
            df.where(F.col(uid_col).isNotNull())
            .withColumn("_next_stime", F.lead("stime").over(w))
            .withColumn("_next_lat", F.lead("lat").over(w))
            .withColumn("_next_lon", F.lead("lon").over(w))
        )

        time_diff_expr = (
            F.unix_timestamp(F.col("_next_stime"))
            - F.unix_timestamp(F.col("stime"))
        ).cast("double")

        lat1 = F.radians(F.col("lat").cast("double"))
        lon1 = F.radians(F.col("lon").cast("double"))
        lat2 = F.radians(F.col("_next_lat").cast("double"))
        lon2 = F.radians(F.col("_next_lon").cast("double"))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            F.pow(F.sin(dlat / F.lit(2.0)), F.lit(2.0))
            + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / F.lit(2.0)), F.lit(2.0))
        )
        a_clamped = F.least(F.greatest(a, F.lit(0.0)), F.lit(1.0))
        c = F.lit(2.0) * F.atan2(F.sqrt(a_clamped), F.sqrt(F.lit(1.0) - a_clamped))
        haversine_dist = F.lit(6371000.0) * c

        metric_df = (
            enriched_df
            .withColumn(
                "_time_value",
                F.when(F.col("_next_stime").isNull(), F.lit(0.0))
                .otherwise(F.greatest(time_diff_expr, F.lit(0.0))),
            )
            .withColumn(
                "_dist_value",
                F.when(
                    F.col("_next_lat").isNull() | F.col("_next_lon").isNull(),
                    F.lit(0.0),
                ).otherwise(F.coalesce(haversine_dist.cast("double"), F.lit(0.0))),
            )
        )

        uid_metric_df = (
            metric_df
            .groupBy(F.col(uid_col).alias("uid"))
            .agg(
                F.sum(F.col("_time_value")).alias("time"),
                F.sum(F.col("_dist_value")).alias("distance"),
                F.count(F.lit(1)).alias("cnt"),
            )
        )
        return uid_metric_df

    def _calc_single_table_rows(self, src_table):
        uid_metric_df = self._build_uid_metric_df(src_table)
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

        table_date = self._table_date(src_table)
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

    def run_14days_stats(self, table_names=None, out_table="dataset_multicity_14days_stats"):
        if table_names is None:
            table_names = DEFAULT_TABLES

        all_rows = []
        for table_name in table_names:
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


if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_14days_stats(
            table_names=DEFAULT_TABLES,
            out_table="dataset_multicity_14days_stats",
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster
