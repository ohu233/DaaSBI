from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window
import os


DB_NAME = "ss_seu_df"
SRC_TABLE = "dataset_20230917"
OUT_TABLE = "dataset_multicity_20230917"
STATS_TABLE = "dataset_multicity_14days_stats"
STAT_DATE = "20230917"

# 当前你只有一个表，且样例里可能不是跨城市数据。
# 先设 False，保证能产出结果。
# 后面如果要严格筛选跨城市用户，改成 True。
FILTER_MULTICITY_ONLY = False

MAX_SPEED_MPS = 83.33          # 300 km/h
PINGPONG_TIME_THRESHOLD = 300  # seconds


def build_spark():
    warehouse = f"file://{os.path.expanduser('~/hive/warehouse')}"

    spark = (
        SparkSession.builder
        .appName("preprocess_multicity_one_table")
        .master("local[*]")
        .config("spark.hadoop.hive.metastore.uris", "thrift://localhost:9083")
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.sql.hive.metastore.version", "4.1.0")
        .config("spark.sql.hive.metastore.jars", "maven")
        .config("spark.sql.ansi.enabled", "false")
        .enableHiveSupport()
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def parse_time_col(col_name="stime"):
    s = F.trim(F.col(col_name).cast("string"))

    ts_seconds = F.coalesce(
        F.unix_timestamp(s, "yyyy/M/d H:mm"),
        F.unix_timestamp(s, "yyyy/M/d HH:mm"),
        F.unix_timestamp(s, "yyyy/M/d H:mm:ss"),
        F.unix_timestamp(s, "yyyy/M/d HH:mm:ss"),
        F.unix_timestamp(s, "yyyy-MM-dd HH:mm:ss"),
        F.unix_timestamp(s, "yyyy-MM-dd HH:mm"),
        F.unix_timestamp(s, "yyyy/MM/dd HH:mm:ss"),
        F.unix_timestamp(s, "yyyy/MM/dd HH:mm"),
    )

    return F.from_unixtime(ts_seconds).cast("timestamp")


def add_missing_optional_columns(df):
    cols = set(df.columns)

    if "cid" not in cols:
        df = df.withColumn("cid", F.lit(None).cast("string"))

    if "province" not in cols:
        df = df.withColumn("province", F.lit(None).cast("string"))

    return df


def add_time_dist_columns(df):
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
    dist = F.lit(6371000.0) * c

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
            .otherwise(F.coalesce(dist.cast("double"), F.lit(0.0))),
        )
        .drop("prev_stime_ts", "prev_lat", "prev_lon")
    )


def merge_same_coord(df):
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


def remove_drift(df):
    with_dist = add_time_dist_columns(df)

    is_overspeed = (
        (F.col("time_value") > 0)
        & (F.col("dist_value") > 0)
        & ((F.col("dist_value") / F.col("time_value")) > F.lit(MAX_SPEED_MPS))
    )

    return with_dist.where(~is_overspeed).drop("time_value", "dist_value")


def fix_pingpong(df):
    w = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))

    d = (
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
            < F.lit(PINGPONG_TIME_THRESHOLD)
        )
    )

    avg_lat = (F.col("_prev_lat") + F.col("_next_lat")) / F.lit(2.0)
    avg_lon = (F.col("_prev_lon") + F.col("_next_lon")) / F.lit(2.0)
    avg_ts = (
        F.col("_prev_ts").cast("long") + F.col("_next_ts").cast("long")
    ) / F.lit(2.0)

    return (
        d
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
        )
    )


def build_detail_df(spark):
    src = f"{DB_NAME}.{SRC_TABLE}"
    df = spark.table(src)

    print(f"[INFO] Source table: {src}")
    print("[INFO] Source schema:")
    df.printSchema()

    df = add_missing_optional_columns(df)

    required = {"uid", "index", "stime", "lat", "lon", "city"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"源表缺少必要字段: {sorted(missing)}")

    base = (
        df
        .where(
            F.col("uid").isNotNull()
            & F.col("index").isNotNull()
            & F.col("stime").isNotNull()
            & F.col("lat").isNotNull()
            & F.col("lon").isNotNull()
        )
        .withColumn("uid", F.col("uid").cast("string"))
        .withColumn("index_i", F.col("index").cast("long"))
        .withColumn("stime_ts", parse_time_col("stime"))
        .withColumn("lat_d", F.col("lat").cast("double"))
        .withColumn("lon_d", F.col("lon").cast("double"))
        .where(
            F.col("index_i").isNotNull()
            & F.col("stime_ts").isNotNull()
            & F.col("lat_d").isNotNull()
            & F.col("lon_d").isNotNull()
        )
    )

    if FILTER_MULTICITY_ONLY:
        multicity_uids = (
            base.where(F.col("city").isNotNull())
            .groupBy("uid")
            .agg(F.countDistinct("city").alias("city_cnt"))
            .where(F.col("city_cnt") > 1)
            .select("uid")
        )
        base = base.join(multicity_uids, on="uid", how="inner")

    dedup = base.dropDuplicates()

    merged = merge_same_coord(dedup)
    drift_removed = remove_drift(merged)
    pingpong_fixed = fix_pingpong(drift_removed)
    result = add_time_dist_columns(pingpong_fixed)

    idx_window = Window.partitionBy("uid").orderBy(F.col("index_i"), F.col("stime_ts"))
    result = result.withColumn("index_i", F.row_number().over(idx_window) - F.lit(1))

    return result.select(
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


def build_stats_df(spark, detail_df):
    uid_metric = (
        detail_df.where(F.col("uid").isNotNull())
        .groupBy("uid")
        .agg(
            F.sum(F.coalesce(F.col("time_value"), F.lit(0.0))).alias("time"),
            F.sum(F.coalesce(F.col("dist_value"), F.lit(0.0))).alias("distance"),
            F.count(F.lit(1)).alias("cnt"),
        )
    )

    if uid_metric.limit(1).count() == 0:
        rows = [
            (STAT_DATE, "time", None, None, None, None),
            (STAT_DATE, "distance", None, None, None, None),
            (STAT_DATE, "count", 0.0, 0.0, 0.0, 0.0),
        ]
    else:
        r = (
            uid_metric
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
                STAT_DATE,
                "time",
                float(r["time_max"]) if r["time_max"] is not None else None,
                float(r["time_min"]) if r["time_min"] is not None else None,
                float(r["time_avg"]) if r["time_avg"] is not None else None,
                float(r["time_median"]) if r["time_median"] is not None else None,
            ),
            (
                STAT_DATE,
                "distance",
                float(r["distance_max"]) if r["distance_max"] is not None else None,
                float(r["distance_min"]) if r["distance_min"] is not None else None,
                float(r["distance_avg"]) if r["distance_avg"] is not None else None,
                float(r["distance_median"]) if r["distance_median"] is not None else None,
            ),
            (
                STAT_DATE,
                "count",
                float(r["count_max"]) if r["count_max"] is not None else None,
                float(r["count_min"]) if r["count_min"] is not None else None,
                float(r["count_avg"]) if r["count_avg"] is not None else None,
                float(r["count_median"]) if r["count_median"] is not None else None,
            ),
        ]

    schema = StructType([
        StructField("stat_date", StringType(), False),
        StructField("metric", StringType(), False),
        StructField("max_value", DoubleType(), True),
        StructField("min_value", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("median_value", DoubleType(), True),
    ])

    return spark.createDataFrame(rows, schema=schema)


def main():
    spark = build_spark()

    try:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        spark.sql(f"USE {DB_NAME}")

        detail_df = build_detail_df(spark)

        print("[INFO] Detail preview:")
        detail_df.show(10, truncate=False)

        detail_count = detail_df.count()
        print(f"[INFO] Detail rows: {detail_count}")

        detail_df.write.mode("overwrite").saveAsTable(OUT_TABLE)
        print(f"[OK] Saved table: {DB_NAME}.{OUT_TABLE}")

        stats_df = build_stats_df(spark, detail_df)
        stats_df.show(truncate=False)

        stats_df.write.mode("overwrite").saveAsTable(STATS_TABLE)
        print(f"[OK] Saved table: {DB_NAME}.{STATS_TABLE}")

        print("[INFO] Hive tables:")
        spark.sql(f"SHOW TABLES IN {DB_NAME}").show(truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()