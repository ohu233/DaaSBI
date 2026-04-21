from pyspark.sql import SparkSession
import pyspark.sql.functions as F


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

    def filter_uid_multi_city(self, src_table="total_time_space_diff"):
        """
        Keep only records whose uid appears in more than one distinct city.

        Returns:
            detail_df: full records for qualifying uids
            uid_city_df: uid-level city distinct count (>1)
        """
        df = self.__session.table(src_table)

        uid_city_df = (
            df.where(F.col("uid").isNotNull() & F.col("city").isNotNull())
            .groupBy("uid")
            .agg(F.countDistinct("city").alias("city_cnt"))
            .where(F.col("city_cnt") > 1)
        )

        detail_df = df.join(uid_city_df.select("uid"), on="uid", how="inner")
        return detail_df, uid_city_df

    def print_time_space_count_stats(self, detail_df):
        """
        Aggregate by uid and print max/min/avg/median for:
        - time  = sum(time_diff)
        - space = sum(space_diff)
        - count = record count
        """
        uid_metric_df = (
            detail_df
            .groupBy("uid")
            .agg(
                F.sum(F.coalesce(F.col("time_diff"), F.lit(0))).alias("time"),
                F.sum(F.coalesce(F.col("space_diff"), F.lit(0.0))).alias("space"),
                F.count(F.lit(1)).alias("cnt")
            )
        )

        stats_row = (
            uid_metric_df
            .agg(
                F.max("time").alias("time_max"),
                F.min("time").alias("time_min"),
                F.avg("time").alias("time_avg"),
                F.expr("percentile_approx(time, 0.5)").alias("time_median"),
                F.max("space").alias("space_max"),
                F.min("space").alias("space_min"),
                F.avg("space").alias("space_avg"),
                F.expr("percentile_approx(space, 0.5)").alias("space_median"),
                F.max("cnt").alias("count_max"),
                F.min("cnt").alias("count_min"),
                F.avg("cnt").alias("count_avg"),
                F.expr("percentile_approx(cnt, 0.5)").alias("count_median")
            )
            .collect()[0]
        )

        print("===== time/space/count stats (by uid) =====")
        print(
            f"time  -> max: {stats_row['time_max']}, min: {stats_row['time_min']}, "
            f"avg: {stats_row['time_avg']}, median: {stats_row['time_median']}"
        )
        print(
            f"space -> max: {stats_row['space_max']}, min: {stats_row['space_min']}, "
            f"avg: {stats_row['space_avg']}, median: {stats_row['space_median']}"
        )
        print(
            f"count -> max: {stats_row['count_max']}, min: {stats_row['count_min']}, "
            f"avg: {stats_row['count_avg']}, median: {stats_row['count_median']}"
        )

    def run_filter(
        self,
        src_table="total_time_space_diff",
        out_detail="cross_city_travel",
        out_uid="cross_city_travel_uid_count"
    ):
        """
        Save two result tables:
        - out_detail: all rows belonging to uid with city_cnt > 1
        - out_uid: uid and city_cnt summary
        """
        detail_df, uid_city_df = self.filter_uid_multi_city(src_table=src_table)

        detail_df.cache()
        uid_city_df.cache()

        detail_df.write.mode("overwrite").saveAsTable(out_detail)
        uid_city_df.write.mode("overwrite").saveAsTable(out_uid)
        self.print_time_space_count_stats(detail_df)

        detail_cnt = detail_df.count()
        uid_cnt = uid_city_df.count()

        print(f"Saved table: {out_detail}, rows: {detail_cnt}")
        print(f"Saved table: {out_uid}, uid rows: {uid_cnt}")

        detail_df.unpersist()
        uid_city_df.unpersist()


if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_filter(
            src_table="total_time_space_diff",
            out_detail="cross_city_travel",
            out_uid="cross_city_travel_uid_count"
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster
