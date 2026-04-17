import re
from typing import List

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


class HiveTable:
    def __init__(self, db: str = "ss_seu_df"):
        session = (
            SparkSession.builder
            .enableHiveSupport()
            .getOrCreate()
        )
        session.sql(f"USE {db}")
        self.__session = session
        self.db = db

    def stop(self):
        self.__session.stop()

    def list_dataset_tables(self, table_prefix: str = "dataset", latest_n: int = 14) -> List[str]:
        """
        从 Hive 中获取形如 dataset_YYYYMMDD 的表，按日期倒序取 latest_n 张表。
        """
        pattern = re.compile(rf"^{re.escape(table_prefix)}_(\d{{8}})$")

        raw_tables = (
            self.__session
            .sql(f"SHOW TABLES IN {self.db} LIKE '{table_prefix}_*'")
            .select("tableName")
            .collect()
        )

        table_date_pairs = []
        for row in raw_tables:
            table_name = row["tableName"]
            m = pattern.match(table_name)
            if m:
                table_date_pairs.append((table_name, m.group(1)))

        table_date_pairs.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in table_date_pairs[:latest_n]]

    def filter_uid_with_multi_city(self, src_table: str):
        """
        仅保留 src_table 中 city 去重数大于 1 的 uid 的全部记录。
        """
        df = self.__session.table(src_table)

        uid_multi_city = (
            df.groupBy("uid")
            .agg(F.countDistinct("city").alias("city_cnt"))
            .where(F.col("city_cnt") > 1)
            .select("uid")
        )

        return df.join(uid_multi_city, on="uid", how="inner")

    def run(
        self,
        table_prefix: str = "dataset",
        out_prefix: str = "dataset_multicity",
        latest_n: int = 14,
        explicit_tables: List[str] = None,
    ):
        tables = explicit_tables if explicit_tables else self.list_dataset_tables(table_prefix=table_prefix, latest_n=latest_n)

        if not tables:
            print(f"未找到可处理的表（前缀: {table_prefix}_YYYYMMDD）。")
            return

        print(f"待处理表数量: {len(tables)}")
        for t in tables:
            out_table = t.replace(f"{table_prefix}_", f"{out_prefix}_", 1)
            result_df = self.filter_uid_with_multi_city(t)
            result_df.write.mode("overwrite").saveAsTable(out_table)

            row_cnt = result_df.count()
            uid_cnt = result_df.select("uid").distinct().count()
            print(f"[{t}] -> [{out_table}] 完成，记录数: {row_cnt}，uid数: {uid_cnt}")


if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run(
            table_prefix="dataset",
            out_prefix="dataset_multicity",
            latest_n=14,
            explicit_tables=None,
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster