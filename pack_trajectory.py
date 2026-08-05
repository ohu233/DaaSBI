'''
输入：dataset_multicity_YYYYMMDD_processed（YYYYMMDD为日期，20230917-20230923和20250914-20250920），每行一个轨迹点，包含以下列：
- uid: 用户唯一id
- index: 轨迹点在用户轨迹中的索引
- idx: 处理后重新编号的索引（可选）
- stime: 轨迹点的时间戳
- cid: 基站编号
- lat: 轨迹点的纬度
- lon: 轨迹点的经度
- city: 轨迹点所属城市
- province: 轨迹点所属省份
- time_value: 与上一轨迹点的时间差（单位：秒）
- dist_value: 与上一轨迹点的空间距离（单位：米）
- velocity: 速度（单位：米/秒）
- attribution: 处理标记（origin/drift/pingpong/merge）

对dataset_multicity_YYYYMMDD_processed进行处理：
1. 按uid分组，组内按 (stime, index, idx) 中存在的列升序排序
2. 对每个字段（除uid外）使用 collect_list 窗口函数聚合为数组，保留组内顺序
3. 每个uid只保留一行（取组内第一行的数组）

输出表 dataset_multicity_YYYYMMDD_packed，每个uid一行，单元格为按轨迹顺序的数组（array类型），包含以下列：
- uid: 用户唯一id
- sort_key: array<bigint>，每个元素是该位置轨迹点在 uid 内按 (stime, index, idx) 排序后的序号（从1开始），用于解码端校验对齐 / 重排还原
- index: array<long>，按顺序的轨迹点索引
- stime: array<...>，按顺序的时间戳
- cid: array<...>
- lat: array<double>
- lon: array<double>
- city: array<...>
- province: array<...>
- time_value: array<...>
- dist_value: array<...>
- velocity: array<...>
- attribution: array<...>
- ...（源表的其他列保持原名，类型变为对应array类型）

各 array 列（含 sort_key）等长、同位对应——派生自同一个 struct 数组，sort_key[i] 标识 lat[i]/lon[i]/stime[i]... 来自 uid 内排序后的第 sort_key[i] 行。

最后生成表：14张
- dataset_multicity_YYYYMMDD_packed（14张）
'''

from pyspark.sql import SparkSession
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
    # 组内排序候选列，按优先级回退
    SORT_CANDIDATES = ("stime", "index", "idx")
    # 轨迹点数下限：uid 内轨迹点数少于此值的整条丢弃
    MIN_TRAJ_POINTS = 30

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

    def _table_exists(self, table_name):
        # Strict generic check: query metastore tables once and do exact-name match.
        table_names = {
            row["tableName"]
            for row in self.__session.sql("SHOW TABLES").select("tableName").collect()
        }
        return table_name in table_names

    def _resolve_src_table(self, date_str, src_prefix="dataset_multicity"):
        # 候选：优先 _processed（pingpongdrift 输出），回退到未处理的 multicity 表
        candidates = [
            f"{src_prefix}_{date_str}_processed",
            f"{src_prefix}_{date_str}",
        ]
        for table_name in candidates:
            if self._table_exists(table_name):
                return table_name
        raise ValueError(
            f"Cannot find source table for date {date_str}. Tried: {', '.join(candidates)}"
        )

    def _columns_of(self, table_name):
        return list(self.__session.table(table_name).columns)

    def _pick_sort_cols(self, columns):
        """从源表列里挑组内排序键：(stime, index, idx) 中存在的列；都没有则回退到全部非 uid 列。"""
        sort_cols = [c for c in self.SORT_CANDIDATES if c in columns]
        return sort_cols or [c for c in columns if c != "uid"]

    def _build_packed_sql(self, src_table, uid_col="uid", min_points=None):
        """构造并执行一条 SQL：按 uid 把每个字段聚合成按顺序的 array，每个 uid 一行。

        对齐保证：先把整行打包成 named_struct（含一个排序 key 字段 _f_sk），再用
        collect_list 收集 struct 数组，最后用 transform(x -> x._fN) 把每个字段从同一个
        struct 数组里提取出来。所有字段数组（含 sort_key）都派生自同一个 struct 数组，
        长度相同、位置严格一一对应；struct 内 NULL 字段保留、整个 struct 元素不会被丢弃。

        sort_key：base 表里用 ROW_NUMBER() OVER (PARTITION BY uid ORDER BY sort_cols)
        预先算好的 uid 内序号，跟随 struct 进 collect_list。解码端可据此校验对齐，
        或在 collect_list 顺序有疑问时按 sort_key 重排还原原始行序。

        min_points：uid 内轨迹点数下限，少于此数的 uid 整条丢弃（默认 MIN_TRAJ_POINTS）。

        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING 让窗口覆盖整个 uid，
        保证每个元素都拿到完整的有序数组；最外层 ROW_NUMBER() = 1 把每 uid 多行去重为一行。
        """
        if min_points is None:
            min_points = self.MIN_TRAJ_POINTS

        columns = self._columns_of(src_table)
        if uid_col not in columns:
            raise ValueError(f"Table {src_table} does not have {uid_col} column")

        value_cols = [c for c in columns if c != uid_col]
        if not value_cols:
            raise ValueError(f"Table {src_table} has no value columns to pack")

        sort_cols = self._pick_sort_cols(columns)
        sort_sql = ", ".join(f"`{c}`" for c in sort_cols)

        # named_struct 字段用安全别名 _f0, _f1, ... 避免 index/time 等保留字冲突；
        # 额外注入 _f_sk = 该行在 uid 内的排序序号，作为每个值的 sort key 供解码端对齐
        struct_args = ", ".join(
            f"'_f{i}', `{c}`" for i, c in enumerate(value_cols)
        ) + ", '_f_sk', `_sk`"
        # 从 struct 数组提取每个字段；所有 transform 作用于同一个 _traj 数组，保证对齐
        extract_exprs = ",\n            ".join(
            f"transform(`_traj`, x -> x._f{i}) AS `{c}`"
            for i, c in enumerate(value_cols)
        )
        # sort_key 单独提取，置于 uid 之后，解码端先读到
        sort_key_expr = f"transform(`_traj`, x -> x._f_sk) AS `sort_key`"

        sql = f"""
        WITH base AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY `{uid_col}` ORDER BY {sort_sql}
                   ) AS `_sk`,
                   COUNT(*) OVER (
                       PARTITION BY `{uid_col}`
                   ) AS `_cnt`
            FROM {src_table}
            WHERE `{uid_col}` IS NOT NULL
        ),
        filtered AS (
            -- 丢弃轨迹点数 < min_points 的 uid（整条）
            SELECT *
            FROM base
            WHERE `_cnt` >= {int(min_points)}
        ),
        with_traj AS (
            SELECT
                `{uid_col}`,
                collect_list(named_struct({struct_args})) OVER (
                    PARTITION BY `{uid_col}` ORDER BY `_sk`
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS `_traj`,
                ROW_NUMBER() OVER (
                    PARTITION BY `{uid_col}` ORDER BY `_sk`
                ) AS _rn
            FROM filtered
        )
        SELECT
            `{uid_col}`,
            {sort_key_expr},
            {extract_exprs}
        FROM with_traj
        WHERE _rn = 1
        """
        return self.__session.sql(sql)

    def _build_single_day_packed_table(
        self,
        date_str,
        src_prefix="dataset_multicity",
        out_prefix="dataset_multicity",
        out_suffix="_packed",
        min_points=None,
    ):
        src_table = self._resolve_src_table(date_str=date_str, src_prefix=src_prefix)
        out_table = f"{out_prefix}_{date_str}{out_suffix}"
        packed_df = self._build_packed_sql(src_table, min_points=min_points)
        packed_df.write.mode("overwrite").saveAsTable(out_table)
        print(f"Saved table: {out_table} rows={packed_df.count()}, from: {src_table}")
        return out_table

    def run_pack_all(
        self,
        date_list=None,
        src_prefix="dataset_multicity",
        out_prefix="dataset_multicity",
        out_suffix="_packed",
        min_points=None,
    ):
        if date_list is None:
            date_list = DEFAULT_DATES

        failed_dates = []
        for date_str in date_list:
            print(f"[INFO] Start packing date: {date_str}")
            try:
                self._build_single_day_packed_table(
                    date_str=date_str,
                    src_prefix=src_prefix,
                    out_prefix=out_prefix,
                    out_suffix=out_suffix,
                    min_points=min_points,
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
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_pack_all(
            date_list=DEFAULT_DATES,
            src_prefix="dataset_multicity",
            out_prefix="dataset_multicity",
            out_suffix="_packed",
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster
