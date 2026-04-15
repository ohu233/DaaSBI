from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.window import Window
import pandas as pd
from pyproj import CRS, Transformer
import numpy as np
'''
有用的初始字段：uid，stime，cid，province，city，date

处理逻辑：
1. 选取cell_info表的cid，lat，lon，date字段，对数据进行去重：相同cid仅保留date大的那条记录
2. 处理后的cell_info数据与move_vp进行关联，关联条件：move_vp的cid = cell_info的cid，关联后得到每条move_vp记录对应的经纬度信息（date字段不参与关联，只参与1的筛选）
3. 按照uid，date进行分组，并用uid+date作为trajectory_id，分组后每组内部按stime排序，每个uid-date对内部生成index索引
4. 对每组uid-date对内部进行如下计算：
    - 计算当前行与下一行的时间差，单位为秒，命名为time_diff
    - 计算当前行与下一行的空间距离，单位为米，命名为space_diff
    - 计算该uid-date对的起点和终点的出行时间间隔，单位为秒，命名为traj_time
    - 计算该uid-date对的起点和终点的空间距离，单位为米，命名为traj_space
5. 最终生成的字段：uid，index，stime，cid，lat，lon，province，city，date，time_diff，space_diff，traj_time，traj_space，trajectory_id，表命名为total_time_space_diff
6. 额外成表：输出字段为：trajectory_id，time_diff，space_diff，表命名为time_space_diff

'''
# ----------------------------
# 投影与网格参数（当前：三省一市）
# ----------------------------
X_MIN, X_MAX = 853241.740909714, 1678241.740909714
Y_MIN, Y_MAX = 3012170.012633881, 3956170.012633881
GRID_SIZE = 1000  # 1km × 1km

# 坐标系
WGS84_WKT = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
BJ54_3DEG_CM111_WKT = 'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'

def _get_transformer():
    if not hasattr(_get_transformer, "_tfm"):
        wgs84 = CRS.from_wkt(WGS84_WKT)
        bj54 = CRS.from_wkt(BJ54_3DEG_CM111_WKT)
        _get_transformer._tfm = Transformer.from_crs(wgs84, bj54, always_xy=True)
    return _get_transformer._tfm

proj_schema = StructType([
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True)
])

# =========================================================

class HiveTable:
    def __init__(self, db="ss_seu_df"):
        session = (
            SparkSession.builder
            .enableHiveSupport()
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        session.sql(f"USE {db}")
        self.__session = session

        @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
        def proj_x(lon, lat):
            lon_s = pd.to_numeric(lon, errors="coerce")
            lat_s = pd.to_numeric(lat, errors="coerce")
            mask = (lon_s.ge(-180) & lon_s.le(180) & lat_s.ge(-90) & lat_s.le(90))
            out = pd.Series(np.nan, index=lon_s.index, dtype="float64")
            if mask.any():
                tfm = _get_transformer()
                xs, _ys = tfm.transform(lon_s[mask].to_numpy(), lat_s[mask].to_numpy())
                out.loc[mask] = xs
            return out

        @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
        def proj_y(lon, lat):
            lon_s = pd.to_numeric(lon, errors="coerce")
            lat_s = pd.to_numeric(lat, errors="coerce")
            mask = (lon_s.ge(-180) & lon_s.le(180) & lat_s.ge(-90) & lat_s.le(90))
            out = pd.Series(np.nan, index=lat_s.index, dtype="float64")
            if mask.any():
                tfm = _get_transformer()
                _xs, ys = tfm.transform(lon_s[mask].to_numpy(), lat_s[mask].to_numpy())
                out.loc[mask] = ys
            return out

        self.proj_x = proj_x
        self.proj_y = proj_y

    def stop(self):
        self.__session.stop()

    def process_all(self, src_table='move_vp', cell_info_table='cell_info'):
        """
        完整处理逻辑（不切分表）:
        1. cell_info 去重：相同 cid 仅保留 date 最大的记录
        2. move_vp 与去重后的 cell_info 按 cid 关联，获取经纬度
        3. 按 uid, date 分组，trajectory_id = hash(uid, date)
        4. 组内按 stime 排序，生成 index，计算 time_diff, space_diff, traj_time, traj_space
        最终字段：uid, index, stime, cid, lat, lon, province, city, date,
                  time_diff, space_diff, traj_time, traj_space, trajectory_id
        """
        query = f"""
            WITH cell_latest AS (
                SELECT cid, lat, lon
                FROM (
                    SELECT cid, lat, lon,
                           ROW_NUMBER() OVER (PARTITION BY cid ORDER BY date DESC) AS rn
                    FROM {cell_info_table}
                ) t
                WHERE rn = 1
            ),
            joined AS (
                SELECT
                    mv.uid,
                    mv.stime,
                    mv.cid,
                    ci.lat,
                    ci.lon,
                    mv.province,
                    mv.city,
                    mv.date,
                    concat(mv.uid, '_', mv.date) AS trajectory_id
                FROM {src_table} mv
                LEFT JOIN cell_latest ci ON mv.cid = ci.cid
            ),
            windowed AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY uid, date ORDER BY stime) AS index,
                    LEAD(stime) OVER (PARTITION BY uid, date ORDER BY stime) AS next_stime,
                    LEAD(lat)   OVER (PARTITION BY uid, date ORDER BY stime) AS next_lat,
                    LEAD(lon)   OVER (PARTITION BY uid, date ORDER BY stime) AS next_lon,
                    FIRST_VALUE(stime) OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_stime,
                    LAST_VALUE(stime)  OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_stime,
                    FIRST_VALUE(lat)   OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_lat,
                    FIRST_VALUE(lon)   OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_lon,
                    LAST_VALUE(lat)    OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_lat,
                    LAST_VALUE(lon)    OVER (PARTITION BY uid, date ORDER BY stime
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_lon
                FROM joined
            )
            SELECT
                uid,
                index,
                stime,
                cid,
                lat,
                lon,
                province,
                city,
                date,
                UNIX_TIMESTAMP(next_stime) - UNIX_TIMESTAMP(stime) AS time_diff,
                2 * 6371000 * asin(sqrt(
                    pow(sin((radians(next_lat) - radians(lat)) / 2), 2) +
                    cos(radians(lat)) * cos(radians(next_lat)) *
                    pow(sin((radians(next_lon) - radians(lon)) / 2), 2)
                )) AS space_diff,
                UNIX_TIMESTAMP(last_stime) - UNIX_TIMESTAMP(first_stime) AS traj_time,
                2 * 6371000 * asin(sqrt(
                    pow(sin((radians(last_lat) - radians(first_lat)) / 2), 2) +
                    cos(radians(first_lat)) * cos(radians(last_lat)) *
                    pow(sin((radians(last_lon) - radians(first_lon)) / 2), 2)
                )) AS traj_space,
                trajectory_id
            FROM windowed
        """
        return self.__session.sql(query)

    def run_calculate(self, src_table='move_vp', cell_info_table='cell_info',
                      out_total='total_time_space_diff', out_summary='time_space_diff'):
        """
        计算时空特征并保存两张结果表:
        - out_total:   完整记录表（含所有字段）
        - out_summary: 轨迹汇总表（trajectory_id 为粒度）
        """
        df_total = self.process_all(src_table=src_table, cell_info_table=cell_info_table)
        df_total.write.mode("overwrite").saveAsTable(out_total)
        print(f"已保存到表: {out_total}，记录数: {df_total.count()}")

        df_summary = self.__session.sql(f"""
            SELECT
                trajectory_id,
                SUM(time_diff)  AS time_diff,
                SUM(space_diff) AS space_diff
            FROM {out_total}
            GROUP BY trajectory_id
        """)
        df_summary.write.mode("overwrite").saveAsTable(out_summary)
        print(f"已保存到表: {out_summary}，记录数: {df_summary.count()}")

# =========================================================

if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_calculate(
            src_table="move_vp",
            cell_info_table="cell_info",
            out_total="total_time_space_diff",
            out_summary="time_space_diff"
        )
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster