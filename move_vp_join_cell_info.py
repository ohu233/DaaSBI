from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.window import Window
import pandas as pd
from pyproj import CRS, Transformer
import numpy as np
'''
历史文件：仅作为参考
'''
# ----------------------------
# 投影与网格参数（当前：三省一市）
# 三省一市
# x_min, x_max =853241.740909714,1678241.740909714
# y_min, y_max =3012170.012633881,3956170.012633881
#
# 江苏省
# x_min, x_max =988144.781509014,1552144.781509014
# y_min, y_max =3417504.294282121,3946504.294282121
# ----------------------------
X_MIN, X_MAX = 853241.740909714, 1678241.740909714
Y_MIN, Y_MAX = 3012170.012633881, 3956170.012633881
GRID_SIZE = 1000  # 1km × 1km

# 坐标系
WGS84_WKT = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
BJ54_3DEG_CM111_WKT = 'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'

# ---------
# 懒加载 Transformer
# ---------
def _get_transformer():
    if not hasattr(_get_transformer, "_tfm"):
        wgs84 = CRS.from_wkt(WGS84_WKT)
        bj54 = CRS.from_wkt(BJ54_3DEG_CM111_WKT)
        _get_transformer._tfm = Transformer.from_crs(wgs84, bj54, always_xy=True)
    return _get_transformer._tfm

# ---------
# Pandas UDF：批量投影 (lon, lat) -> struct<x:double, y:double>
# ---------
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
            # 开启 Arrow
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        session.sql(f"USE {db}")
        self.__session = session

        # 在 SparkSession 建好后再注册 Pandas UDF，避免 _jvm 为空
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

        # 挂到实例上，供后续调用
        self.proj_x = proj_x
        self.proj_y = proj_y

    def stop(self):
        self.__session.stop()

    def query_move_route(self, limit_uids=20):
        """
        先 DISTINCT，再按 uid 升序取前 N
        """
        # 先拿前 N 个 uid 并收集到 driver
        uid_df = self.__session.sql(f"SELECT DISTINCT uid FROM move_rn ORDER BY uid LIMIT {limit_uids}")
        uids = [r.uid for r in uid_df.collect()]
        if not uids:
            return self.__session.createDataFrame([], schema="""
                uid STRING, move_id INT, move_vp_id BIGINT, stime TIMESTAMP, grid_id STRING,
                cid BIGINT, province STRING, city STRING, date INT, time STRING,
                province STRING, city STRING, date INT,
                cid_r BIGINT, ctype STRING, zone_id STRING, 
                province_r STRING, city_r STRING, date_r INT,
                lat DOUBLE, lon DOUBLE
            """)

        uid_condition = "'" + "','".join(map(str, uids)) + "'"

        # 直接 WHERE uid IN (...)，并与 route_node / route_node r_next 做 LEFT JOIN
        # 若采用广播会timeout
        query = f"""
            SELECT
                m.uid,
                m.move_id,
                m.move_vp_id,
                m.stime,
                m.grid_id,
                m.cid,
                m.province,
                m.city,
                m.date,
                r.cid,     AS cid_r,
                r.ctype    AS ctype,
                r.zone_id  AS zone_id,
                r.province AS province_r,
                r.city     AS city_r,
                r.date     AS date_r,
                r.lat      AS lat,
                r.lon      AS lon
            FROM move_vp m
            LEFT JOIN cell_info r
                ON m.cid = r.cid
            WHERE m.uid IN ({uid_condition})
        """
        return self.__session.sql(query)

    def preprocess_and_save(self, df, table_name= 'move_vp_join_cell_info'):
        """
        Pandas UDF 进行投影
        网格编码写回 Hive
        """
        if df is None:
            raise RuntimeError("Input DataFrame is None")

        # 当前节点批量投影
        with_proj = (
            df
            .withColumn("x", self.proj_x(F.col("lon"), F.col("lat")))
            .withColumn("y", self.proj_y(F.col("lon"), F.col("lat")))
        )

        # 网格分箱
        loc_x_expr = F.when(
            (F.col("x") >= F.lit(X_MIN)) & (F.col("x") < F.lit(X_MAX)),
            F.floor((F.col("x") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
        )
        loc_y_expr = F.when(
            (F.col("y") >= F.lit(Y_MIN)) & (F.col("y") < F.lit(Y_MAX)),
            F.floor((F.col("y") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
        )

        with_grid = (
            with_proj
            .withColumn("loc_x", loc_x_expr)
            .withColumn("loc_y", loc_y_expr)
        )

        # 下一节点批量投影
        with_next_proj = (
            with_grid
            .withColumn("next_x", self.proj_x(F.col("next_lon"), F.col("next_lat")))
            .withColumn("next_y", self.proj_y(F.col("next_lon"), F.col("next_lat")))
        )

        # 下一节点网格
        next_loc_x_expr = F.when(
            (F.col("next_x") >= F.lit(X_MIN)) & (F.col("next_x") < F.lit(X_MAX)),
            F.floor((F.col("next_x") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
        )
        next_loc_y_expr = F.when(
            (F.col("next_y") >= F.lit(Y_MIN)) & (F.col("next_y") < F.lit(Y_MAX)),
            F.floor((F.col("next_y") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
        )

        with_next_grid = (
            with_next_proj
            .withColumn("next_loc_x", next_loc_x_expr)
            .withColumn("next_loc_y", next_loc_y_expr)
        )
        
        # 添加 Haversine 距离计算
        R = 6371000.0  # 地球半径 (米)

        lon1 = F.radians(F.col("lon"))
        lat1 = F.radians(F.col("lat"))
        lon2 = F.radians(F.col("next_lon"))
        lat2 = F.radians(F.col("next_lat"))

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = F.pow(F.sin(dlat / 2), 2) + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / 2), 2)
        c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
        distance = R * c

        final_df = (
            with_next_grid
            .withColumn("distance", distance)
        )

        # 落表
        final_df.write.mode("overwrite").saveAsTable(table_name)

    def run(self, limit_uids=20, out_table='move_vp_join_cell_info'):
        df = self.query_move_route(limit_uids=limit_uids)
        self.preprocess_and_save(df, table_name=out_table)
'''
# =========================================================
    # 生成 O-D 表 
    # TODO：生成的od表有问题（time diff计算错误），按理说相同uid和moi_id的rnseq唯一，实在不行用time排序
    # 解决办法：直接转化成pandas dataframe 目前未改动
    def build_od_table(self, source_table='preprocessed_v5', od_table='test_dataset_v4'):
        src = self.__session.table(source_table)

        # 定义窗口：按 uid, moi_id 分区，按 rn_seq 排序
        window_spec = Window.partitionBy("uid", "moi_id").orderBy("rn_seq")

        # 使用窗口函数获取下一行的时间戳并计算时间差
        time_diff_df = (
            src
            .withColumn("next_time", F.lead("time").over(window_spec))
            .withColumn("time_diff", F.col("next_time").cast("long") - F.col("time").cast("long"))
        )

        # 过滤掉 next_rn_id 为-1的行
        time_diff_df = time_diff_df.filter(F.col("next_rn_id") != '-1')
        
        # 直接使用预处理表中已计算好的距离，无需重复计算
        od_df = (
            time_diff_df
            .select(
                F.col("uid"),
                F.col("mode"),
                F.col("loc_x").alias("locx_o"),
                F.col("loc_y").alias("locy_o"),
                F.col("next_loc_x").alias("locx_d"),
                F.col("next_loc_y").alias("locy_d"),
                F.col("time").alias("time"),
                F.col("next_time"),
                F.col("time_diff"),
                F.col("distance")  # 直接使用预处理表中的距离字段
            )
        )

        od_df.write.mode("overwrite").saveAsTable(od_table)
'''
# =========================================================

if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run(limit_uids=20, out_table="preprocessed_v5")
        table.build_od_table(source_table="preprocessed_v5", od_table="test_dataset_v4")
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster