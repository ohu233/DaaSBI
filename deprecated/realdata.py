from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.window import Window
import pandas as pd
from pyproj import CRS, Transformer
import numpy as np
'''
用于生成包含当前步和下一步信息的数据集，使用哈希生成 trajectory_id
下一步：将cid文件进行去重处理，与hash后的表格的cid字段进行关联，进而得到lon，lat字段，最后进行投影转换，得到x,y字段
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

    def process_move_vp_with_next(self):
        """
        TODO: 映射xy有问题
        从intermediate_dataset4RL_hash_2读取数据，关联cell_info获取经纬度信息
        """
        query = """
            WITH cell_info_dedup AS (
                SELECT cid, lon, lat
                FROM (
                    SELECT 
                        cid, lon, lat, date,
                        ROW_NUMBER() OVER (PARTITION BY cid ORDER BY date DESC) AS rn
                    FROM cell_info
                ) tmp
                WHERE rn = 1
            )
            SELECT 
                t.*,
                c1.lon,
                c1.lat,
                c2.lon AS next_lon,
                c2.lat AS next_lat
            FROM intermediate_dataset4RL_hash_2 t
            LEFT JOIN cell_info_dedup c1 ON t.cid = c1.cid
            LEFT JOIN cell_info_dedup c2 ON t.next_cid = c2.cid
            ORDER BY t.trajectory_id, t.move_id, t.move_vp_id
        """
        return self.__session.sql(query)

    def run_dataset4rl(self, out_table='dataset4RL'):
        """
        生成包含当前步和下一步信息及经纬度的数据集
        """
        df = self.process_move_vp_with_next()

        # 基于经纬度计算前后节点球面距离（米）
        earth_radius_m = 6371000.0
        dlat = F.radians(F.col("next_lat") - F.col("lat"))
        dlon = F.radians(F.col("next_lon") - F.col("lon"))
        a = (
            F.pow(F.sin(dlat / F.lit(2.0)), 2)
            + F.cos(F.radians(F.col("lat")))
            * F.cos(F.radians(F.col("next_lat")))
            * F.pow(F.sin(dlon / F.lit(2.0)), 2)
        )
        haversine_distance_m = F.lit(2.0 * earth_radius_m) * F.asin(F.least(F.lit(1.0), F.sqrt(a)))

        in_curr_grid = (
            (F.col("_x_m") >= F.lit(X_MIN)) & (F.col("_x_m") <= F.lit(X_MAX)) &
            (F.col("_y_m") >= F.lit(Y_MIN)) & (F.col("_y_m") <= F.lit(Y_MAX))
        )
        in_next_grid = (
            (F.col("_next_x_m") >= F.lit(X_MIN)) & (F.col("_next_x_m") <= F.lit(X_MAX)) &
            (F.col("_next_y_m") >= F.lit(Y_MIN)) & (F.col("_next_y_m") <= F.lit(Y_MAX))
        )
        
        # 先投影到米，再转换为网格整数坐标：((value - min) / 1000) 向下取整
        df = (
            df
            .withColumn("_x_m", self.proj_x(F.col("lon"), F.col("lat")))
            .withColumn("_y_m", self.proj_y(F.col("lon"), F.col("lat")))
            .withColumn("_next_x_m", self.proj_x(F.col("next_lon"), F.col("next_lat")))
            .withColumn("_next_y_m", self.proj_y(F.col("next_lon"), F.col("next_lat")))
            .withColumn(
                "x",
                F.when(
                    in_curr_grid,
                    F.floor((F.col("_x_m") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            .withColumn(
                "y",
                F.when(
                    in_curr_grid,
                    F.floor((F.col("_y_m") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            .withColumn(
                "next_x",
                F.when(
                    in_next_grid,
                    F.floor((F.col("_next_x_m") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            .withColumn(
                "next_y",
                F.when(
                    in_next_grid,
                    F.floor((F.col("_next_y_m") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            # 当前点网格
            .withColumn("locx_o", F.col("x"))
            .withColumn("locy_o", F.col("y"))
            # 下一点网格
            .withColumn("locx_d", F.col("next_x"))
            .withColumn("locy_d", F.col("next_y"))
            # 前后节点时间差（秒）
            .withColumn(
                "time",
                (
                    F.unix_timestamp(F.col("next_stime").cast("timestamp"))
                    - F.unix_timestamp(F.col("stime").cast("timestamp"))
                ).cast("long")
            )
            # 前后节点经纬度距离（米）
            .withColumn(
                "distance",
                F.when(
                    F.col("lon").isNotNull() & F.col("lat").isNotNull() &
                    F.col("next_lon").isNotNull() & F.col("next_lat").isNotNull(),
                    haversine_distance_m
                )
            )
            .drop("_x_m", "_y_m", "_next_x_m", "_next_y_m")
        )
        
        # 保存结果
        df.write.mode("overwrite").saveAsTable(out_table)
        
        print(f"数据已保存到表: {out_table}")
        print(f"总记录数: {df.count()}")

# =========================================================

if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_dataset4rl(out_table="dataset_hash_with_lonlat_xy")
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster