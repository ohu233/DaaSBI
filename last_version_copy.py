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
        
        # 将经纬度投影为 x,y（当前步和下一步）
        df = (
            df
            .withColumn("x", self.proj_x(F.col("lon"), F.col("lat")))
            .withColumn("y", self.proj_y(F.col("lon"), F.col("lat")))
            .withColumn("next_x", self.proj_x(F.col("next_lon"), F.col("next_lat")))
            .withColumn("next_y", self.proj_y(F.col("next_lon"), F.col("next_lat")))
            # 当前点网格
            .withColumn(
                "loc_x",
                F.when(
                    (F.col("x") >= F.lit(X_MIN)) & (F.col("x") <= F.lit(X_MAX)),
                    F.floor((F.col("x") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            .withColumn(
                "loc_y",
                F.when(
                    (F.col("y") >= F.lit(Y_MIN)) & (F.col("y") <= F.lit(Y_MAX)),
                    F.floor((F.col("y") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            # 下一点网格
            .withColumn(
                "next_loc_x",
                F.when(
                    (F.col("next_x") >= F.lit(X_MIN)) & (F.col("next_x") <= F.lit(X_MAX)),
                    F.floor((F.col("next_x") - F.lit(X_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
            .withColumn(
                "next_loc_y",
                F.when(
                    (F.col("next_y") >= F.lit(Y_MIN)) & (F.col("next_y") <= F.lit(Y_MAX)),
                    F.floor((F.col("next_y") - F.lit(Y_MIN)) / F.lit(GRID_SIZE)).cast("int")
                )
            )
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