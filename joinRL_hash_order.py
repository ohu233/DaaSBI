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
        对 move_vp 表进行处理:
        1. 按 (uid, move_id, date) 分组，生成唯一的 trajectory_id（使用哈希）
        2. 每组内按 move_vp_id 排序
        3. 将下一行的信息复制到当前行
        """
        query = """
        WITH base AS (
            SELECT
                trajectory_id,
                uid,
                move_id,
                move_vp_id,
                stime,
                grid_id,
                cid,
                province,
                city,
                date,
                x,
                y
            FROM dataset_hash_with_lonlat_xy
            WHERE x IS NOT NULL AND y IS NOT NULL
        ),

        point_rank AS (
            SELECT
                trajectory_id,
                move_vp_id,
                x,
                y,
                ROW_NUMBER() OVER (
                    PARTITION BY trajectory_id
                    ORDER BY move_vp_id ASC
                ) AS rn_first,
                ROW_NUMBER() OVER (
                    PARTITION BY trajectory_id
                    ORDER BY move_vp_id DESC
                ) AS rn_last
            FROM base
        ),

        first_point AS (
            SELECT
                trajectory_id,
                x AS first_x,
                y AS first_y
            FROM point_rank
            WHERE rn_first = 1
        ),

        last_point AS (
            SELECT
                trajectory_id,
                x AS last_x,
                y AS last_y
            FROM point_rank
            WHERE rn_last = 1
        ),

        traj_distance AS (
            SELECT
                f.trajectory_id,
                SQRT(POW(l.last_x - f.first_x, 2) + POW(l.last_y - f.first_y, 2)) AS distance
            FROM first_point f
            JOIN last_point l
                ON f.trajectory_id = l.trajectory_id
        ),

        traj_order AS (
            SELECT
                trajectory_id,
                distance,
                ROW_NUMBER() OVER (
                    ORDER BY distance DESC, trajectory_id
                ) AS traj_rank
            FROM traj_distance
        )

        SELECT
            b.*,
            o.distance,
            o.traj_rank
        FROM base b
        JOIN traj_order o
            ON b.trajectory_id = o.trajectory_id
        ORDER BY o.traj_rank ASC, b.move_vp_id ASC
        """
        return self.__session.sql(query)

    def run_dataset4rl(self, out_table='dataset4RL'):
        """
        生成包含当前步和下一步信息的数据集
        """
        df = self.process_move_vp_with_next()

        # 保存结果
        df.write.mode("overwrite").saveAsTable(out_table)
        
        print(f"数据已保存到表: {out_table}")
        print(f"总记录数: {df.count()}")

# =========================================================

if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_dataset4rl(out_table="intermediate_dataset4RL_hash_sorted")
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster