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
3. 按照date对数据进行切分，切分成每天的数据表并保存
4. 对每个数据表按uid分组，分组后按stime排序，每个uid内部生成index索引
5. 最终生成的字段：uid，index，stime，cid，lat，lon，province，city，date
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

    def process_by_date(self, src_table='move_vp', cell_info_table='cell_info', date=None):
        """
        对指定日期的数据进行处理:
        1. cell_info 去重：相同 cid 仅保留 date 最大的记录
        2. move_vp 与去重后的 cell_info 按 cid 关联，获取经纬度
        3. 按 uid 分组，组内按 stime 排序，生成 index 索引
        4. 最终字段：uid, index, stime, cid, province, city, date, lat, lon
        """
        where_clause = f"AND mv.date = '{date}'" if date else ""
        query = f"""
            WITH cell_latest AS (
                SELECT cid, lat, lon
                FROM (
                    SELECT cid, lat, lon,
                           ROW_NUMBER() OVER (PARTITION BY cid ORDER BY date DESC) AS rn
                    FROM {cell_info_table}
                ) t
                WHERE rn = 1
            )
            SELECT
                mv.uid,
                ROW_NUMBER() OVER (PARTITION BY mv.uid ORDER BY mv.stime) AS index,
                mv.stime,
                mv.cid,
                ci.lat,
                ci.lon,
                mv.province,
                mv.city,
                mv.date
            FROM {src_table} mv
            LEFT JOIN cell_latest ci ON mv.cid = ci.cid
            WHERE 1=1 {where_clause}
        """
        return self.__session.sql(query)

    def run_preprocess(self, src_table='move_vp', cell_info_table='cell_info', out_table_prefix='dataset'):
        """
        按 date 切分数据，每天保存为独立的表
        """
        dates = [
            row['date']
            for row in self.__session.sql(
                f"SELECT DISTINCT date FROM {src_table} ORDER BY date"
            ).collect()
        ]
        for date in dates:
            df = self.process_by_date(src_table=src_table, cell_info_table=cell_info_table, date=date)
            out_table = f"{out_table_prefix}_{str(date).replace('-', '')}"
            df.write.mode("overwrite").saveAsTable(out_table)
            print(f"[{date}] 已保存到表: {out_table}，记录数: {df.count()}")

# =========================================================

if __name__ == "__main__":
    table = HiveTable(db="ss_seu_df")
    try:
        table.run_preprocess(src_table="move_vp", cell_info_table="cell_info", out_table_prefix="dataset")
    finally:
        table.stop()

# spark-submit: --master yarn --deploy-mode cluster