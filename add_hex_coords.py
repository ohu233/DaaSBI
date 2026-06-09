import pickle
import pandas as pd
import numpy as np
from pyproj import Transformer

# 加载道路编码字典 {(x,y,z): {"lon","lat","code"}}
with open("data\hex_grid.pkl", "rb") as f:
    grid = pickle.load(f)
print(f"加载 pkl, {len(grid):,} 个栅格")

# 与 query_hex.py 完全一致的参数
SIDE = 200.0
SQRT3 = np.sqrt(3)

to_bj = Transformer.from_crs("EPSG:4326", "EPSG:2434", always_xy=True)
origin_x, origin_y = to_bj.transform(114.662853, 27.441905)


def lonlat_to_hex(lon, lat):
    px, py = to_bj.transform(lon, lat)
    cx = px - origin_x
    cy = py - origin_y

    q = (cx * 2.0 / 3.0) / SIDE
    r = (-cx / 3.0 + SQRT3 / 3.0 * cy) / SIDE

    s = -q - r
    qi = round(q)
    ri = round(r)
    si = round(s)

    qd = abs(qi - q)
    rd = abs(ri - r)
    sd = abs(si - s)

    if qd > rd and qd > sd:
        qi = -ri - si
    elif rd > sd:
        ri = -qi - si
    else:
        si = -qi - ri

    return qi, si, ri


# 读取数据
src = "data\multicity_dataset\dataset_multicity.csv"
df = pd.read_csv(src)
print(f"读取 {len(df)} 条记录")

# 批量计算六边形坐标，同时查 pkl 获取 code
hex_x, hex_y, hex_z, codes = [], [], [], []
hit = 0
for lon, lat in zip(df["lon"], df["lat"]):
    x, y, z = lonlat_to_hex(lon, lat)
    hex_x.append(x)
    hex_y.append(y)
    hex_z.append(z)
    val = grid.get((x, y, z))
    if val is not None:
        codes.append(val["code"])
        hit += 1
    else:
        codes.append(-1)  # 不在范围内的标记

df["hex_x"] = hex_x
df["hex_y"] = hex_y
df["hex_z"] = hex_z
df["hex_code"] = codes

# 输出
out = "data\dataset_multicity_with_hex.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"命中: {hit}/{len(df)} ({hit/len(df)*100:.1f}%)")
print(f"已保存 {len(df)} 条记录到 {out}")
print(f"新增列: hex_x, hex_y, hex_z, hex_code")
print(f"坐标范围: x [{df.hex_x.min()}, {df.hex_x.max()}], "
      f"y [{df.hex_y.min()}, {df.hex_y.max()}], "
      f"z [{df.hex_z.min()}, {df.hex_z.max()}]")
print(f"hex_code 分布: {df.hex_code.value_counts().to_dict()}")
