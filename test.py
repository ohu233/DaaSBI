from pyproj import CRS, Transformer
import math

# 使用你代码中的坐标系定义和网格参数
WGS84_WKT = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
BJ54_3DEG_CM111_WKT = 'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'

# 网格参数
X_MIN, X_MAX = 853241.740909714, 1678241.740909714
Y_MIN, Y_MAX = 3012170.012633881, 3956170.012633881
GRID_SIZE = 1000  # 1km × 1km

# 上海人民广场坐标
shanghai_lon = 121.48
shanghai_lat = 31.23

print(f"上海人民广场原始坐标: 经度={shanghai_lon}°, 纬度={shanghai_lat}°")

# 创建坐标转换器
wgs84 = CRS.from_wkt(WGS84_WKT)
bj54 = CRS.from_wkt(BJ54_3DEG_CM111_WKT)
transformer = Transformer.from_crs(wgs84, bj54, always_xy=True)

# 投影转换
x, y = transformer.transform(shanghai_lon, shanghai_lat)
print(f"投影后坐标: x={x:.2f}m, y={y:.2f}m")

# 检查是否在网格范围内
in_x_range = X_MIN <= x < X_MAX
in_y_range = Y_MIN <= y < Y_MAX
print(f"是否在X范围内: {in_x_range} ({X_MIN:.2f} <= {x:.2f} < {X_MAX:.2f})")
print(f"是否在Y范围内: {in_y_range} ({Y_MIN:.2f} <= {y:.2f} < {Y_MAX:.2f})")

# 计算网格坐标（按照你的代码逻辑）
if in_x_range:
    loc_x = int(math.floor((x - X_MIN) / GRID_SIZE))
else:
    loc_x = None
    
if in_y_range:
    loc_y = int(math.floor((y - Y_MIN) / GRID_SIZE))
else:
    loc_y = None

print(f"网格坐标: loc_x={loc_x}, loc_y={loc_y}")

# 如果在范围内，计算网格中心点坐标
if loc_x is not None and loc_y is not None:
    grid_center_x = X_MIN + (loc_x + 0.5) * GRID_SIZE
    grid_center_y = Y_MIN + (loc_y + 0.5) * GRID_SIZE
    print(f"所在网格中心点投影坐标: x={grid_center_x:.2f}m, y={grid_center_y:.2f}m")
    
    # 转换回经纬度
    reverse_transformer = Transformer.from_crs(bj54, wgs84, always_xy=True)
    center_lon, center_lat = reverse_transformer.transform(grid_center_x, grid_center_y)
    print(f"网格中心点经纬度: 经度={center_lon:.6f}°, 纬度={center_lat:.6f}°")
else:
    print("该坐标不在定义的网格范围内")