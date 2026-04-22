from wgs84tobj1954 import wgs84_to_beijing1954, assign_grid_cell
import pandas as pd

# 使用上海人民广场坐标进行测试
shanghai_lon = 121.48
shanghai_lat = 31.23

print(f"测试坐标: 上海人民广场")
print(f"经度: {shanghai_lon}°, 纬度: {shanghai_lat}°")
print("-" * 50)

# 1. 测试坐标转换
try:
    x, y = wgs84_to_beijing1954(shanghai_lon, shanghai_lat)
    print(f"投影后坐标:")
    print(f"X: {x:.3f}m")
    print(f"Y: {y:.3f}m")
except Exception as e:
    print(f"坐标转换出错: {e}")
    exit()

# 2. 使用你代码中定义的三省一市网格范围（和Process_on_DASSBI.py一致）
x_min, x_max = 853241.740909714, 1678241.740909714
y_min, y_max = 3012170.012633881, 3956170.012633881
grid_size = 1000

print(f"\n使用三省一市网格范围:")
print(f"X范围: {x_min:.3f} - {x_max:.3f}")
print(f"Y范围: {y_min:.3f} - {y_max:.3f}")
print(f"网格大小: {grid_size}m × {grid_size}m")

# 3. 测试网格分配
try:
    loc_x, loc_y = assign_grid_cell(x, y, x_min, x_max, y_min, y_max, grid_size)
    print(f"\n网格坐标: loc_x={loc_x}, loc_y={loc_y}")
    
    if loc_x is None or loc_y is None:
        print("⚠️  上海坐标不在定义的网格范围内")
        print(f"坐标是否在X范围内: {x_min <= x <= x_max}")
        print(f"坐标是否在Y范围内: {y_min <= y <= y_max}")
    else:
        print("✓ 上海坐标在网格范围内")
        
        # 计算网格中心点坐标
        center_x = x_min + (loc_x + 0.5) * grid_size
        center_y = y_min + (loc_y + 0.5) * grid_size
        print(f"所在网格中心点: X={center_x:.3f}m, Y={center_y:.3f}m")
        
        # 计算网格的四个角点
        grid_x_start = x_min + loc_x * grid_size
        grid_x_end = x_min + (loc_x + 1) * grid_size
        grid_y_start = y_min + loc_y * grid_size
        grid_y_end = y_min + (loc_y + 1) * grid_size
        
        print(f"网格边界:")
        print(f"  X: {grid_x_start:.3f}m - {grid_x_end:.3f}m")
        print(f"  Y: {grid_y_start:.3f}m - {grid_y_end:.3f}m")

except Exception as e:
    print(f"网格分配出错: {e}")

# 4. 测试和Process_on_DASSBI.py的一致性
print(f"\n" + "="*60)
print("与 Process_on_DASSBI.py 的计算结果对比:")

# 模拟Process_on_DASSBI.py中的计算逻辑
import math

# 检查坐标是否在范围内（Process_on_DASSBI.py的逻辑）
in_x_range = (x >= x_min) and (x < x_max)
in_y_range = (y >= y_min) and (y < y_max)

if in_x_range:
    process_loc_x = int(math.floor((x - x_min) / grid_size))
else:
    process_loc_x = None

if in_y_range:
    process_loc_y = int(math.floor((y - y_min) / grid_size))
else:
    process_loc_y = None

print(f"wgs84tobj1954.py 结果: loc_x={loc_x}, loc_y={loc_y}")
print(f"Process_on_DASSBI.py 逻辑: loc_x={process_loc_x}, loc_y={process_loc_y}")

if loc_x == process_loc_x and loc_y == process_loc_y:
    print("✓ 两种方法计算结果一致")
else:
    print("⚠️ 两种方法计算结果不一致，需要检查差异")
    
# 5. 创建测试数据验证完整流程
print(f"\n" + "="*60)
print("测试多个城市坐标:")

test_cities = [
    {'name': '上海人民广场', 'lat': 31.23, 'lon': 121.48},
    {'name': '南京市中心', 'lat': 32.04, 'lon': 118.78},
    {'name': '杭州市中心', 'lat': 30.25, 'lon': 120.17},
    {'name': '合肥市中心', 'lat': 31.86, 'lon': 117.27},
]

results = []
for city in test_cities:
    try:
        x_coord, y_coord = wgs84_to_beijing1954(city['lon'], city['lat'])
        loc_x_coord, loc_y_coord = assign_grid_cell(x_coord, y_coord, x_min, x_max, y_min, y_max, grid_size)
        
        results.append({
            'name': city['name'],
            'lat': city['lat'],
            'lon': city['lon'],
            'x': round(x_coord, 3),
            'y': round(y_coord, 3),
            'loc_x': loc_x_coord,
            'loc_y': loc_y_coord,
            'in_range': loc_x_coord is not None and loc_y_coord is not None
        })
        
    except Exception as e:
        print(f"处理 {city['name']} 时出错: {e}")

# 显示结果
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))