"""将 hex_grid_NanJing.pkl 转换为 CSV 文件"""
import pickle
import csv
import os

pkl_path = os.path.join(os.path.dirname(__file__), "hex_grid_NanJing.pkl")
csv_path = os.path.join(os.path.dirname(__file__), "hex_grid_NanJing.csv")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["key_0", "key_1", "key_2", "lon", "lat", "code"])
    for k, v in data.items():
        writer.writerow([k[0], k[1], k[2], v["lon"], v["lat"], v["code"]])

print(f"已写入 {len(data)} 行到 {csv_path}")
