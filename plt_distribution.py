import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/multicity_dataset/dataset_multicity.csv')

# 剔除任意数据为0的行
df = df[(df['time_value'] != 0) & (df['dist_value'] != 0) & (df['velocity'] != 0)]

# 筛选0-90分位数据并绘制分布图
for col, color, title in [
    ('time_value', 'blue', 'Distribution of Time Value (0-90th percentile)'),
    ('dist_value', 'green', 'Distribution of Distance Value (0-90th percentile)'),
]:
    p90 = df[col].quantile(0.90)
    filtered = df[df[col] <= p90][col]

    plt.figure(figsize=(8, 5))
    plt.hist(filtered, bins=50, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()